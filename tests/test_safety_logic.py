"""Unit tests for the safety and prediction logic. No CARLA, Ditto or broker needed."""
import json
import math

import numpy as np
import pytest
from central_gpu_processor import (
    Box,
    CentroidTracker,
    SafetyEngine,
    Track,
    apply_event,
    approaching_from_behind,
    deceleration_to_avoid_crash,
    fuse,
    lidar_to_model_frame,
    model_box_to_world,
    overhead_marker,
    parse_lidar,
    path_check_applies,
    path_time_to_collision,
    person_track,
    post_encroachment_time,
    thing_to_track,
    time_to_collision,
    ttc_check_applies,
)
from physical_ml_subscriber import SafetyGateSubscriber, SimMotors, parse_gate_event
from waymo_predictor_node import (
    HISTORY_FRAMES,
    LEFT,
    RIGHT,
    STRAIGHT,
    HistoryBuffer,
    along_polyline,
    curve_path,
    ego_to_world,
    history_to_ego,
    measured_velocity,
    pick_branch,
    predict_world_paths,
    turn_intent,
    turn_rate,
    world_to_ego,
)


# --- TTC / PET / DRAC ---
def test_ttc_head_on():
    a, b = Track("a", 0, 0, 5, 0), Track("b", 22.5, 0, -5, 0)
    assert time_to_collision(a, b, radius=2.5) == pytest.approx(2.0)


def test_ttc_crossing_paths():
    a, b = Track("a", -10, 0, 5, 0), Track("b", 0, -10, 0, 5)
    # Both reach the origin at t=2 s; the 2.5 m contact circle is reached slightly earlier.
    assert time_to_collision(a, b, radius=2.5) == pytest.approx(2 - 2.5 / (5 * math.sqrt(2)))


@pytest.mark.parametrize("a,b", [
    (Track("a", 0, 0, -5, 0), Track("b", 20, 0, 5, 0)),   # diverging
    (Track("a", 0, 0, 5, 0), Track("b", 20, 0, 5, 0)),    # same velocity
    (Track("a", 0, 0, 5, 0), Track("b", 20, 50, -5, 0)),  # passing wide
])
def test_ttc_is_infinite_without_a_conflict(a, b):
    assert time_to_collision(a, b) == math.inf


def test_ttc_is_zero_in_contact():
    assert time_to_collision(Track("a", 0, 0), Track("b", 1, 0), radius=2.5) == 0.0


def test_pet_is_the_arrival_gap_at_the_crossing_point():
    a, b = Track("a", -10, 0, 5, 0), Track("b", 0, -20, 0, 5)
    assert post_encroachment_time(a, b) == pytest.approx(2.0)


@pytest.mark.parametrize("a,b", [
    (Track("a", 0, 0, 5, 0), Track("b", 0, 5, 5, 0)),     # parallel
    (Track("a", -10, 0, -5, 0), Track("b", 0, -10, 0, 5)),  # crossing point is behind a
])
def test_pet_is_infinite_without_a_crossing(a, b):
    assert post_encroachment_time(a, b) == math.inf


def straight_path(x, y, vx, vy, steps=30):
    times = 0.1 * np.arange(1, steps + 1)
    return np.stack((x + vx * times, y + vy * times), axis=-1)


def test_path_ttc_finds_the_first_moment_the_paths_meet():
    a = Track("a", 0, 0, path=straight_path(0, 0, 5, 0))
    b = Track("b", 20, 0, path=straight_path(20, 0, -5, 0))
    # Closing at 10 m/s from 20 m apart, within 2.5 m at 1.75 s; paths are sampled every 0.1 s
    assert path_time_to_collision(a, b, radius=2.5) == pytest.approx(1.8)


def test_path_ttc_ignores_paths_that_cross_at_different_times():
    a = Track("a", 0, 0, path=straight_path(0, 0, 5, 0))
    b = Track("b", 5, -100, path=straight_path(5, -100, 0, 5))
    assert path_time_to_collision(a, b) == math.inf


def test_path_ttc_needs_both_paths():
    a = Track("a", 0, 0, path=straight_path(0, 0, 5, 0))
    assert path_time_to_collision(a, Track("b", 1, 0)) == math.inf


def test_drac():
    a, b = Track("a", 0, 0, 10, 0), Track("b", 12.5, 0)
    assert deceleration_to_avoid_crash(a, b, radius=2.5) == pytest.approx(5.0)
    assert deceleration_to_avoid_crash(Track("a", 0, 0, -1, 0), b) == 0.0


# --- Fusion ---
def test_fuse_keeps_identity_and_telemetry_velocity():
    fused = fuse([Track("vehicle_ml", 0, 0, 3, 0)], [Track("vision:0", 1, 0, 9, 9)])
    assert len(fused) == 1
    assert fused[0].id == "vehicle_ml"
    assert (fused[0].vx, fused[0].vy) == (3, 0)
    assert 0 < fused[0].x < 1


def test_fuse_keeps_unreporting_actors():
    far = Track("vision:0", 50, 50)
    assert fuse([Track("vehicle_ml", 0, 0)], [far])[1] is far


def test_fuse_matches_a_named_actor_by_id_beyond_the_gate():
    fused = fuse([Track("vehicle_ml", 0, 0, 3, 0)], [Track("vehicle_ml", 50, 0)])
    assert [t.id for t in fused] == ["vehicle_ml"]


def test_fuse_takes_detected_velocity_when_telemetry_reports_none():
    fused = fuse([Track("vehicle_ml", 0, 0)], [Track("vehicle_ml", 1, 0, 4, 2)])
    assert (fused[0].vx, fused[0].vy) == (4, 2)


def test_fuse_keeps_the_detections_predicted_path():
    path = straight_path(0, 0, 5, 0)
    fused = fuse([Track("vehicle_ml", 0, 0, 5, 0)], [Track("vehicle_ml", 0, 0, path=path)])
    assert fused[0].path is path


def test_fuse_uses_each_detection_once():
    detection = Track("vision:0", 1, 0)
    assert len(fuse([Track("a", 0, 0), Track("b", 2, 0)], [detection])) == 2


def test_tracker_keeps_id_and_estimates_velocity():
    tracker = CentroidTracker()
    first = tracker.update([(0.0, 0.0)], 0.0)[0]
    second = tracker.update([(1.0, 0.0)], 0.5)[0]
    assert second.id == first.id
    assert second.vx == pytest.approx(2.0)


def test_tracker_smooths_jitter_into_a_steady_velocity():
    tracker = CentroidTracker()
    rng = np.random.default_rng(0)
    tracks = []
    for tick in range(60):  # 5 m/s along x, with 0.15 m of noise, at 10 Hz
        tracks = tracker.update([(0.5 * tick + rng.normal(0, 0.15), rng.normal(0, 0.15))], tick * 0.1)
    assert len(tracks) == 1
    assert tracks[0].vx == pytest.approx(5.0, abs=1.0)
    assert abs(tracks[0].vy) < 1.0


def test_tracker_only_reports_a_track_seen_often_enough():
    tracker = CentroidTracker(confirm=(2, 5))
    assert tracker.update([(0.0, 0.0)], 0.0) == []
    assert len(tracker.update([(0.1, 0.0)], 0.1)) == 1


def test_a_single_false_box_never_becomes_a_track():
    tracker = CentroidTracker(confirm=(2, 5))
    assert tracker.update([(0.0, 0.0)], 0.0) == []
    assert tracker.update([], 0.1) == []
    assert tracker.update([(50.0, 0.0)], 0.2) == []


def test_tracker_carries_a_missed_car_on_at_its_velocity_then_drops_it():
    tracker = CentroidTracker(max_age=0.5, confirm=(2, 5))
    for tick in range(10):
        tracks = tracker.update([(0.5 * tick, 0.0)], tick * 0.1)
    before = tracks[0]
    coasted = tracker.update([], 1.0)
    assert coasted[0].id == before.id
    assert coasted[0].x == pytest.approx(5.0, abs=0.3)
    assert tracker.update([], 2.0) == []


def test_a_spot_that_only_flickers_never_becomes_a_steady_ghost():
    tracker = CentroidTracker(max_age=0.5, confirm=(5, 10))
    reported = 0
    for tick in range(100):  # a pole above the threshold one tick in three, always in the same place
        reported += len(tracker.update([(10.0, 10.0)] if tick % 3 == 0 else [], tick * 0.1))
    assert reported == 0


def test_a_person_detected_most_of_the_time_is_reported_without_gaps():
    tracker = CentroidTracker(max_age=0.5, confirm=(5, 10))
    reported = []
    for tick in range(60):  # missed one tick in four
        reported.append(len(tracker.update([] if tick % 4 == 3 else [(0.15 * tick, 0.0)], tick * 0.1)))
    assert reported[10:] == [1] * 50


def test_tracker_turns_a_box_yaw_to_face_the_way_the_track_moves():
    tracker = CentroidTracker()
    for tick in range(5):  # moving along +x, with a box that says it faces the other way
        tracks = tracker.update([(0.5 * tick, 0.0, 180.0)], tick * 0.1)
    assert abs((tracks[0].yaw + 180.0) % 360.0 - 180.0) < 1.0


def test_tracker_keeps_a_stopped_cars_yaw_from_flipping():
    tracker = CentroidTracker()
    tracks = tracker.update([(0.0, 0.0, 90.0)], 0.0)
    for tick, yaw in enumerate((-90.0, 90.0, -90.0), 1):
        tracks = tracker.update([(0.0, 0.0, yaw)], tick * 0.1)
    assert tracks[0].yaw == pytest.approx(90.0)
    assert tracks[0].yaw_rate == pytest.approx(0.0)


def test_parse_lidar_reads_xyzi_rows():
    raw = np.array([1, 2, 3, 0.5, 4, 5, 6, 0.25], dtype=np.float32).tobytes()
    points = parse_lidar(raw)
    assert points.dtype == np.float32
    assert points.tolist() == [[1, 2, 3, 0.5], [4, 5, 6, 0.25]]


def test_lidar_frame_is_y_left_with_the_road_at_ground_z():
    # A point on the road, 2 m to the right of a sensor 7 m up over the junction centre
    points = np.array([[1.0, 2.0, -7.0, 0.5]], dtype=np.float32)
    assert lidar_to_model_frame(points, (0.0, 0.0, 7.0), ground_z=-1.8).tolist()[0] == pytest.approx([1.0, -2.0, -1.8, 0.5])
    assert points[0, 1] == 2.0  # input untouched


def test_a_turned_lidar_still_puts_a_spot_in_the_same_place():
    # 10 m along the world's x from a sensor turned 90 degrees: it reads that as 10 m to its left, which in CARLA is -y
    turned = np.array([[0.0, -10.0, -7.0, 0.5]], dtype=np.float32)
    straight = np.array([[10.0, 0.0, -7.0, 0.5]], dtype=np.float32)
    mount = (1.0, 2.0, 7.0)
    assert lidar_to_model_frame(turned, mount, ground_z=-1.8, yaw=90.0).tolist()[0] == pytest.approx(lidar_to_model_frame(straight, mount, ground_z=-1.8).tolist()[0])


def test_two_lidars_put_the_same_spot_at_the_same_place():
    # A spot on the road at CARLA (3, 4) from the junction centre, seen from opposite corners
    a = lidar_to_model_frame(np.array([[23.0, 24.0, -7.0, 0.5]], dtype=np.float32), (-20.0, -20.0, 7.0), ground_z=-1.8)
    b = lidar_to_model_frame(np.array([[-17.0, -16.0, -7.0, 0.5]], dtype=np.float32), (20.0, 20.0, 7.0), ground_z=-1.8)
    assert a.tolist()[0] == pytest.approx([3.0, -4.0, -1.8, 0.5])
    assert b.tolist()[0] == pytest.approx([3.0, -4.0, -1.8, 0.5])
    # And a box found there comes back at that spot in the world, with the junction at (100, 200)
    box = model_box_to_world([*a[0, :3], 4.5, 2.0, 1.4, 0.0], 100.0, 200.0, 0.0, ground_z=-1.8)
    assert (box.x, box.y, box.z) == pytest.approx((103.0, 204.0, 0.0), abs=1e-5)


def test_model_box_round_trips_with_the_lidar_frame():
    # A box on the road, 3 m right and 5 m forward of a gantry at world (100, 200)
    box = model_box_to_world([5.0, -3.0, -1.8 + 0.7, 4.5, 2.0, 1.4, math.pi / 2], 100.0, 200.0, 0.0, ground_z=-1.8)
    assert (box.x, box.y) == pytest.approx((105.0, 203.0))
    assert box.z == pytest.approx(0.7)
    assert (box.length, box.width, box.height) == (4.5, 2.0, 1.4)
    assert box.yaw == pytest.approx(-90.0)  # counter-clockwise in the model is negative yaw in CARLA


def test_thing_to_track_prefers_velocity_feature():
    thing = {"features": {"location": {"properties": {"x": 1, "y": 2}},
                          "velocity": {"properties": {"vx": 3, "vy": 4}}}}
    track = thing_to_track("v", thing)
    assert (track.x, track.y, track.vx, track.vy) == (1, 2, 3, 4)


def test_thing_to_track_converts_kuksa_kinematics_from_kmh():
    thing = {"features": {"kinematics": {"properties": {"x": 0, "y": 0, "heading": 90, "speed": 36}}}}
    track = thing_to_track("v", thing)
    assert track.vx == pytest.approx(0, abs=1e-9)
    assert track.vy == pytest.approx(10)


def test_thing_to_track_without_position():
    assert thing_to_track("v", {"features": {}}) is None


def test_apply_event_merges_partial_updates():
    thing = {"features": {"location": {"properties": {"x": 1, "y": 2}}}}
    apply_event(thing, "/features/location/properties/x", 5)
    apply_event(thing, "/features/location/properties", {"z": 7})
    assert thing["features"]["location"]["properties"] == {"x": 5, "y": 2, "z": 7}


# --- Brake decision ---
class FakeDitto:
    def __init__(self, tracks, engaged=False, stale=()):
        self.tracks = tracks
        self.engaged = engaged
        self.stale = list(stale)
        self.brakes = []
        self.metrics = []

    def telemetry(self, now):
        return self.tracks, self.stale

    def brake_engaged(self):
        return self.engaged

    def send_brake(self, ttc, reason):
        self.brakes.append((ttc, reason))
        return True

    def publish_metrics(self, report):
        self.metrics.append(report)


def on_collision_course(ttc):
    """Head-on at a closing speed of 10 m/s with the given TTC."""
    return [Track("vehicle_ml", 0, 0, 5, 0), Track("human_vehicle", 2.5 + 10 * ttc, 0, -5, 0)]


def test_brakes_below_threshold():
    ditto = FakeDitto(on_collision_course(1.4))
    SafetyEngine(ditto).step(100.0)
    assert len(ditto.brakes) == 1
    assert ditto.brakes[0][0] == pytest.approx(1.4)


def turning_across(path_ttc_steps=None):
    """Straight-line TTC sees nothing (parallel lanes), but the predicted paths meet."""
    ml_path = straight_path(0, 0, 5, 0)
    other_path = ml_path.copy() if path_ttc_steps is None else straight_path(0, 500, 5, 0)
    return [Track("vehicle_ml", 0, 0, 5, 0, ml_path), Track("human_vehicle", 40, 10, -5, 0, other_path)]


def test_brakes_on_a_path_conflict_that_straight_line_ttc_misses():
    ditto = FakeDitto(turning_across())
    SafetyEngine(ditto).step(100.0)
    assert len(ditto.brakes) == 1
    ttc, reason = ditto.brakes[0]
    assert math.isfinite(ttc)  # an infinite TTC would be invalid JSON, and Ditto would reject the brake
    json.dumps(ttc, allow_nan=False)
    assert "predicted path" in reason


def test_no_brake_when_predicted_paths_stay_apart():
    ditto = FakeDitto(turning_across(path_ttc_steps=0))
    SafetyEngine(ditto).step(100.0)
    assert ditto.brakes == []


@pytest.mark.parametrize("other,expected", [
    (Track("o", -10, 0, yaw=0.0), True),      # behind, same way: the case braking can't help
    (Track("o", 10, 0, yaw=0.0), False),      # ahead, same way
    (Track("o", 10, 0, yaw=180.0), False),    # ahead, oncoming
    (Track("o", -10, 0, yaw=180.0), False),   # behind, going away
    (Track("o", -5, -5, yaw=90.0), False),    # behind and crossing: moving could still clear it
    (Track("o", -10, 0), False),              # heading unknown, so assume it might matter
])
def test_approaching_from_behind(other, expected):
    assert approaching_from_behind(Track("me", 0, 0, yaw=0.0), other) is expected


def test_approaching_from_behind_follows_my_heading():
    me = Track("me", 0, 0, yaw=90.0)  # facing +y
    assert approaching_from_behind(me, Track("o", 0, -10, yaw=90.0)) is True
    assert approaching_from_behind(me, Track("o", 0, 10, yaw=90.0)) is False


def test_no_brake_for_a_car_coming_up_from_behind():
    # Stopped at a light, with a car closing from behind: every trigger fires, and braking is no use
    path = straight_path(0, 0, 0, 0)
    tracks = [Track("vehicle_ml", 0, 0, 0, 0, path, 0.0), Track("carla:1", -10, 0, 8, 0, path, 0.0)]
    ditto = FakeDitto(tracks)
    SafetyEngine(ditto).step(100.0)
    assert ditto.brakes == []


def test_path_check_ignores_a_stopped_car_ahead_until_ttc_says_so():
    path = straight_path(0, 0, 7, 0)
    stopped = straight_path(20, 0, 0, 0)
    tracks = [Track("vehicle_ml", 0, 0, 7, 0, path, 0.0), Track("carla:1", 20, 0, 0, 0, stopped, 0.0)]
    ditto = FakeDitto(tracks)
    SafetyEngine(ditto).step(100.0)
    assert ditto.brakes == []  # paths meet within 3 s, but this is just a car to slow down for

    tracks = [Track("vehicle_ml", 0, 0, 7, 0, path, 0.0), Track("carla:1", 10, 0, 0, 0, stopped, 0.0)]
    ditto = FakeDitto(tracks)
    SafetyEngine(ditto).step(100.0)
    assert len(ditto.brakes) == 1  # TTC under 1.5 s still brakes


def test_path_check_ignores_a_slower_car_being_followed():
    path = straight_path(0, 0, 8, 0)
    tracks = [Track("vehicle_ml", 0, 0, 8, 0, path, 0.0), Track("carla:1", 20, 0, 3, 0, straight_path(20, 0, 3, 0), 5.0)]
    ditto = FakeDitto(tracks)
    SafetyEngine(ditto).step(100.0)
    assert ditto.brakes == []


def test_path_check_applies_to_crossing_traffic():
    # A car crossing from the side, with no straight-line conflict at this offset
    me = Track("vehicle_ml", 0, 0, 5, 0, yaw=0.0)
    assert path_check_applies(me, Track("o", 10, -10, 0, 5, yaw=90.0))
    assert not path_check_applies(me, Track("o", 10, -10, 0, 0, yaw=90.0))  # stopped
    assert not path_check_applies(me, Track("o", 10, 0, 5, 0, yaw=0.0))     # following


def test_straight_line_ttc_ignores_stopped_cars_mid_turn():
    turning = Track("me", 0, 0, 5, 0, yaw=0.0, yaw_rate=-30.0)
    straight = Track("me", 0, 0, 5, 0, yaw=0.0)
    assert not ttc_check_applies(turning, Track("o", 8, 0))        # a queue it happens to be pointing at
    assert ttc_check_applies(turning, Track("o", 8, 0, -5, 0))     # a moving car still counts
    assert ttc_check_applies(straight, Track("o", 8, 0))           # going straight, it's really ahead


def test_overhead_marker_looks_identical_from_the_gantry():
    # Project corners onto the road from a gantry 40 m up: raised marker and real box must coincide
    def on_road(x, y, z, gantry=(100.0, 50.0, 40.0)):
        scale = gantry[2] / (gantry[2] - z)
        return gantry[0] + (x - gantry[0]) * scale, gantry[1] + (y - gantry[1]) * scale

    box = Box(120.0, 60.0, 0.75, 4.8, 2.1, 1.5, 30.0, True)
    marker = overhead_marker(box, 100.0, 50.0, 40.0, marker_height=7.0)
    assert marker.z == pytest.approx(7.0)
    assert on_road(marker.x, marker.y, marker.z) == pytest.approx(on_road(box.x, box.y, 0.0))
    assert on_road(marker.x + marker.length / 2, marker.y, marker.z)[0] == pytest.approx(
        on_road(box.x + box.length / 2, box.y, 0.0)[0])
    assert marker.yaw == box.yaw


def test_no_brake_while_already_stopped():
    # Waiting at the stop line with a car heading straight at it: braking changes nothing
    tracks = [Track("vehicle_ml", 0, 0, 0, 0, yaw=0.0), Track("carla:1", 5, 0, -5, 0, yaw=180.0)]
    ditto = FakeDitto(tracks)
    SafetyEngine(ditto).step(100.0)
    assert ditto.brakes == []


def test_still_brakes_for_a_stopped_car_ahead():
    tracks = [Track("vehicle_ml", 0, 0, 8, 0, straight_path(0, 0, 8, 0), 0.0),
              Track("carla:1", 10, 0, 0, 0, straight_path(10, 0, 0, 0), 0.0)]
    ditto = FakeDitto(tracks)
    SafetyEngine(ditto).step(100.0)
    assert len(ditto.brakes) == 1


def test_no_brake_above_threshold():
    ditto = FakeDitto(on_collision_course(1.6))
    SafetyEngine(ditto).step(100.0)
    assert ditto.brakes == []


def test_no_repeat_once_engaged():
    ditto = FakeDitto(on_collision_course(1.0), engaged=True)
    SafetyEngine(ditto).step(100.0)
    assert ditto.brakes == []


def test_unconfirmed_brake_is_retried_but_rate_limited():
    ditto = FakeDitto(on_collision_course(1.0))
    engine = SafetyEngine(ditto)
    engine.step(100.0)
    engine.step(100.05)
    assert len(ditto.brakes) == 1
    engine.step(100.3)
    assert len(ditto.brakes) == 2


def test_only_hazards_to_the_controlled_vehicle_brake():
    tracks = [Track("human_vehicle", 0, 0, 5, 0), Track("carla:1", 10, 0, -5, 0),
              Track("vehicle_ml", 0, 500)]
    ditto = FakeDitto(tracks)
    SafetyEngine(ditto).step(100.0)
    assert ditto.brakes == []


def test_metrics_stay_valid_json_without_a_conflict():
    ditto = FakeDitto([Track("vehicle_ml", 0, 0), Track("human_vehicle", 50, 0)])
    SafetyEngine(ditto).step(100.0)
    assert ditto.metrics[0]["ttc"] is None
    json.dumps(ditto.metrics[0], allow_nan=False)


# --- Onboard gate ---
def event(path, value):
    return json.dumps({"topic": "org.example/vehicle_ml/things/twin/events/modified",
                       "path": path, "value": value}).encode()


@pytest.mark.parametrize("path,value,expected", [
    ("/features/SafetyGate/properties/emergency_brake", True, True),
    ("/features/SafetyGate/properties", {"emergency_brake": False}, False),
    ("/features/SafetyGate", {"properties": {"emergency_brake": True}}, True),
    ("/", {"features": {"SafetyGate": {"properties": {"emergency_brake": True}}}}, True),
    ("/features/SafetyGate/properties/reason", "x", None),
    ("/features/location/properties", {"emergency_brake": True}, None),
    ("/features/SafetyGate/properties/emergency_brake", "true", None),
])
def test_parse_gate_event(path, value, expected):
    assert parse_gate_event(event(path, value)) is expected


@pytest.mark.parametrize("payload", [b"", b"not json", b"[]", b'{"path": 3}'])
def test_parse_gate_event_ignores_garbage(payload):
    assert parse_gate_event(payload) is None


def brake_event(value):
    return event("/features/SafetyGate/properties", {"emergency_brake": value})


def test_motors_are_held_until_the_first_state_arrives():
    motors = SimMotors()
    gate = SafetyGateSubscriber(motors)
    assert gate.held
    assert motors.stops == 1
    gate.handle_payload(brake_event(False))
    assert not gate.held


def test_brake_cuts_motors_immediately_and_latches():
    motors = SimMotors()
    gate = SafetyGateSubscriber(motors)
    gate.handle_payload(brake_event(False))
    stops = motors.stops
    gate.handle_payload(brake_event(True))
    assert gate.held
    assert motors.stops == stops + 1
    gate.handle_payload(b"not json")
    gate.handle_payload(event("/features/SafetyGate/properties/reason", "x"))
    assert gate.held


def test_link_loss_holds_motors_until_explicit_release():
    gate = SafetyGateSubscriber(SimMotors())
    gate.running = True
    gate.handle_payload(brake_event(False))
    gate._on_disconnect(None, None, None, "gone", None)
    assert gate.held
    gate._on_connect(_NoSubscribe(), None, None, _Ok(), None)
    assert gate.held  # reconnecting alone is not a release
    gate.handle_payload(brake_event(False))
    assert not gate.held


class _Ok:
    is_failure = False


class _NoSubscribe:
    def subscribe(self, topic, qos):
        pass


# --- Prediction ---
def test_history_is_a_ten_frame_ring_buffer():
    history = HistoryBuffer()
    for i in range(HISTORY_FRAMES - 1):
        history.add(1, i, 0, 0, 0)
    assert history.ready() == {}
    for i in range(5):
        history.add(1, 100 + i, 0, 0, 0)
    ready = history.ready()[1]
    assert len(ready) == HISTORY_FRAMES
    assert ready[-1][0] == 104


def test_history_forgets_gone_actors():
    history = HistoryBuffer(length=1)
    history.add(1, 0, 0, 0, 0)
    history.add(2, 0, 0, 0, 0)
    history.retain([2])
    assert list(history.ready()) == [2]


def test_ego_frame_is_x_forward_y_left():
    # CARLA y points right. Facing +x, a point at world +y is on the RIGHT, so ego y is negative.
    ego = world_to_ego(np.array([[1.0, 0.0], [0.0, 1.0]]), 0, 0, 0)
    assert ego == pytest.approx(np.array([[1.0, 0.0], [0.0, -1.0]]))
    # Facing +y (yaw 90): that same point is now straight ahead
    assert world_to_ego(np.array([0.0, 1.0]), 0, 0, 90) == pytest.approx(np.array([1.0, 0.0]))


def test_ego_transform_round_trips():
    points = np.array([[3.0, -4.0], [10.0, 7.5]])
    ego = world_to_ego(points, 5, -2, 37)
    assert ego_to_world(ego, 5, -2, 37) == pytest.approx(points)


def test_history_to_ego_has_the_current_pose_at_the_origin():
    history = [(float(i), 0.0, 10.0, 0.0) for i in range(HISTORY_FRAMES)]
    ego = history_to_ego(history, 0.0)
    assert ego.shape == (HISTORY_FRAMES, 4)
    assert ego[-1] == pytest.approx(np.array([0.0, 0.0, 10.0, 0.0]))
    assert ego[0, 0] == pytest.approx(-(HISTORY_FRAMES - 1))


def test_turn_rate_wraps_across_180():
    assert turn_rate([170.0, 175.0, -180.0, -175.0]) == pytest.approx(50.0)
    assert turn_rate([10.0]) == 0.0


@pytest.mark.parametrize("signal,rate,expected", [
    (LEFT, 0.0, LEFT),
    (RIGHT, -30.0, RIGHT),      # the signal wins
    (STRAIGHT, -30.0, LEFT),    # yaw falling is a left turn in CARLA's frame
    (STRAIGHT, 30.0, RIGHT),
    (STRAIGHT, 2.0, STRAIGHT),  # lane-keeping wobble isn't a turn
])
def test_turn_intent(signal, rate, expected):
    assert turn_intent(signal, rate) == expected


@pytest.mark.parametrize("changes,intent,expected", [
    ([0.0, -90.0], LEFT, 1),
    ([0.0, -90.0], STRAIGHT, 0),
    ([0.0, 90.0], LEFT, 0),      # no left available: go straight
    ([-90.0, 90.0], RIGHT, 1),
    ([-90.0, 90.0], STRAIGHT, 0),  # T-junction, no straight: least bad
])
def test_pick_branch(changes, intent, expected):
    assert pick_branch(changes, intent) == expected


def test_curve_path_is_straight_with_no_turn_rate():
    path = curve_path(0.0, 0.0, 0.0, 10.0, 0.0)
    assert path.shape == (30, 2)
    assert path[-1] == pytest.approx(np.array([30.0, 0.0]))


def test_curve_path_turns_left_on_a_negative_rate():
    path = curve_path(0.0, 0.0, 0.0, 10.0, -30.0)
    assert path[-1, 1] < -5.0  # CARLA's y points right, so left is negative


def test_along_polyline_follows_a_corner_and_stops_at_the_end():
    corner = [(0.0, 0.0), (10.0, 0.0), (10.0, -10.0)]
    path = along_polyline(corner, 5.0, steps=30)
    assert path[19] == pytest.approx(np.array([10.0, 0.0]))    # 2 s: at the corner
    assert path[23] == pytest.approx(np.array([10.0, -2.0]))   # 2.4 s: round it
    assert path[-1] == pytest.approx(np.array([10.0, -5.0]))   # 3 s: 15 m along
    assert along_polyline(corner, 50.0)[-1] == pytest.approx(np.array([10.0, -10.0]))  # out of road


def test_model_output_is_converted_to_the_world_frame():
    def straight_ahead(histories):
        times = 0.1 * np.arange(1, 31, dtype=np.float32)
        return histories[:, -1, 2:4][:, None, :] * times[None, :, None]

    # Heading along world +y at 10 m/s: 3 s ahead is 30 m further along +y
    ready = {7: [(5.0, float(i), 0.0, 10.0) for i in range(HISTORY_FRAMES)]}
    path = predict_world_paths(ready, {7: 90.0}, straight_ahead)[7]
    assert path.shape == (30, 2)
    assert path[0] == pytest.approx(np.array([5.0, HISTORY_FRAMES - 1 + 1.0]))
    assert path[-1] == pytest.approx(np.array([5.0, HISTORY_FRAMES - 1 + 30.0]), abs=1e-4)


def test_velocity_is_measured_only_when_carla_reports_none():
    assert measured_velocity(3.0, 4.0, 10.0, 0.0, (0.0, 0.0, 0.0, 0.0), 0.1) == (3.0, 4.0)
    assert measured_velocity(0.0, 0.0, 1.0, 2.0, (0.0, 0.0, 0.0, 0.0), 0.1) == pytest.approx((10.0, 20.0))
    assert measured_velocity(0.0, 0.0, 1.0, 2.0, None, 0.1) == (0.0, 0.0)


def _car_heading_along_x():
    path = np.column_stack((5.0 * 0.1 * np.arange(1, 31), np.zeros(30)))
    return Track("vehicle_ml", 0.0, 0.0, 5.0, 0.0, path, 0.0)


def test_brakes_for_a_person_walking_into_the_cars_path():
    ditto = FakeDitto([_car_heading_along_x(), person_track("walker:1", 10.0, -3.0, 0.0, 1.5)])
    SafetyEngine(ditto).step(100.0)
    assert len(ditto.brakes) == 1


def test_a_person_standing_at_the_kerb_is_not_a_track_at_all():
    assert person_track("walker:1", 10.0, -3.0, 0.0, 0.5) is None


def test_no_brake_for_a_person_walking_along_the_kerb():
    ditto = FakeDitto([_car_heading_along_x(), person_track("walker:1", 10.0, -2.5, 1.5, 0.0)])
    SafetyEngine(ditto).step(100.0)
    assert ditto.brakes == []


def test_a_person_close_to_a_car_is_never_merged_into_it():
    car = Track("vehicle_ml", 0.0, 0.0, 5.0, 0.0)
    person = person_track("walker:1", 1.0, 1.0, 1.5, 0.0)
    assert fuse([car], [person]) == [car, person]


def test_people_are_only_checked_against_the_ml_car():
    other = Track("carla:1", 0.0, 0.0, 5.0, 0.0, np.zeros((30, 2)))
    assert SafetyEngine(FakeDitto([])).evaluate([other, person_track("walker:1", 1.0, 0.0, 1.5, 0.0)]) == []


def test_a_held_track_outlasts_its_sightings_until_its_memory_runs_out():
    tracker = CentroidTracker(max_age=2.0, confirm=(5, 10), hold=True, gate=1.5)
    for tick in range(6):
        tracker.update([(0.0, 0.0)], tick * 0.1)
    assert len(tracker.update([], 1.5)) == 1
    assert tracker.update([], 2.6) == []
    assert tracker.confirmed == set()


def test_a_faster_tracker_believes_a_person_setting_off_sooner():
    def speed_after_setting_off(**settings):
        tracker = CentroidTracker(gate=1.5, **settings)
        for tick in range(5):
            tracker.update([(0.0, 0.0)], tick * 0.1)
        for tick in range(5):
            tracks = tracker.update([(0.16 * (tick + 1), 0.0)], 0.5 + tick * 0.1)
        return tracks[0].vx

    assert speed_after_setting_off(beta=0.25) > 1.5 * speed_after_setting_off()
