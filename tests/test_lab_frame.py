"""The lab-to-CARLA mapping and the one pose reader. No CARLA, no broker."""
import json
import math
import random

import pytest
from lab_frame import CALIBRATION, LabFrame, carla_pose, load, read_pose

FLIPPED = LabFrame(scale=1.0, rotation_deg=0.0, flip_y=True, offset_x=0.0, offset_y=0.0)


def pose_thing(**value):
    return {"features": {"pose": {"properties": value}}}


# --- The transform ---
def test_the_lab_is_right_handed_and_carla_left_handed():
    assert FLIPPED.to_carla_point(1.0, 0.0) == pytest.approx((1.0, 0.0))
    assert FLIPPED.to_carla_point(0.0, 1.0) == pytest.approx((0.0, -1.0))  # lab y is left, CARLA y is right
    assert FLIPPED.to_carla_heading(90.0) == pytest.approx(-90.0)
    assert FLIPPED.to_carla_velocity(0.0, 1.0) == pytest.approx((0.0, -1.0))


def test_a_left_turn_in_the_lab_is_still_a_left_turn_in_carla():
    # A left turn raises the lab heading; in CARLA a left turn lowers the yaw
    assert FLIPPED.to_carla_heading(10.0) < FLIPPED.to_carla_heading(0.0)


def test_scale_and_offset_move_positions_but_only_scale_velocity():
    frame = LabFrame(scale=10.0, rotation_deg=0.0, flip_y=True, offset_x=-47.8, offset_y=20.4)
    assert frame.to_carla_point(1.0, 0.5) == pytest.approx((-37.8, 15.4))
    assert frame.to_carla_velocity(1.0, 0.5) == pytest.approx((10.0, -5.0))  # never offset
    assert frame.to_carla_speed(1.5) == pytest.approx(15.0)


def test_rotation_turns_the_lab_onto_the_junction_before_the_flip():
    frame = LabFrame(scale=1.0, rotation_deg=90.0, flip_y=True, offset_x=0.0, offset_y=0.0)
    assert frame.to_carla_point(1.0, 0.0) == pytest.approx((0.0, -1.0))  # lab +x turns to lab +y, which is CARLA -y
    assert frame.to_carla_heading(0.0) == pytest.approx(-90.0)


@pytest.mark.parametrize("frame", [
    FLIPPED,
    LabFrame(scale=10.0, rotation_deg=37.0, flip_y=True, offset_x=-47.8, offset_y=20.4),
    LabFrame(scale=2.5, rotation_deg=-120.0, flip_y=False, offset_x=3.0, offset_y=-4.0),
])
def test_heading_velocity_and_position_always_agree(frame):
    rng = random.Random(7)
    for _ in range(50):
        x, y, heading, speed = rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-180, 180), rng.uniform(0.1, 2)
        vx, vy = speed * math.cos(math.radians(heading)), speed * math.sin(math.radians(heading))
        cvx, cvy = frame.to_carla_velocity(vx, vy)
        yaw = math.radians(frame.to_carla_heading(heading))
        carla_speed = frame.to_carla_speed(speed)
        assert (cvx, cvy) == pytest.approx((carla_speed * math.cos(yaw), carla_speed * math.sin(yaw)))
        # moving along the velocity for a moment lands where the transformed point says
        ax, ay = frame.to_carla_point(x, y)
        bx, by = frame.to_carla_point(x + 0.01 * vx, y + 0.01 * vy)
        assert ((bx - ax) / 0.01, (by - ay) / 0.01) == pytest.approx((cvx, cvy))
        assert frame.to_lab_point(ax, ay) == pytest.approx((x, y))


# --- The calibration file ---
GOOD = {"calibrated": False, "scale": 10.0, "rotation_deg": 0.0, "flip_y": True, "offset_x": -47.8, "offset_y": 20.4}


def write(tmp_path, fields):
    path = tmp_path / "lab_frame.json"
    path.write_text(json.dumps(fields), encoding="utf-8")  # NaN and Infinity come out as JSON's non-standard tokens
    return path


def test_a_good_file_loads(tmp_path):
    frame = load(write(tmp_path, GOOD))
    assert frame.scale == 10.0 and frame.flip_y and not frame.calibrated


def test_the_committed_calibration_loads_and_says_it_isnt_calibrated_yet():
    frame = load(CALIBRATION)
    assert frame.flip_y and not frame.calibrated


def test_a_missing_file_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "nowhere.json")


@pytest.mark.parametrize("change", [
    {"scale": 0.0}, {"scale": -10.0}, {"scale": float("nan")}, {"offset_x": float("inf")},
    {"flip_y": "yes"}, {"calibrated": 1}, {"rotation_deg": None}, {"scale": True},
])
def test_a_bad_value_fails_loudly(tmp_path, change):
    with pytest.raises(ValueError):
        load(write(tmp_path, {**GOOD, **change}))


def test_a_missing_key_or_broken_json_fails_loudly(tmp_path):
    fields = dict(GOOD)
    del fields["flip_y"]
    with pytest.raises(ValueError):
        load(write(tmp_path, fields))
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load(broken)


# --- The pose reader ---
def test_the_pose_feature_is_in_the_lab_frame():
    pose = read_pose(pose_thing(x=1.0, y=2.0, heading=90.0, speed=0.5, t=3.0, source="simulated stand-in"))
    assert pose["frame"] == "lab" and pose["source"] == "simulated stand-in" and pose["t"] == 3.0
    assert (pose["vx"], pose["vy"]) == pytest.approx((0.0, 0.5), abs=1e-9)


def test_the_pose_feature_wins_over_the_older_formats():
    thing = pose_thing(x=1.0, y=2.0, heading=0.0, speed=0.0)
    thing["features"]["location"] = {"properties": {"x": 99.0, "y": 99.0}}
    assert read_pose(thing)["x"] == 1.0


def test_the_older_formats_are_already_in_carlas_frame():
    thing = {"features": {"location": {"properties": {"x": 1, "y": 2}}, "velocity": {"properties": {"vx": 3, "vy": 4}}}}
    assert read_pose(thing)["frame"] == "carla"
    assert carla_pose(thing, LabFrame(scale=10.0, rotation_deg=45.0, flip_y=True, offset_x=5.0, offset_y=5.0))["x"] == 1


def test_kinematics_speed_is_km_per_hour():
    pose = read_pose({"features": {"kinematics": {"properties": {"x": 0, "y": 0, "heading": 90, "speed": 36}}}})
    assert (pose["vx"], pose["vy"]) == pytest.approx((0.0, 10.0), abs=1e-9)


@pytest.mark.parametrize("thing", [
    {"features": {}},
    pose_thing(y=2.0, heading=0.0, speed=0.0),                      # no x
    pose_thing(x=float("nan"), y=2.0, heading=0.0, speed=0.0),
    pose_thing(x=1.0, y=2.0, heading=float("inf"), speed=0.0),
    pose_thing(x="1", y=2.0, heading=0.0, speed=0.0),
])
def test_no_pose_or_a_broken_one_is_no_pose(thing):
    assert read_pose(thing) is None


def test_a_lab_pose_comes_out_in_carla_metres_with_its_heading_and_velocity():
    frame = LabFrame(scale=10.0, rotation_deg=0.0, flip_y=True, offset_x=-47.8, offset_y=20.4)
    pose = carla_pose(pose_thing(x=1.0, y=0.5, heading=90.0, speed=1.0, t=1.0, source="simulated stand-in"), frame)
    assert (pose["x"], pose["y"]) == pytest.approx((-37.8, 15.4))
    assert pose["yaw"] == pytest.approx(-90.0)
    assert (pose["vx"], pose["vy"]) == pytest.approx((0.0, -10.0), abs=1e-9)
    assert pose["frame"] == "carla" and pose["source"] == "simulated stand-in"
