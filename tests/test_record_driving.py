"""Labelling recorded autopilot driving. No CARLA needed."""
import numpy as np
import pytest
from driving_features import INPUT_SIZE
from record_driving import (
    ROUTE_DISTANCE,
    FollowTrack,
    convert_chunk,
    split_tracks,
)
from traffic_monitor import LaneDriver
from waymo_predictor_node import LEFT, RIGHT, STRAIGHT


class Road:
    """A road along +x that forks at x=fork: straight on, or curving away to one side."""

    def __init__(self, fork=None, side=-1.0):
        self.fork = fork
        self.side = side  # -1 is a left turn, because CARLA's y points right


class Waypoint:
    def __init__(self, road, x, y=0.0, yaw=0.0, is_junction=False):
        self.road = road
        self.x, self.y = x, y
        self.is_junction = is_junction
        self.transform = type("T", (), {
            "location": type("L", (), {"x": x, "y": y, "z": 0.0})(),
            "rotation": type("R", (), {"yaw": yaw})(),
        })()

    def next(self, step):
        fork = self.road.fork
        if fork is not None and self.y == 0.0 and self.x + step > fork >= self.x:
            turn_yaw = 90.0 * self.road.side
            return [Waypoint(self.road, self.x + step), Waypoint(self.road, self.x, self.y + step * self.road.side, turn_yaw)]
        if self.yaw_is_turned():
            return [Waypoint(self.road, self.x, self.y + step * self.road.side, self.transform.rotation.yaw)]
        return [Waypoint(self.road, self.x + step)]

    def yaw_is_turned(self):
        return self.transform.rotation.yaw != 0.0


def straight_rows(car_id=1, x_start=0.0, ticks=1000, speed=5.0, first_tick=100):
    rows = np.zeros((ticks, 10))
    rows[:, 0] = first_tick + np.arange(ticks)
    rows[:, 1] = car_id
    rows[:, 2] = x_start + speed * 0.1 * np.arange(ticks)
    rows[:, 4] = speed
    rows[:, 7] = 0.1    # steer
    rows[:, 8] = 0.6    # throttle
    return rows


def convert(cars, road=None, walkers=None, first_tick=100, find=None):
    ticks = int(cars[:, 0].max()) - first_tick + 1
    road = road or Road()
    find = find or (lambda x, y: Waypoint(road, float(round(x))))
    walkers = np.zeros((0, 5)) if walkers is None else walkers
    return convert_chunk(cars, walkers, np.zeros((ticks, 0), bool), np.zeros((0, 2)), first_tick, find)


def test_tracks_are_cut_at_a_missed_tick():
    rows = straight_rows(ticks=20)
    rows[10:, 0] += 3
    assert [len(t) for t in split_tracks(rows)] == [10, 10]


def test_tracks_are_cut_where_a_car_was_moved():
    rows = straight_rows(ticks=20)
    rows[10:, 2] += 100.0
    assert [len(t) for t in split_tracks(rows)] == [10, 10]


def test_a_lone_tick_is_not_a_track():
    assert split_tracks(straight_rows(ticks=1)) == []


def test_follow_track_picks_the_branch_the_car_took():
    road = Road()
    straight_on = Waypoint(road, 50.0)
    turned = Waypoint(road, 50.0, -1.0, -90.0)
    went_straight = np.array([(float(x), 0.0) for x in range(100)])
    went_left = np.array([(float(min(x, 50)), -float(max(x - 50, 0))) for x in range(100)])
    assert FollowTrack(went_straight).choice([straight_on, turned]) is straight_on
    assert FollowTrack(went_left).choice([straight_on, turned]) is turned


def test_converts_the_controls_and_shapes():
    inputs, targets, routes, counts = convert(straight_rows())
    assert inputs.shape == (counts["kept"], INPUT_SIZE)
    assert targets[0] == pytest.approx((0.1, 5.0))  # its steering, and its speed a second later
    assert set(routes) == {STRAIGHT}
    assert counts["kept"] > 0


def test_the_end_of_a_track_has_no_label():
    rows = straight_rows(ticks=1000, speed=5.0)  # 500 m
    _, _, _, counts = convert(rows)
    assert counts["no_future"] == pytest.approx(ROUTE_DISTANCE / 0.5, abs=2)
    assert counts["kept"] + counts["no_future"] == 1000


def test_a_turn_ahead_labels_the_route():
    road = Road(fork=100.0, side=-1.0)
    straight = np.array([(float(x), 0.0) for x in range(101)])
    turned = np.array([(100.0, -float(y)) for y in range(1, 200)])
    track = np.vstack((straight, turned))
    rows = np.zeros((len(track), 10))
    rows[:, 0] = 100 + np.arange(len(track))
    rows[:, 1] = 1
    rows[:, 2:4] = track

    def find(x, y):
        return Waypoint(road, float(round(x)), float(round(y)), -90.0 if y < 0.0 else 0.0)

    _, _, routes, _ = convert(rows, road=road, find=find)
    assert LEFT in routes
    assert RIGHT not in routes
    assert routes[0] == STRAIGHT  # the turn is more than 30 m away at the start


def test_no_sample_until_the_car_is_out_of_a_junction():
    rows = straight_rows()
    road = Road()
    _, _, _, counts = convert(rows, find=lambda x, y: Waypoint(road, float(round(x)), is_junction=x < 10.0))
    assert counts["no_lane"] == 20  # 10 m at 0.5 m per tick


def test_no_sample_when_facing_against_the_lane():
    road = Road()
    _, _, _, counts = convert(straight_rows(), find=lambda x, y: Waypoint(road, float(round(x)), yaw=180.0))
    assert counts["kept"] == 0


def test_a_car_leaving_its_route_is_dropped_then_picked_up_again():
    rows = straight_rows()
    drift = np.arange(200)
    rows[200:400, 3] = np.minimum(drift, drift[::-1]) * 0.05  # out to 5 m to the side, and back
    _, _, _, counts = convert(rows)
    assert counts["lost_lane"] >= 1
    assert counts["kept"] > 500  # both before the drift and after it


def test_other_cars_and_walkers_reach_the_inputs_but_not_itself():
    me = straight_rows(car_id=1)
    other = straight_rows(car_id=2, x_start=10.0)
    walkers = np.array([(100.0 + i, 6.0 + 0.5 * i, 3.0, 0.0, 0.0) for i in range(1000)], float)
    inputs, _, _, counts = convert(np.vstack((me, other)), walkers=walkers)
    assert counts["kept"] > 0
    car_slots = inputs[:, 28:58].reshape(-1, 6, 5)
    walker_slots = inputs[:, 58:].reshape(-1, 3, 5)
    assert car_slots[:, :, 0].sum(axis=1).max() == 1.0   # the other car, never itself
    assert walker_slots[0, 0, 0] == 1.0
    assert walker_slots[0, 0, 2] == pytest.approx(-3.0)  # to the right of the car


def stops_then_goes(stop_tick=40, wait=200, ticks=1000):
    """5 m/s, standing still for a while from stop_tick, which is 20 m along."""
    rows = straight_rows(ticks=ticks)
    t = np.arange(ticks)
    rows[:, 2] = 0.5 * np.minimum(t, stop_tick) + 0.5 * np.maximum(t - stop_tick - wait, 0)
    rows[:, 4] = np.where((t >= stop_tick) & (t < stop_tick + wait), 0.0, 5.0)
    return rows


def convert_with_red_line(rows):
    road = Road()
    ticks = int(rows[:, 0].max()) - 100 + 1
    return convert_chunk(rows, np.zeros((0, 5)), np.ones((ticks, 1), bool), np.array([(30.0, 0.0)]), 100,
                         lambda x, y: Waypoint(road, float(round(x))))


def test_the_target_speed_is_from_a_second_later():
    inputs, targets, _, _ = convert(stops_then_goes(stop_tick=500))
    slowing = np.flatnonzero(targets[:, 1] < 1.0)[0]
    assert inputs[slowing, 0] == pytest.approx(5.0)  # it's still at speed a second before it stops


def test_a_car_that_drives_through_a_red_line_is_dropped():
    inputs, _, _, counts = convert_with_red_line(straight_rows(ticks=400))
    assert counts["drove_through"] == pytest.approx(64, abs=2)  # 30 m at 0.5 m per tick, and a line counts from 1.75 m past
    assert not inputs[:, 4].any()


def test_a_car_that_stops_for_a_red_line_is_kept():
    inputs, _, _, _ = convert_with_red_line(stops_then_goes())
    assert inputs[0, 4] == 1.0
    assert inputs[0, 5] == pytest.approx(30.0, abs=2.0)
    assert (inputs[:, 0] > 1.0)[inputs[:, 4] == 1.0].sum() >= 39  # the approach, at speed


def test_samples_can_be_limited_to_chosen_ticks_of_one_car():
    me = straight_rows(car_id=1)
    other = straight_rows(car_id=2, x_start=10.0)
    wanted = {(110, 1), (111, 1), (112, 1)}
    road = Road()
    inputs, _, _, counts = convert_chunk(np.vstack((me, other)), np.zeros((0, 5)), np.zeros((1000, 0), bool), np.zeros((0, 2)), 100,
                                         lambda x, y: Waypoint(road, float(round(x))), wanted)
    assert counts["kept"] == 3
    assert inputs[:, 28:58].reshape(-1, 6, 5)[:, :, 0].sum(axis=1) == pytest.approx(1.0)  # the other car is still seen


def test_a_branch_is_judged_at_the_end_of_the_junction():
    class Curve:
        """A lane through a junction 35 m long that has only turned 30 degrees after 15 m."""

        def __init__(self, s=0):
            self.s = s
            self.is_junction = s < 35
            self.transform = type("T", (), {"rotation": type("R", (), {"yaw": -90.0 * min(s, 35) / 35.0})()})()

        def next(self, step):
            return [Curve(self.s + 1)]

    driver = LaneDriver.__new__(LaneDriver)
    assert driver._branch_heading(Curve()) == pytest.approx(-90.0)


def test_lane_driver_finds_its_place_for_a_car_it_isnt_driving():
    driver = LaneDriver(Waypoint(Road(), 0.0), None)
    assert driver.advance_to(5.2, 1.0) == pytest.approx(np.hypot(0.2, 1.0))
    assert driver.path_ahead(1)[0][0] == pytest.approx(5.0)
    driver.advance_to(4.0, 0.0)
    assert driver.path_ahead(1)[0][0] == pytest.approx(5.0)  # never moves backwards along the route
