"""The live traffic report's logic. No CARLA needed."""
import pytest
from traffic_monitor import (
    JunctionCounter,
    classify_pass,
    closing_speed,
    is_close_call_speed,
    relative_position,
)


@pytest.mark.parametrize("entry,exit_yaw,expected", [
    (0.0, 0.0, "straight"),
    (0.0, -90.0, "left"),      # yaw falls on a left turn in CARLA's frame
    (0.0, 90.0, "right"),
    (170.0, -100.0, "right"),  # wraps across +/-180
    (0.0, 20.0, "straight"),   # following a bend isn't a turn
])
def test_classify_pass(entry, exit_yaw, expected):
    assert classify_pass(entry, exit_yaw) == expected


def test_counter_counts_once_on_leaving_the_junction():
    counter = JunctionCounter((0.0, 0.0))
    counter.update(1, -30.0, 0.0, 0.0)
    assert counter.counts["straight"] == 0
    counter.update(1, -5.0, 0.0, 0.0)
    counter.update(1, 0.0, 0.0, -45.0)
    assert sum(counter.counts.values()) == 0  # still inside
    counter.update(1, 0.0, -30.0, -90.0)
    assert counter.counts == {"straight": 0, "left": 1, "right": 0}
    counter.update(1, 0.0, -40.0, -90.0)
    assert sum(counter.counts.values()) == 1


def test_counter_tracks_vehicles_separately():
    counter = JunctionCounter((0.0, 0.0))
    counter.update(1, 0.0, 0.0, 0.0)
    counter.update(2, 1.0, 0.0, 180.0)
    counter.update(1, 30.0, 0.0, 0.0)
    counter.update(2, 1.0, 30.0, 90.0)
    assert counter.counts["straight"] == 1
    assert sum(counter.counts.values()) == 2


def test_relative_position_is_ahead_and_left():
    me = (0.0, 0.0, 0.0, 0.0, 0.0)
    assert relative_position(me, (10.0, 0.0, 0.0, 0.0, 0.0)) == pytest.approx((10.0, 0.0))
    assert relative_position(me, (0.0, -3.0, 0.0, 0.0, 0.0)) == pytest.approx((0.0, 3.0))  # CARLA's -y is left
    facing_plus_y = (0.0, 0.0, 0.0, 0.0, 90.0)
    assert relative_position(facing_plus_y, (0.0, 10.0, 0.0, 0.0, 0.0)) == pytest.approx((10.0, 0.0))
    assert relative_position(facing_plus_y, (3.0, 0.0, 0.0, 0.0, 0.0)) == pytest.approx((0.0, 3.0))


def test_queueing_is_not_a_close_call():
    stopped = (0.0, 0.0, 0.0, 0.0, 0.0)
    assert not is_close_call_speed(stopped, (-6.0, 0.0, 3.2, 0.0, 0.0))  # pulling up behind at 3.2 m/s
    assert is_close_call_speed(stopped, (-6.0, 0.0, 9.8, 0.0, 0.0))      # not stopping in time


def test_slower_crossing_traffic_still_counts():
    stopped = (0.0, 0.0, 0.0, 0.0, 0.0)
    assert is_close_call_speed(stopped, (3.0, 5.0, 0.0, -3.0, -90.0))


def test_closing_speed():
    assert closing_speed((0, 0, 5, 0), (10, 0, -5, 0)) == pytest.approx(10.0)    # head-on
    assert closing_speed((0, 0, -5, 0), (10, 0, 5, 0)) == pytest.approx(-10.0)   # separating
    assert closing_speed((0, 0, 5, 0), (10, 0, 5, 0)) == pytest.approx(0.0)      # same speed
    assert closing_speed((0, 0, 5, 0), (0, 10, 5, 0)) == pytest.approx(0.0)      # parallel
