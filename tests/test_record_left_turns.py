"""The rule that teaches waiting for a gap, and the time-to-collision it uses. No CARLA needed."""
import math

import pytest

pytest.importorskip("paho")

from ml_driver import crossing_ttc
from record_left_turns import (
    CLEAR_INSIDE,
    COMMITTED_INSIDE,
    GO_SPEED,
    HOLD_INSIDE,
    MOVING,
    MOVING_MARGIN,
    SET_OFF_MARGIN,
    START_LAG,
    teacher_target,
    time_to_clear,
)
from waymo_predictor_node import LEFT, STRAIGHT


def seen(inside=HOLD_INSIDE, ttc=None, speed=0.0, route=LEFT, stop_gap=None):
    return {"route": route, "turn_gap": -inside, "crossing_ttc": ttc, "speed": speed, "stop_gap": stop_gap}


ROUTE = [(0.0, float(y)) for y in range(30)]  # heading along +y


def test_time_to_clear_from_a_stop_and_at_speed():
    assert time_to_clear(16.0, GO_SPEED) == pytest.approx(16.0 / GO_SPEED)
    assert time_to_clear(16.0, 0.0) > 16.0 / GO_SPEED
    assert time_to_clear(0.5, 0.0) == pytest.approx(math.sqrt(2.0 * 0.5 / 1.5))


def test_waits_at_the_waiting_point_when_a_car_is_too_soon():
    needed = time_to_clear(CLEAR_INSIDE - (HOLD_INSIDE - 0.5), 0.0) + START_LAG + SET_OFF_MARGIN
    assert teacher_target(seen(HOLD_INSIDE - 0.5, needed - 0.5), 3.0) == 0.0


def test_goes_when_the_gap_is_big_enough():
    needed = time_to_clear(CLEAR_INSIDE - (HOLD_INSIDE - 0.5), 0.0) + START_LAG + SET_OFF_MARGIN
    assert teacher_target(seen(HOLD_INSIDE - 0.5, needed + 0.5), 0.0) == GO_SPEED


def test_keeps_going_on_a_gap_it_wouldnt_set_off_on():
    inside = HOLD_INSIDE + 1.0
    needed_moving = time_to_clear(CLEAR_INSIDE - inside, 2.0) + MOVING_MARGIN
    assert teacher_target(seen(inside, needed_moving + 0.2, speed=2.0), 0.0) == GO_SPEED
    assert teacher_target(seen(inside, needed_moving + 0.2, speed=MOVING - 0.5), 0.0) == 0.0


def test_goes_with_no_traffic():
    assert teacher_target(seen(5.0), 0.0) == GO_SPEED


def test_keeps_going_once_past_the_waiting_point():
    assert teacher_target(seen(COMMITTED_INSIDE, 0.5, speed=3.0), 0.0) == GO_SPEED


def test_stays_put_if_stopped_past_the_waiting_point_with_a_car_due():
    assert teacher_target(seen(COMMITTED_INSIDE + 0.5, 3.0), 4.0) == 0.0
    assert teacher_target(seen(COMMITTED_INSIDE + 0.5, None), 0.0) == GO_SPEED


def test_creeps_up_to_the_waiting_point_rather_than_stopping_early():
    assert teacher_target(seen(HOLD_INSIDE - 5.0, 1.0), 0.0) > 1.5


def test_leaves_other_routes_and_red_lights_to_the_model():
    assert teacher_target(seen(route=STRAIGHT, ttc=1.0), 3.0) == 3.0
    assert teacher_target(seen(ttc=1.0, stop_gap=5.0), 3.0) == 3.0


def test_crossing_time_is_how_long_a_car_takes_to_reach_the_route():
    car = (20.0, 10.0, -5.0, 0.0, 180.0)  # 20 m to the side, crossing at 5 m/s
    assert crossing_ttc(ROUTE, [car], []) == pytest.approx(20.0 / 5.0)


def test_the_first_car_to_arrive_counts():
    near, far = (20.0, 10.0, -5.0, 0.0), (40.0, 12.0, -5.0, 0.0)
    assert crossing_ttc(ROUTE, [far, near], []) == pytest.approx(20.0 / 5.0)


def test_a_car_that_misses_the_route_or_has_passed_it_doesnt_count():
    assert crossing_ttc(ROUTE, [(20.0, 50.0, -5.0, 0.0)], []) == math.inf
    assert crossing_ttc(ROUTE, [(-5.0, 10.0, -5.0, 0.0)], []) == math.inf


def test_a_car_going_along_the_route_doesnt_count():
    assert crossing_ttc(ROUTE, [(0.0, -5.0, 0.0, 5.0)], []) == math.inf


def test_a_stopped_car_doesnt_count():
    assert crossing_ttc(ROUTE, [(5.0, 10.0, 0.0, 0.0)], []) == math.inf


def test_a_car_due_to_stop_at_a_red_light_doesnt_count():
    car = (20.0, 10.0, -5.0, 0.0)
    assert crossing_ttc(ROUTE, [car], [(10.0, 10.0)]) == math.inf
    assert crossing_ttc(ROUTE, [car], [(10.0, 20.0)]) < math.inf  # someone else's line


def test_a_car_too_close_to_stop_for_its_light_still_counts():
    fast = (20.0, 10.0, -12.0, 0.0)
    assert crossing_ttc(ROUTE, [fast], [(10.0, 10.0)]) == pytest.approx(20.0 / 12.0)  # needs 20 m to stop, has 10


def test_a_car_too_far_off_in_time_doesnt_count():
    assert crossing_ttc(ROUTE, [(100.0, 10.0, -5.0, 0.0)], []) == math.inf
