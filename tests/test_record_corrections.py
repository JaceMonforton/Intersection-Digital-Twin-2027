"""When autopilot takes over from the model. No CARLA needed."""
import math

from record_corrections import OFF_LANE, STUCK_SECONDS, takeover_reason


def ego(speed=5.0):
    return (0.0, 0.0, speed, 0.0, 0.0)  # heading along world +x; CARLA's y points right


def reason(me=None, stop_gap=math.inf, cars=(), walkers=(), stopped_for=0.0):
    return takeover_reason(me or ego(), stop_gap, cars, walkers, stopped_for)


def test_no_reason_on_an_empty_road():
    assert reason() is None


def test_a_red_line_is_only_trouble_when_too_close_to_stop():
    assert reason(stop_gap=20.0) is None
    assert reason(stop_gap=3.0) == "red line"
    assert reason(me=ego(speed=0.2), stop_gap=0.5) is None  # crawling, so nothing is urgent


def test_a_stopped_car_ahead_is_trouble_only_when_near():
    assert reason(cars=[(30.0, 0.0, 0.0, 0.0, 0.0)]) is None
    assert reason(cars=[(7.0, 0.0, 0.0, 0.0, 0.0)]) == "car ahead"


def test_a_car_ahead_going_as_fast_is_no_trouble():
    assert reason(cars=[(7.0, 0.0, 5.0, 0.0, 0.0)]) is None


def test_cars_beside_or_behind_are_no_trouble():
    assert reason(cars=[(7.0, 3.5, 0.0, 0.0, 0.0)]) is None
    assert reason(cars=[(-7.0, 0.0, 0.0, 0.0, 0.0)]) is None


def test_a_pedestrian_in_the_way_is_trouble():
    assert reason(walkers=[(5.0, 2.0, 0.0, 0.0)]) == "pedestrian"
    assert reason(walkers=[(5.0, 5.0, 0.0, 0.0)]) is None


def test_straying_from_the_lane_is_trouble():
    assert takeover_reason(ego(), math.inf, (), (), 0.0, OFF_LANE - 0.1) is None
    assert takeover_reason(ego(), math.inf, (), (), 0.0, OFF_LANE) == "off lane"


def test_standing_still_for_no_reason_is_trouble():
    assert reason(me=ego(speed=0.0), stopped_for=STUCK_SECONDS - 0.1) is None
    assert reason(me=ego(speed=0.0), stopped_for=STUCK_SECONDS) == "stopped"
