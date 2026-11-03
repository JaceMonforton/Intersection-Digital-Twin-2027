"""The ML car's lane driver and its gap choice in traffic. No CARLA needed."""
import pytest
from left_turn_scenario import UNSAFE_GAP, Path2D
from traffic_left_turn import (
    can_resume,
    light_role,
    oncoming_etas,
    path_blocked,
    should_go,
)
from traffic_monitor import (
    BRAKE_DECEL,
    LaneDriver,
    RedLightWatch,
    clear_of_stop_lines,
    is_running_red,
    lead_gap,
    next_speed,
    red_light_gap,
    speed_for_gap,
)


def test_the_brake_always_wins():
    assert next_speed(5.0, 7.0, 0.1, True) == pytest.approx(5.0 - BRAKE_DECEL * 0.1)
    assert next_speed(0.1, 7.0, 0.1, True) == 0.0


def test_speed_rises_and_falls_towards_the_target():
    assert 5.0 < next_speed(5.0, 7.0, 0.1, False) < 7.0
    assert 5.0 < next_speed(7.0, 5.0, 0.1, False) < 7.0


def test_speed_for_gap_stops_in_time():
    assert speed_for_gap(0.0) == 0.0
    assert speed_for_gap(-3.0) == 0.0
    assert speed_for_gap(6.0) == pytest.approx(6.0)  # v^2 = 2 * 3 * 6
    assert speed_for_gap(float("inf")) == float("inf")


def test_lead_gap_only_counts_my_lane_ahead():
    me = (0.0, 0.0, 0.0, 0.0, 0.0)
    assert lead_gap(me, [(20.0, 0.0, 0, 0, 0), (10.0, 0.5, 0, 0, 0)]) == pytest.approx(10.0)
    assert lead_gap(me, [(-5.0, 0.0, 0, 0, 0), (5.0, 3.7, 0, 0, 0)]) == float("inf")


class FakeWaypoint:
    """A straight road along +x."""

    def __init__(self, x=0.0):
        self.x = x
        self.transform = type("T", (), {
            "location": type("L", (), {"x": x, "y": 0.0, "z": 0.0})(),
            "rotation": type("R", (), {"yaw": 0.0})(),
        })()

    def next(self, step):
        return [FakeWaypoint(self.x + step)]


def test_lane_driver_moves_along_the_lane():
    driver = LaneDriver(FakeWaypoint(), None)
    assert driver.pose()[:2] == (0.0, 0.0)
    driver.speed = 5.0
    for _ in range(10):
        driver.step(0.1, 5.0, False)
    assert driver.pose()[0] == pytest.approx(5.0)


def test_red_light_gap_finds_the_stop_line_on_my_path():
    ahead = [(float(x), 0.0) for x in range(30)]
    assert red_light_gap(ahead, [(12.0, 0.5)]) == pytest.approx(11.0)  # within the lane's width
    assert red_light_gap(ahead, [(12.0, 3.7)]) == float("inf")         # the next lane's light
    assert red_light_gap(ahead, []) == float("inf")


def test_cars_are_never_placed_inside_a_lights_trigger_zone():
    def point(x):
        location = type("L", (), {"x": x, "y": 0.0})()
        return type("P", (), {"location": location})()

    near, far = point(10.0), point(40.0)
    assert clear_of_stop_lines([near, far], [(0.0, 0.0)]) == [far]


class FakeLight:
    id = 1

    def __init__(self, state):
        self.state = state

    def get_state(self):
        return self.state


class FakeWorld:
    def __init__(self, light):
        self.light = light

    def get_actors(self):
        return self

    def filter(self, _):
        return [self.light]


class FakeStop:
    transform = type("T", (), {"location": type("L", (), {"x": 0.0, "y": 0.0})(),
                               "rotation": type("R", (), {"yaw": 0.0})()})()


def red_watch():
    light = FakeLight("red")
    light.get_stop_waypoints = lambda: [FakeStop]
    carla = type("C", (), {"TrafficLightState": type("S", (), {"Red": "red", "Green": "green"})})
    return RedLightWatch(carla, FakeWorld(light))


def test_a_car_stopping_hard_at_the_line_has_not_run_the_light():
    watch = red_watch()
    for t, x, speed in ((0.0, -1.0, 4.5), (0.1, -0.5, 3.0), (0.2, 0.0, 0.5), (0.3, 0.1, 0.0), (6.0, 0.1, 0.0)):
        watch.update({7: (x, 0.0, speed, 0.0, 0.0)}, t)
    assert watch.runs == []


def test_a_car_carrying_on_through_has_run_the_light():
    watch = red_watch()
    for t, x in ((0.0, -1.0), (0.5, 1.5), (1.0, 4.0), (1.5, 6.5)):
        watch.update({7: (x, 0.0, 5.0, 0.0, 0.0)}, t)
    assert [run["vehicle"] for run in watch.runs] == [7]


def test_crossing_on_green_is_fine():
    watch = red_watch()
    watch.lines[0][0].state = "green"
    for t, x in ((0.0, -1.0), (0.5, 1.5), (1.0, 4.0), (1.5, 6.5)):
        watch.update({7: (x, 0.0, 5.0, 0.0, 0.0)}, t)
    assert watch.runs == []


def test_is_running_red():
    line = (10.0, 0.0, 0.0)  # x, y, yaw of the stop line
    assert is_running_red((10.5, 0.0, 6.0, 0.0, 0.0), *line)
    assert not is_running_red((10.5, 0.0, 0.5, 0.0, 0.0), *line)   # creeping
    assert not is_running_red((10.5, 0.0, 6.0, 0.0, 90.0), *line)  # crossing traffic on green
    assert not is_running_red((30.0, 0.0, 6.0, 0.0, 0.0), *line)   # nowhere near it


def test_lane_driver_reports_its_path_ahead():
    driver = LaneDriver(FakeWaypoint(), None)
    driver.s = 5.0
    assert driver.path_ahead(3) == [(5.0, 0.0), (6.0, 0.0), (7.0, 0.0)]


def test_lane_driver_stops_when_braked():
    driver = LaneDriver(FakeWaypoint(), None)
    driver.speed = 6.0
    for _ in range(20):
        driver.step(0.1, 7.0, True)
    assert driver.speed == 0.0


def oncoming_lane():
    return Path2D([(float(x), 0.0, 0.0) for x in range(100)])


def test_oncoming_etas_finds_cars_in_the_lane_before_the_conflict():
    vehicles = {
        1: (50.0, 0.0, 10.0, 0.0, 0.0),     # in the lane, 30 m short: 3 s
        2: (50.0, 8.0, 10.0, 0.0, 0.0),     # another lane
        3: (90.0, 0.0, 10.0, 0.0, 0.0),     # already past
        4: (50.0, 0.0, -10.0, 0.0, 180.0),  # wrong way
        5: (70.0, 0.0, 0.0, 0.0, 0.0),      # stopped 10 m short, which isn't "never"
    }
    etas = oncoming_etas(oncoming_lane(), 80.0, vehicles)
    assert set(etas) == {1, 5}
    assert etas[1] == pytest.approx(3.0)
    assert etas[5] == pytest.approx(2.5)


def test_path_blocked_only_by_a_car_on_the_path_ahead():
    path = oncoming_lane()
    assert path_blocked(path, 10.0, {1: (16.0, 0.5, 0, 0, 0)})
    assert not path_blocked(path, 10.0, {1: (16.0, 3.7, 0, 0, 0)})  # the next lane
    assert not path_blocked(path, 10.0, {1: (5.0, 0.0, 0, 0, 0)})   # behind
    assert not path_blocked(path, 10.0, {1: (40.0, 0.0, 0, 0, 0)})  # too far to matter yet


def test_a_car_stopped_to_yield_is_not_a_reason_to_wait():
    vehicles = {1: (70.0, 0.0, 0.0, 0.0, 0.0), 2: (50.0, 0.0, 10.0, 0.0, 0.0)}
    assert set(oncoming_etas(oncoming_lane(), 80.0, vehicles, moving_only=True)) == {2}


@pytest.mark.parametrize("light,turner,expected", [
    (0.0, 0.0, "turner"), (10.0, 0.0, "turner"), (180.0, 0.0, "oncoming"), (-178.0, 2.0, "oncoming"),
    (90.0, 0.0, "side"), (-90.0, 180.0, "side"),
])
def test_light_role(light, turner, expected):
    assert light_role(light, turner) == expected


def test_unsafe_goes_only_into_a_gap_too_small_to_clear():
    turn_time = 4.0
    assert should_go("unsafe", {1: turn_time - UNSAFE_GAP}, turn_time, False)
    assert not should_go("unsafe", {}, turn_time, False)
    assert not should_go("unsafe", {1: 20.0}, turn_time, False)


def test_safe_waits_for_a_clear_gap():
    assert should_go("safe", {}, 4.0, False)
    assert should_go("safe", {1: 8.0}, 4.0, False)
    assert not should_go("safe", {1: 3.0}, 4.0, False)
    assert not should_go("safe", {}, 4.0, True)  # someone is in the junction


def test_can_resume_needs_a_gap_long_enough_to_clear():
    assert can_resume({}, False, 3.0)
    assert can_resume({1: 4.0}, False, 3.0)
    assert not can_resume({1: 3.0}, False, 3.0)  # one car passing isn't enough if the next is this close
    assert not can_resume({}, True, 3.0)         # someone is in the junction
