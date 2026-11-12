"""The ML car's own safety monitor. No CARLA needed."""
from safety_monitor import (
    DEGRADED_SPEED,
    LINK_HOLD,
    RESUME_TICKS,
    SafetyMonitor,
    stopping_helps,
)


def test_stale_perception_holds_the_car_until_it_has_been_fresh_for_a_while():
    monitor = SafetyMonitor()
    assert monitor.check(True, None, 8.0) == 8.0
    assert monitor.check(False, None, 8.0) is None
    for _ in range(RESUME_TICKS - 1):
        assert monitor.check(True, None, 8.0) is None  # fresh again, but not for long enough
    assert monitor.check(True, None, 8.0) == 8.0
    assert monitor.times["stale"] == 1 and monitor.ticks["stale"] == RESUME_TICKS


def test_a_lost_roadside_link_caps_the_speed_and_never_raises_it():
    monitor = SafetyMonitor()
    assert monitor.check(True, True, 8.0) == 8.0
    assert monitor.check(True, False, 8.0) == DEGRADED_SPEED
    assert monitor.check(True, False, 2.0) == 2.0  # a slower target stands
    for _ in range(LINK_HOLD - 1):
        assert monitor.check(True, True, 8.0) == DEGRADED_SPEED  # back, but not trusted yet
    assert monitor.check(True, True, 8.0) == 8.0
    assert monitor.times["link"] == 2  # the 2.0 tick broke it into two spells


def test_a_config_with_no_roadside_unit_is_never_capped_for_the_link():
    monitor = SafetyMonitor()
    assert all(monitor.check(True, None, 12.0) == 12.0 for _ in range(20))
    assert not monitor.ticks


def test_the_link_cap_lets_a_car_already_in_the_junction_clear_it():
    monitor = SafetyMonitor()
    assert monitor.check(True, False, 8.0, inside=True) == 8.0
    assert monitor.check(True, False, 8.0, inside=False) == DEGRADED_SPEED


EGO = (0.0, 0.0, 8.0, 0.0, 0.0)              # going +x at 8 m/s
ROUTE = [(float(x), 0.0) for x in range(60)]


def test_a_car_about_to_cross_close_ahead_is_stopped_for():
    crossing = (20.0, -15.0, 0.0, 8.0, 90.0)  # reaches the route 20 m ahead in about 2 s
    assert stopping_helps(EGO, ROUTE, [crossing], [], 8.0)
    monitor = SafetyMonitor()
    assert monitor.check(True, None, 8.0, conflict=True) == 0.0 and monitor.times["space"] == 1


def test_no_stop_for_an_empty_road_a_far_crossing_or_a_car_from_behind():
    assert not stopping_helps(EGO, ROUTE, [], [], 8.0)
    assert not stopping_helps(EGO, ROUTE, [(50.0, -60.0, 0.0, 8.0, 90.0)], [], 8.0)  # 7 s away
    assert not stopping_helps(EGO, ROUTE, [(-12.0, 0.0, 14.0, 0.0, 0.0)], [], 8.0)  # stopping wouldn't help


def test_a_person_stepping_into_the_lane_is_stopped_for():
    assert stopping_helps(EGO, ROUTE, [], [(18.0, -3.0, 0.0, 1.5)], 8.0)
