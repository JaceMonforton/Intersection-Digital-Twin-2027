"""The ML car's own safety monitor. No CARLA needed."""
from safety_monitor import DEGRADED_SPEED, LINK_HOLD, RESUME_TICKS, SafetyMonitor


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
