"""The driving model's pass test verdict. No CARLA needed."""
import pytest
from pass_test import counts_as_autopilot_collision, summarise, verdict


def side(km=10.0, collisions=1, red_runs=1, speed=3.0):
    return {"km": km, "collisions": collisions, "red_runs": red_runs, "metres_per_car_second": speed}


def test_matching_autopilot_passes():
    assert verdict(side(), side())["passed"] is True


def test_up_to_a_quarter_more_is_allowed():
    assert verdict(side(km=8.0), side(km=10.0))["checks"]["collisions"]["passed"] is True  # exactly 1.25 times
    assert verdict(side(km=7.0), side(km=10.0))["checks"]["collisions"]["passed"] is False


def test_too_many_red_lights_alone_fails():
    result = verdict(side(red_runs=5), side())
    assert result["checks"]["red_runs"]["passed"] is False
    assert result["checks"]["collisions"]["passed"] is True
    assert result["passed"] is False


def test_too_little_distance_alone_fails():
    assert verdict(side(speed=2.6), side(speed=3.0))["passed"] is False
    assert verdict(side(speed=2.7), side(speed=3.0))["checks"]["distance"]["passed"] is True


def test_any_collision_fails_when_autopilot_had_none():
    assert verdict(side(collisions=1), side(collisions=0))["checks"]["collisions"]["passed"] is False
    assert verdict(side(collisions=0), side(collisions=0))["checks"]["collisions"]["passed"] is True


def test_a_run_cut_short_still_counts_its_full_time():
    full = {"distance": 900.0, "planned_seconds": 300.0, "collisions": [], "red_light_runs": [], "times_lost": 0,
            "got_stuck": False, "brake_requests": 0, "missed_ticks": 0}
    stuck = {**full, "distance": 100.0, "got_stuck": True, "collisions": [{"with": "vehicle.audi.a2"}, {"with": "static.pole"}]}
    total = summarise([full, stuck])
    assert total["metres_per_car_second"] == pytest.approx(1000.0 / 600.0)
    assert total["collisions"] == 2
    assert total["collisions_by_kind"] == {"vehicle": 1, "static": 1}
    assert total["got_stuck"] == 1


def test_a_contact_with_the_ml_car_isnt_autopilots():
    assert not counts_as_autopilot_collision("vehicle_ml")
    assert counts_as_autopilot_collision("autopilot")
    assert counts_as_autopilot_collision(None)
