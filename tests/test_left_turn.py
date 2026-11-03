"""The left-turn plan and its analysis. No CARLA or Ditto needed."""
import math

import pytest
from left_turn_scenario import (
    BRAKE_DECEL,
    TURN_ACCEL,
    TURN_SPEED,
    Path2D,
    Result,
    Sample,
    Turner,
    analyse,
    classify,
    conflict_point,
    departure_time,
    measured_pet,
    outline,
    outline_gap,
    rectangles_gap,
    rectangles_overlap,
    replay_without_brake,
    time_to_cover,
    yaw_difference,
    zone_times,
)


def left_turn_paths():
    """Turner heading +x then turning left (-y in CARLA); oncoming heading -x in the adjacent lane,
    3.5 m to the left. Their paths cross at (40, -3.5)."""
    turner = [(float(x), 0.0, 0.0) for x in range(41)] + [(40.0, -float(y), 0.0) for y in range(1, 31)]
    oncoming = [(float(x), -3.5, 0.0) for x in range(150, -30, -1)]
    return Path2D(turner), Path2D(oncoming)


def test_path_is_sampled_by_distance():
    path = Path2D([(0, 0, 0), (10, 0, 0), (10, 10, 0)])
    assert path.length == 20
    assert path.position(5) == (5, 0)
    assert path.position(15) == (10, 5)
    assert path.position(999) == (10, 10)
    assert path.yaw(5) == pytest.approx(0)
    assert path.yaw(15) == pytest.approx(90)


def test_conflict_point_of_crossing_paths():
    x, y, sa, sb, gap = conflict_point(*left_turn_paths())
    assert (x, y) == pytest.approx((40, -3.5), abs=0.5)
    assert gap < 1.0
    assert sa == pytest.approx(43.5, abs=1.0)
    assert sb == pytest.approx(110, abs=1.0)


def test_left_is_negative_yaw_change_in_carla():
    assert yaw_difference(0, -90) == -90  # facing +x, turning towards -y, which is left
    assert yaw_difference(170, -170) == 20  # wraps across +/-180


def test_time_to_cover_accelerates_then_cruises():
    ramp = TURN_SPEED ** 2 / (2 * TURN_ACCEL)
    assert time_to_cover(ramp, TURN_SPEED, TURN_ACCEL) == pytest.approx(TURN_SPEED / TURN_ACCEL)
    assert time_to_cover(ramp + TURN_SPEED, TURN_SPEED, TURN_ACCEL) == pytest.approx(TURN_SPEED / TURN_ACCEL + 1)


def test_unsafe_leaves_before_safe():
    assert departure_time("unsafe", 15.0) < departure_time("safe", 15.0)


def test_turner_waits_then_rolls_then_brakes():
    turner = Turner(depart_at=1.0)
    assert turner.step(0.5, 0.1, False) == (0.0, 0.0)
    _, v = turner.step(1.0, 0.1, False)
    assert v == pytest.approx(TURN_ACCEL * 0.1)
    turner.v = TURN_SPEED
    _, v = turner.step(2.0, 0.1, True)
    assert v == pytest.approx(TURN_SPEED - BRAKE_DECEL * 0.1)
    for _ in range(30):
        _, v = turner.step(3.0, 0.1, True)
    assert v == 0.0


@pytest.mark.parametrize("b,expected", [
    ((0, 0, 0), True),        # same spot
    ((4, 0, 0), True),        # nose to tail, 0.8 m overlap
    ((6, 0, 0), False),       # nose to tail, clear
    ((0, 2.0, 0), True),      # side by side, 0.1 m overlap
    ((0, 2.3, 0), False),     # side by side, clear
    ((3, 3, 90), True),       # T-bone: b's rear corner reaches a's side
    ((5, 5, 45), False),      # corners would overlap as circles, not as rectangles
])
def test_rectangles_overlap(b, expected):
    assert rectangles_overlap(outline(0, 0, 0), outline(*b)) is expected


def test_circle_contact_distance_is_not_real_contact():
    # Parallel in adjacent lanes, centres 2.4 m apart: inside the 2.5 m TTC circle, but not touching
    apart = Sample(0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 2.4, -5.0, 0.0, False, 0.0, 180.0)
    assert outline_gap(apart) == pytest.approx(0.3)


def test_rectangles_gap():
    assert rectangles_gap(outline(0, 0, 0), outline(0, 0, 0)) == 0.0
    # Nose to tail: 4.8 m long, centres 5.8 m apart
    assert rectangles_gap(outline(0, 0, 0), outline(5.8, 0, 0)) == pytest.approx(1.0)
    # Side by side: 2.1 m wide, centres 2.6 m apart
    assert rectangles_gap(outline(0, 0, 0), outline(0, 2.6, 0)) == pytest.approx(0.5)
    # At right angles: a's side at x=1.05, b's tail at x=1.6
    assert rectangles_gap(outline(0, 0, 90), outline(4.0, 0, 0)) == pytest.approx(0.55)


def test_measured_pet():
    assert measured_pet((1.0, 2.0), (5.0, 6.0)) == pytest.approx(3.0)
    assert measured_pet((5.0, 6.0), (1.0, 2.0)) == pytest.approx(3.0)
    assert measured_pet((1.0, 4.0), (3.0, 6.0)) == pytest.approx(-1.0)  # in the zone together
    assert measured_pet(None, (1.0, 2.0)) is None


def sample(t, tx, ox, braked=False):
    return Sample(t, tx, 0.0, 5.0, 0.0, ox, 0.0, -5.0, 0.0, braked)


def test_zone_times():
    samples = [sample(float(t), float(t) * 2, 100.0) for t in range(10)]
    assert zone_times(samples, "turner_x", "turner_y", (10.0, 0.0)) == (4.0, 6.0)
    assert zone_times(samples, "oncoming_x", "oncoming_y", (10.0, 0.0)) is None


def test_analyse_records_the_brake_moment():
    samples = [sample(0.0, 0.0, 30.0), sample(1.0, 5.0, 25.0, braked=True), sample(2.0, 5.0, 20.0, braked=True)]
    result = analyse("unsafe", samples, (50.0, 50.0))
    assert result.brake_fired_at == 1.0
    assert result.separation_when_braked == pytest.approx(20.0)
    assert result.turner_entered_zone is False
    assert result.pet is None


@pytest.mark.parametrize("change,reason", [
    ({"contact": True, "min_separation": 1.0}, "contact"),
    ({"min_ttc": 1.0}, "TTC"),
    ({"pet": 0.5}, "PET"),
    ({"min_gap": 0.4}, "near miss"),
    ({"max_drac": 5.0}, "DRAC"),
    ({"brake_fired_at": 3.0}, "brake"),
])
def test_each_threshold_alone_makes_a_run_unsafe(change, reason):
    verdict, reasons = classify(Result("x", **change))
    assert verdict == "UNSAFE"
    assert any(reason in r for r in reasons)


def test_a_clear_pass_is_not_a_near_miss():
    assert classify(Result("x", pet=3.0, min_gap=1.5))[0] == "SAFE"


def test_a_clean_run_is_safe():
    assert classify(Result("x", pet=3.0, min_ttc=math.inf, max_drac=1.0, min_separation=5.0)) == ("SAFE", [])


def test_unbraked_plans_separate_the_two_cases():
    turner, oncoming = left_turn_paths()
    conflict = conflict_point(turner, oncoming)
    unsafe = replay_without_brake("unsafe", turner, oncoming, 20.0, conflict)
    safe = replay_without_brake("safe", turner, oncoming, 20.0, conflict)
    assert unsafe.verdict == "UNSAFE"
    assert safe.verdict == "SAFE"
    assert safe.pet > 1.5
    assert unsafe.pet < safe.pet
