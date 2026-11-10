"""Scoring and reporting of the five-test suite. No CARLA needed."""
import io
import math
import os
import random

import numpy as np
import pytest

pytest.importorskip("paho")

from run_professor_test_suite import (
    T2_FIRST_WAVE,
    T2_GAP,
    T2_HEADWAY,
    T2_SAFE_GAP,
    Console,
    StageResult,
    TurnChooser,
    alert_banner,
    along_and_across,
    follow_path,
    grade_blind_corner,
    grade_cut_in,
    grade_left_turn,
    grade_light_change,
    grade_lights,
    latency_ms,
    launch_now,
    live_table,
    matrix_rows,
    next_car_text,
    paths_cross,
    results_table,
    round_rows,
    shown,
    stopping_distance,
    straight_back,
    traffic_schedule,
    write_summary,
    yaw_class,
)
from waymo_predictor_node import LEFT, STRAIGHT


def result(**fields):
    return StageResult(1, "A test", **fields)


def test_a_clean_stop_passes():
    assert grade_lights(result(stopping_margin=3.0), True, True, True, True).passed


@pytest.mark.parametrize("fields, flags", (
    ({"stopping_margin": -1.0}, (True, True, True, True)),  # past the line
    ({"stopping_margin": None}, (False, False, False, False)),  # never stopped
    ({"stopping_margin": 3.0, "red_light_violations": 1}, (True, True, True, True)),
    ({"stopping_margin": 3.0, "collisions": [{"t": 1}]}, (True, True, True, True)),
    ({"stopping_margin": 3.0}, (True, True, False, True)),  # didn't go on green
    ({"stopping_margin": 3.0}, (True, True, True, False)),  # never got back up to speed
))
def test_any_failure_fails_the_light_test(fields, flags):
    assert not grade_lights(result(**fields), *flags).passed


def _turn(**fields):
    return result(**{"min_gap": 3.0, **fields})


def test_a_safe_left_turn_passes():
    assert grade_left_turn(_turn(), True, True, True).passed


def test_the_left_turn_must_be_finished_without_a_collision():
    assert not grade_left_turn(_turn(), False, True, True).passed
    assert not grade_left_turn(_turn(collisions=[{"t": 1}]), True, True, True).passed


def test_a_left_turn_with_no_oncoming_traffic_tested_nothing():
    assert not grade_left_turn(_turn(), True, False, True).passed


def test_a_left_turn_must_give_way_and_keep_its_distance():
    assert not grade_left_turn(_turn(), True, True, False).passed
    assert not grade_left_turn(_turn(min_gap=T2_SAFE_GAP - 0.1), True, True, True).passed
    assert grade_left_turn(_turn(brake_events=1), True, True, True).passed  # the gate is recorded, not obeyed, so it isn't graded


def test_the_light_change_test_needs_the_light_to_really_change():
    assert grade_light_change(_turn(), True, True, True, True).passed
    assert not grade_light_change(_turn(), True, True, True, False).passed
    assert not grade_light_change(_turn(collisions=[{"t": 1}]), True, True, True, True).passed


def test_oncoming_traffic_has_a_wave_then_gaps_to_turn_through():
    times = traffic_schedule(random.Random(1))
    steps = np.diff(times)
    assert times[0] == 0.0
    assert steps.min() >= T2_HEADWAY[0]
    assert steps[np.array(times[:-1]) < T2_FIRST_WAVE - T2_HEADWAY[1]].max() <= T2_HEADWAY[1]
    assert steps.max() >= T2_GAP[0]


def test_routes_that_meet_are_found_and_others_are_not():
    mine = np.array([(float(x), 0.0) for x in range(50)])
    crossing = np.array([(20.0, float(y)) for y in range(-30, 30)])
    assert paths_cross(mine, crossing) == (20, 30)
    assert paths_cross(mine, crossing + (0.0, 100.0)) is None


def test_launch_times_arrive_together():
    assert not launch_now(60.0, 5.0, 40.0, 10.0)  # 12 s against 4 s: too soon
    assert launch_now(20.0, 5.0, 40.0, 10.0)      # 4 s each


def test_along_and_across_a_car_facing_north():
    ahead, side = along_and_across((0.0, 0.0, 0.0, 0.0, 90.0), (1.0, 10.0))
    assert ahead == pytest.approx(10.0)
    assert side == pytest.approx(-1.0)


def test_the_blind_corner_needs_the_gate_mqtt_a_stop_and_no_crash():
    assert grade_blind_corner(result(brake_at=2.0), True, True).passed
    assert not grade_blind_corner(result(), True, True).passed  # the gate never fired
    assert not grade_blind_corner(result(brake_at=2.0), False, True).passed  # not heard on MQTT
    assert not grade_blind_corner(result(brake_at=2.0), True, False).passed  # didn't stop
    assert not grade_blind_corner(result(brake_at=2.0, collisions=[{"t": 1}]), True, True).passed


def test_a_cut_in_that_never_happened_cant_pass():
    assert grade_cut_in(result(), True).passed
    assert not grade_cut_in(result(), False).passed
    assert not grade_cut_in(result(worst_off_lane=1.5), True).passed


def test_no_checks_means_no_pass():
    assert not StageResult(1, "Couldn't run").passed


def test_the_alert_says_what_happened():
    banner = alert_banner("Dynamic Cut-in", 0.94, 5.8)
    assert banner == ("[ALERT - SAFETY GATE TRIGGERED] Cause: Dynamic Cut-in | TTC: 0.94s | Actuation: MQTT Brake 100% | "
                      "Target Stopping Distance: 2.8m")
    assert "TTC: n/a" in alert_banner("x", None, 1.0)


def test_stopping_distance():
    assert stopping_distance(6.0, braking=6.0) == pytest.approx(3.0)


def test_latency_needs_a_timestamp():
    assert latency_ms("2026-01-01T00:00:00+00:00", 1767225600.25) == pytest.approx(250.0)
    assert latency_ms(None, 1.0) is None
    assert latency_ms("nonsense", 1.0) is None


@pytest.mark.parametrize("yaw, kind", ((5.0, "same"), (178.0, "oncoming"), (-175.0, "oncoming"), (90.0, "side"), (-88.0, "side")))
def test_directions(yaw, kind):
    assert yaw_class(yaw, 0.0) == kind


def _record(passed=True):
    return {"test": 1, "name": "A test", "passed": passed, "duration": 12.0, "collisions": [], "red_light_violations": 0, "stopping_margin": 2.5,
            "min_gap": None, "mqtt_latency_ms": None, "gate_poll_ms": 12.0, "checks": {"no_collision": passed}, "notes": []}


def test_the_matrix_shows_failures_and_missing_values_plainly():
    row = matrix_rows([_record(False)])[0]
    assert "**FAIL**" in row
    assert "2.50 m" in row
    assert "n/a" in row


def test_a_round_only_meets_the_target_under_the_limit():
    rounds = {"Good": {"km": 5.0, "collisions_per_km": 0.1, "red_lights_per_km": 0.0},
              "Bad": {"km": 5.0, "collisions_per_km": 0.5, "red_lights_per_km": 0.0},
              "Unmeasured": {}}
    rows = round_rows(rounds)
    assert rows[0].endswith("| yes |")
    assert rows[1].endswith("| no |")
    assert rows[2].endswith("| no |")


def test_the_summary_compares_smoothness_when_it_has_it(tmp_path):
    path = tmp_path / "summary.md"
    write_summary(path, [_record()], {}, "a note", {"model": {"jerk_p95": 8.0}, "autopilot": {"jerk_p95": 68.0}})
    text = path.read_text(encoding="utf-8")
    assert "Jerk, 95th percentile" in text
    assert "8.000" in text
    assert "too little driving to show the target is met" in text


def test_the_summary_counts_passes_and_names_what_failed(tmp_path):
    path = tmp_path / "summary.md"
    write_summary(path, [_record(True), {**_record(False), "test": 2}], {}, "a note")
    text = path.read_text(encoding="utf-8")
    assert "1 of 2 passed" in text
    assert "failed: no collision" in text
    assert "isn't AI" in text


class Lane:
    """A lane back from a stop line: straight for a while, then curving or entering a junction."""

    def __init__(self, yaw_at=lambda d: 0.0, junction_at=None):
        self.yaw_at, self.junction_at, self.d = yaw_at, junction_at, 0.0
        self.is_junction = False
        self.transform = type("T", (), {"rotation": type("R", (), {"yaw": 0.0})()})()

    def previous(self, distance):
        back = Lane(self.yaw_at, self.junction_at)
        back.d = distance
        back.transform.rotation.yaw = self.yaw_at(distance)
        back.is_junction = self.junction_at is not None and distance >= self.junction_at
        return [back]


def test_straight_back_stops_before_a_bend_or_a_junction():
    assert straight_back(Lane(), 45.0).d == 45.0
    assert straight_back(Lane(yaw_at=lambda d: 0.0 if d < 30 else 40.0), 45.0).d == 25.0
    assert straight_back(Lane(junction_at=20.0), 45.0).d == 15.0
    assert straight_back(Lane(junction_at=10.0), 45.0) is None


def test_the_scripted_car_steers_towards_its_path_and_keeps_speed():
    me = (0.0, 0.0, 5.0, 0.0, 0.0)
    right = [(float(x), 3.0) for x in range(1, 40)]  # CARLA's y points right
    steer, throttle, brake = follow_path(me, right, 5.0)
    assert steer > 0.0
    assert throttle > 0.0
    assert brake == 0.0
    assert follow_path(me, [(float(x), -3.0) for x in range(1, 40)], 5.0)[0] < 0.0


def test_the_scripted_car_brakes_when_told_to_slow():
    _, throttle, brake = follow_path((0.0, 0.0, 7.0, 0.0, 0.0), [(float(x), 0.0) for x in range(1, 40)], 3.0)
    assert throttle == 0.0
    assert brake > 0.0


class Yaw:
    def __init__(self, yaw, turned=None):
        self.transform = type("T", (), {"rotation": type("R", (), {"yaw": yaw})()})()
        self.is_junction = False
        self.turned = yaw if turned is None else turned

    def next(self, step):
        return [Yaw(self.turned)]


def test_the_chooser_takes_the_turn_once_then_goes_straight():
    straight_on, left = Yaw(0.0), Yaw(0.0, -90.0)
    chooser = TurnChooser(LEFT)
    assert chooser.choice([straight_on, left]) is left
    assert chooser.turn == STRAIGHT
    assert chooser.choice([straight_on, left]) is straight_on


class _Terminal(io.StringIO):
    def isatty(self):
        return True


def test_the_live_table_redraws_in_place_in_a_terminal():
    out = _Terminal()
    console = Console(stream=out)
    console.status(["one", "two"])
    console.last_draw = 0.0
    console.status(["three", "four"])
    assert out.getvalue() == "one\ntwo\n\033[2F\033[Jthree\nfour\n"


def test_events_print_above_the_live_table():
    out = _Terminal()
    console = Console(stream=out)
    console.status(["row"])
    console.plain("event")
    assert out.getvalue() == "row\n\033[1F\033[Jevent\nrow\n"
    console.end_status()
    assert out.getvalue().endswith("\033[1F\033[J")


def test_redirected_output_gets_no_live_table():
    out = io.StringIO()
    console = Console(stream=out)
    console.status(["row"])
    console.plain("event")
    assert out.getvalue() == "event\n"


def test_quiet_leaves_out_the_live_table():
    out = _Terminal()
    Console(quiet=True, stream=out).status(["row"])
    assert out.getvalue() == ""


def test_the_live_table_is_cut_to_fit_since_a_wrapped_line_breaks_the_redraw(monkeypatch):
    monkeypatch.setattr("shutil.get_terminal_size", lambda fallback=None: os.terminal_size((11, 24)))
    out = _Terminal()
    Console(stream=out).status(["x" * 50])
    assert out.getvalue() == "x" * 10 + "\n"


def test_the_live_table_lines_up_and_shows_missing_values():
    pairs = (("Target speed", f"{shown(math.nan, 1)} km/h"), ("Steering", "+0.02"), ("Min TTC", "1.20 s"), ("MQTT", "NOT CONNECTED"))
    rows = live_table("[TEST 1/5] A VERY LONG TITLE " * 5, 12.3, pairs)
    assert len({len(row) for row in rows}) == 1
    assert "t=12.3 s" in rows[1]
    assert "n/a km/h" in rows[3]


def test_the_results_table_has_one_row_per_test():
    rows = results_table([_record(True), {**_record(False), "test": 2, "name": "A longer name"}])
    assert len({len(row) for row in rows}) == 1
    assert len(rows) == 6
    assert "PASS" in rows[3] and "FAIL" in rows[4]


def test_the_set_off_note_says_when_no_car_was_due():
    assert next_car_text(math.inf) == "no car due"
    assert next_car_text(None) == "no car due"
    assert next_car_text(7.25) == "the next car 7.2 s away"


def test_a_terminal_too_short_for_the_live_table_goes_without(monkeypatch):
    monkeypatch.setattr("shutil.get_terminal_size", lambda fallback=None: os.terminal_size((80, 3)))
    out = _Terminal()
    Console(stream=out).status(["1", "2", "3"])
    assert out.getvalue() == ""
