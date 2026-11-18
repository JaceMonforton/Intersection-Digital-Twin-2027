"""The car-side connection check's verdicts. No car, no board, no broker."""
import json
import math

import pytest

pytest.importorskip("paho")

from check_rosmaster import (
    BATTERY_FULL,
    BATTERY_LOW,
    Result,
    exit_code,
    gate_running,
    judge_battery,
    judge_encoders,
    judge_imu,
    judge_version,
    parse_ping,
    ping_reply,
)


# --- The board ---
def test_a_failed_library_call_is_a_warning_not_a_broken_car():
    for judge in (judge_battery, judge_version, judge_imu):
        result = judge(None, "AttributeError: no such method")
        assert result.status == "WARN" and "unverified" in result.detail


def test_the_battery_is_judged_against_its_range():
    assert judge_battery(BATTERY_FULL - 0.3, None).status == "PASS"
    assert judge_battery(BATTERY_LOW + 0.1, None).status == "WARN"  # works, but charge it
    assert judge_battery(BATTERY_LOW - 1.0, None).status == "FAIL"
    assert judge_battery(0.0, None).status == "FAIL"  # the board answered with nothing real
    assert judge_battery(float("nan"), None).status == "FAIL"


def test_a_version_is_anything_the_board_actually_answers():
    assert judge_version(3.5, None).status == "PASS"
    assert judge_version(-1, None).status == "FAIL"
    assert judge_version(None, None).status == "FAIL"


def test_the_imu_must_be_finite_and_read_the_car_as_roughly_level():
    assert judge_imu((1.0, -2.0, 140.0), None).status == "PASS"
    assert judge_imu((50.0, 0.0, 0.0), None).status == "WARN"  # tilted: on its side, or a bad mount
    assert judge_imu((math.nan, 0.0, 0.0), None).status == "FAIL"
    assert judge_imu((0.0, 0.0), None).status == "FAIL"


def test_encoders_pass_only_if_a_wheel_really_turned():
    assert judge_encoders((0, 0, 0, 0), (0, 152, 0, 0), None).status == "PASS"
    unmoved = judge_encoders((5, 5, 5, 5), (5, 5, 5, 5), None)
    assert unmoved.status == "WARN" and "turn a wheel" in unmoved.detail  # not a pass: nothing was shown to work
    assert judge_encoders(None, None, "serial timeout").status == "WARN"


# --- The verdict ---
def test_any_failure_fails_and_nothing_passed_isnt_healthy():
    assert exit_code([Result("PASS", "a", ""), Result("WARN", "b", "")]) == 0
    assert exit_code([Result("PASS", "a", ""), Result("FAIL", "b", "")]) == 1
    assert exit_code([Result("SKIP", "a", ""), Result("SKIP", "b", "")]) == 2
    assert exit_code([]) == 2


# --- Pings ---
def test_only_a_well_formed_ping_is_answered():
    assert parse_ping(json.dumps({"run": "abc", "n": 3}).encode()) == ("abc", 3)
    for payload in (b"not json", b"[]", json.dumps({"run": "abc"}).encode(), json.dumps({"run": 4, "n": 3}).encode(),
                    json.dumps({"run": "abc", "n": "3"}).encode(), json.dumps({"run": "abc", "n": True}).encode()):
        assert parse_ping(payload) is None


def test_the_reply_says_who_was_listening_and_echoes_only_the_ping():
    reply = json.loads(ping_reply("abc", 3, {"battery": "PASS 12.1 V"}, heard_gate=True, heard_drive=0, gate_process=False))
    assert (reply["run"], reply["n"]) == ("abc", 3)
    assert reply["report"]["listener"].startswith("check_rosmaster.py")  # not the gate
    assert reply["report"]["gate_process_running"] is False and reply["report"]["heard_gate_state"] is True


# --- The gate ---
def test_the_gate_is_found_among_the_running_processes(tmp_path):
    for pid, command in ((101, b"python3\0scripts/physical_ml_subscriber.py\0--driver\0rosmaster\0"), (102, b"bash\0")):
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "cmdline").write_bytes(command)
    assert gate_running(tmp_path) is True
    (tmp_path / "101" / "cmdline").write_bytes(b"python3\0scripts/check_rosmaster.py\0")
    assert gate_running(tmp_path) is False


def test_without_a_process_list_the_gate_is_unknown(tmp_path):
    assert gate_running(tmp_path / "missing") is None
