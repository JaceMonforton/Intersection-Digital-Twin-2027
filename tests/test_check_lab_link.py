"""The laptop-side connection check's verdicts. No car, no broker."""
import pytest

pytest.importorskip("paho")

from check_lab_link import LATE_AFTER, judge_latency, judge_pose


def pings(count=50, spacing=0.1):
    return {n: 100.0 + n * spacing for n in range(count)}


def answered(sent, delay=0.03, lost=(), extra=None):
    replies = {n: t + delay for n, t in sent.items() if n not in lost}
    replies.update(extra or {})
    return replies


def test_a_quick_link_with_nothing_lost_passes():
    sent = pings()
    assert judge_latency(sent, answered(sent)).status == "PASS"


def test_a_slow_tail_or_a_couple_lost_is_a_warning():
    sent = pings()
    slow = answered(sent, extra={n: sent[n] + 0.2 for n in range(10)})  # p95 above 0.15 s, none over 0.3 s
    assert judge_latency(sent, slow).status == "WARN"
    assert judge_latency(sent, answered(sent, lost=(3, 9))).status == "WARN"


def test_replies_slower_than_the_watchdog_or_many_lost_fail():
    sent = pings()
    over = answered(sent, extra={n: sent[n] + 0.35 for n in range(5)})
    result = judge_latency(sent, over)
    assert result.status == "FAIL" and "0.3" in result.detail
    assert judge_latency(sent, answered(sent, lost=range(10))).status == "FAIL"


def test_late_replies_count_as_lost():
    sent = pings()
    late = answered(sent, extra={n: sent[n] + LATE_AFTER + 0.5 for n in range(10)})
    assert "10 lost" in judge_latency(sent, late).detail


def test_no_replies_at_all_fails_rather_than_passing():
    result = judge_latency(pings(), {})
    assert result.status == "FAIL" and "check_rosmaster.py" in result.detail


def test_poses_pass_at_the_rate_and_only_if_they_read():
    window = 5.0
    assert judge_pose(valid=100, invalid=0, window=window).status == "PASS"
    assert judge_pose(valid=40, invalid=0, window=window).status == "FAIL"  # 8 Hz
    assert judge_pose(valid=100, invalid=3, window=window).status == "FAIL"  # some don't read as a pose
    none = judge_pose(valid=0, invalid=0, window=window)
    assert none.status == "FAIL" and "camera" in none.detail
