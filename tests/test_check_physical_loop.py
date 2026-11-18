"""The loop check's verdicts, from recorded poses and commands. No CARLA, no broker."""
from check_physical_loop import judge

SOURCE = "simulated stand-in"


def run(stopped_after=True, hold_last=True, resumed=True, rate=20, source=SOURCE):
    """A recording like a real run: driving until the 10-11 s outage, then (normally) held at 0."""
    poses = [(i / rate, 0.6 if i / rate < 10.0 else (0.0 if stopped_after else 0.4), source)
             for i in range(int(16 * rate)) if not 10.0 <= i / rate < 11.0]
    commands = [(t, {"session": "s", "seq": n, "steer": 0.0, "speed": 0.5}) for n, t in enumerate([p[0] for p in poses][::2], 1)
                if resumed or t < 11.0]
    if hold_last:
        commands.append((16.0, {"session": "s", "seq": len(commands) + 1, "hold": True}))
    return poses, commands


def verdicts(*args, **kwargs):
    return {name: passed for name, passed, _ in judge(*run(*args, **kwargs))}


def test_a_good_run_passes_everything():
    assert all(verdicts().values())


def test_a_car_still_moving_after_the_outage_fails():
    assert not verdicts(stopped_after=False)["silence stops it"]


def test_the_stop_only_counts_if_commands_kept_coming():
    assert not verdicts(resumed=False)["silence stops it"]  # stopping because nothing asked it to drive proves nothing


def test_no_hold_at_the_end_fails():
    assert not verdicts(hold_last=False)["a hold on exit"]


def test_slow_or_unlabelled_poses_fail():
    assert not verdicts(rate=5)["poses arrive"]
    assert not verdicts(source="camera")["poses arrive"]


def test_nothing_recorded_fails_rather_than_passing():
    assert not any(passed for _, passed, _ in judge([], []))
