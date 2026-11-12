"""The braking measurement's arithmetic. No CARLA needed."""
from measure_braking import braking


def test_a_steady_stop_measures_its_own_deceleration():
    speeds = [10.0 - 0.5 * n for n in range(21)]  # 5 m/s^2 at 0.1 s a tick, stopped after 2 s
    mean, hardest, seconds, metres = braking(speeds)
    assert abs(mean - 5.0) < 1e-6 and abs(hardest - 5.0) < 1e-6
    assert abs(seconds - 2.0) < 1e-6 and abs(metres - 10.0) < 1e-6
