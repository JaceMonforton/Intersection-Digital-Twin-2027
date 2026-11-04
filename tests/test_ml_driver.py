"""The ML car's driver. No CARLA and no model file needed."""
import math
import random

import numpy as np
import pytest
from driving_features import FULL_BRAKE, INPUT_SIZE
from ml_driver import (
    CONTACT_COOLDOWN,
    LANE_LOST,
    STOPPED_SPEED,
    DrivingPolicy,
    ModelDriver,
    StopWatch,
    brake_is_off,
    is_new_contact,
    smoothness,
)
from test_record_driving import Road, Waypoint
from traffic_monitor import LaneDriver


class FakePolicy:
    def __init__(self, output=(0.2, 5.0)):
        self.output = output
        self.seen = []

    def __call__(self, inputs):
        self.seen.append(inputs)
        return self.output


def make_driver(policy=None, find=None):
    road = Road()
    return ModelDriver(policy or FakePolicy(), LaneDriver(Waypoint(road, 0.0), random.Random(0)),
                       find or (lambda x, y: Waypoint(road, float(round(x)), y)), random.Random(0))


def controls(driver, x=1.0, y=0.0, yaw=0.0, brake_off=True):
    return driver.controls((x, y, 5.0, 0.0, yaw), [], [], [], brake_off)


def test_no_model_file_is_a_refusal(tmp_path):
    with pytest.raises(FileNotFoundError):
        DrivingPolicy(tmp_path / "missing.onnx")


@pytest.mark.parametrize("gate, released", (
    ({"emergency_brake": False}, True),
    ({"emergency_brake": True}, False),
    ({"emergency_brake": None}, False),
    ({"emergency_brake": "false"}, False),
    ({}, False),
))
def test_only_an_explicit_false_releases_the_car(gate, released):
    assert brake_is_off(gate) is released


def test_the_model_drives_the_car():
    policy = FakePolicy((0.2, 5.0))
    steer, throttle, brake = controls(make_driver(policy))  # already at 5 m/s
    assert 0.0 < steer <= 0.2  # smoothed on its way to the model's value
    assert throttle > 0.0
    assert brake == 0.0
    assert policy.seen[0].shape == (INPUT_SIZE,)


def test_smoothness_tells_jittery_driving_from_smooth():
    n = 200
    calm = smoothness(np.full(n, 0.1), np.full(n, 0.4), np.zeros(n), np.full(n, 5.0))
    shaky = smoothness(0.2 * (-1.0) ** np.arange(n), np.tile([0.5, 0.0], n // 2), np.tile([0.0, 0.5], n // 2),
                       5.0 + (-1.0) ** np.arange(n))
    assert calm["steer_change_rms"] == 0.0
    assert calm["pedal_switches_per_min"] == 0.0
    assert calm["jerk_p95"] == 0.0
    assert shaky["steer_change_rms"] > 0.1
    assert shaky["pedal_switches_per_min"] > 100.0
    assert shaky["jerk_p95"] > 100.0


def test_a_harsh_stop_is_counted_once():
    speeds = np.r_[np.full(50, 6.0), np.linspace(6.0, 0.0, 8), np.zeros(50)]
    n = len(speeds)
    result = smoothness(np.zeros(n), np.zeros(n), np.zeros(n), speeds)
    assert result["harsh_stops_per_min"] == pytest.approx(1 / (n * 0.1 / 60.0))
    assert result["hardest_brake"] > 6.0


def test_the_display_gets_the_models_real_values_and_who_is_in_control():
    class Recorder:
        def send(self, inputs, output, applied, model_driving):
            self.sent = (inputs, output, applied, model_driving)

    recorder = Recorder()
    driver = make_driver(FakePolicy((0.2, 5.0)))
    driver.telemetry = recorder
    controls(driver)
    assert recorder.sent[1] == (0.2, 5.0)
    assert recorder.sent[3] is True
    controls(driver, brake_off=False)
    assert recorder.sent[1] == (0.2, 5.0)  # still what the model said
    assert recorder.sent[2] == FULL_BRAKE  # not what the car did
    assert recorder.sent[3] is False


def test_a_low_target_speed_brakes():
    _, throttle, brake = controls(make_driver(FakePolicy((0.0, 1.0))))
    assert throttle == 0.0
    assert brake > 0.0


def test_the_gate_overrides_the_model():
    assert controls(make_driver(FakePolicy((0.2, 5.0))), brake_off=False) == FULL_BRAKE


@pytest.mark.parametrize("output", ((math.nan, 0.5), (0.1, math.inf)))
def test_a_broken_model_output_brakes_fully(output):
    assert controls(make_driver(FakePolicy(output))) == FULL_BRAKE


def test_being_lost_counts_once_per_excursion_and_finds_a_new_route():
    driver = make_driver()
    old_route = driver.lane
    controls(driver)
    assert driver.times_lost == 0
    controls(driver, x=2.0, y=LANE_LOST + 1.0)
    controls(driver, x=3.0, y=LANE_LOST + 1.0)
    assert driver.times_lost == 1
    assert driver.lane is not old_route


def test_a_lost_car_in_a_junction_keeps_its_route():
    road = Road()
    driver = make_driver(find=lambda x, y: Waypoint(road, float(round(x)), y, is_junction=True))
    old_route = driver.lane
    controls(driver, x=2.0, y=LANE_LOST + 1.0)
    assert driver.times_lost == 1
    assert driver.lane is old_route


def test_a_red_light_on_the_route_reaches_the_model():
    policy = FakePolicy()
    driver = make_driver(policy)
    driver.controls((1.0, 0.0, 5.0, 0.0, 0.0), [(11.0, 0.0)], [], [], True)
    assert policy.seen[0][4] == 1.0
    assert policy.seen[0][5] == pytest.approx(10.0, abs=1.0)


def test_other_cars_reach_the_model():
    policy = FakePolicy()
    driver = make_driver(policy)
    driver.controls((1.0, 0.0, 5.0, 0.0, 0.0), [], [(11.0, 0.0, 0.0, 0.0, 0.0)], [], True)
    assert policy.seen[0][28:31] == pytest.approx((1.0, 10.0, 0.0))


def test_one_long_contact_counts_once():
    seen = {}
    assert is_new_contact(seen, 7, 0.0)
    assert not is_new_contact(seen, 7, 1.0)
    assert not is_new_contact(seen, 7, 1.0 + CONTACT_COOLDOWN - 0.1)  # still touching
    assert is_new_contact(seen, 8, 2.0)
    assert is_new_contact(seen, 7, 3.0 + 2 * CONTACT_COOLDOWN)


def test_stop_watch_counts_only_unexplained_stops():
    watch = StopWatch()
    assert watch.update(0.0, 0.0, at_red=False) == 0.0
    assert watch.update(10.0, 0.0, at_red=False) == 10.0
    assert watch.update(20.0, 0.0, at_red=True) == 0.0       # a red light explains it
    assert watch.update(30.0, 0.0, at_red=False) == 0.0
    assert watch.update(35.0, STOPPED_SPEED, at_red=False) == 0.0
    assert watch.longest == 10.0


def test_policy_input_is_finite_with_no_traffic():
    policy = FakePolicy()
    controls(make_driver(policy))
    assert np.isfinite(policy.seen[0]).all()
