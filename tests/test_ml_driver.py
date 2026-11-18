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


class ScriptedPolicy:
    def __init__(self):
        self.tick = 0

    def __call__(self, inputs):
        self.tick += 1
        return (0.05 * ((self.tick % 7) - 3), 3.0 + (self.tick % 5))


def scripted_drive(driver):
    """40 ticks with a car closing in ahead, a person by the road, two held ticks and a flickering roadside link."""
    out, x, speed = [], 1.0, 4.0
    for tick in range(40):
        cars = [(x + 30.0 - 0.8 * tick, 0.0, 0.0, 0.0, 0.0)] if 10 <= tick < 30 else []
        walkers = [(x + 15.0, 3.0, 0.0, -1.2)] if 20 <= tick < 25 else []
        brake_off = tick not in (33, 34)
        link_up = None if tick < 36 else (tick % 2 == 0)
        out.append(tuple(float(v) for v in driver.controls((x, 0.0, speed, 0.0, 0.0), [], cars, walkers, brake_off, link_up)))
        x += speed * 0.1
    return out


# Recorded from controls() before it was split into decide() and the speed controller, so the bench drives exactly
# as it did when its results were measured. Regenerate only for a change meant to alter how the car drives.
GOLDEN_DRIVE = [
    (-0.05, 0.08, 0.0), (-0.05, 0.16, 0.0), (-0.025, 0.24, 0.0), (0.012500000000000004, 0.32, 0.0), (0.05625, 0.0, 0.2),
    (0.10312500000000002, 0.0, 0.17000000000000004), (-0.0234375, 0.0, 0.11450000000000005), (-0.06171875, 0.08, 0.0),
    (-0.055859375, 0.16, 0.0), (-0.0279296875, 0.0, 0.2), (0.011035156622529026, 0.0, 0.3199999809265137),
    (0.055517579056322576, 0.0, 0.3199999809265137), (0.10275879250839354, 0.0, 0.3199999809265137),
    (-0.02362060672603547, 0.0, 0.3199999809265137), (-0.061810304108075795, 0.0, 0.3199999809265137),
    (-0.05590515242656693, 0.0, 0.3199999809265137), (-0.027952576213283464, 0.0, 0.3199999809265137),
    (0.0110237122658873, 0.0, 0.5199999809265137), (0.05551185687800171, 0.0, 0.6000000000000001),
    (0.10275593141923309, 0.0, 0.6000000000000001), (-0.02362203727061571, 0.0, 0.6000000000000001),
    (-0.061811019380365914, 0.0, 0.6000000000000001), (-0.05590551006271199, 0.0, 0.6000000000000001),
    (-0.027952755031355993, 0.0, 0.6000000000000001), (0.011023622856851035, 0.0, 0.6000000000000001),
    (0.05551181217348358, 0.0, 0.6000000000000001), (0.10275590906697403, 0.0, 0.6000000000000001),
    (-0.023622048446745217, 0.0, 0.8), (-0.06181102496843067, 0.0, 0.8), (-0.05590551285674436, 0.0, 0.8),
    (-0.02795275642837218, 0.0, 0.68), (0.011023621785813911, 0.0, 0.548), (0.05551181089290696, 0.0, 0.4058),
    (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0),
    (0.05, 0.0, 0.2),
]


def planned_driver():
    from planner import Planner
    from safety_monitor import SafetyMonitor

    driver = make_driver(ScriptedPolicy())
    driver.planner, driver.monitor = Planner((500.0, 500.0)), SafetyMonitor()
    return driver


def test_direct_driving_is_exactly_what_the_bench_measured():
    assert scripted_drive(planned_driver()) == GOLDEN_DRIVE


def test_decide_gives_the_steering_and_target_speed_that_controls_then_applies():
    driver = planned_driver()
    inputs, output, drive, allowed = driver.decide((1.0, 0.0, 4.0, 0.0, 0.0), [], [], [], True)
    assert inputs.shape == (INPUT_SIZE,) and allowed
    assert float(drive[0]) == pytest.approx(output[0])  # the model's steering; the planner only touches the speed
    held = planned_driver()
    held.decide((1.0, 0.0, 4.0, 0.0, 0.0), [], [], [], True)
    assert held.decide((1.4, 0.0, 4.0, 0.0, 0.0), [], [], [], False)[3] is False  # the brake holds it


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
        def send(self, inputs, output, applied, model_driving, plan=None):
            self.sent = (inputs, output, applied, model_driving, plan)

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


def test_with_a_planner_the_controller_gets_its_speed_but_the_display_still_gets_the_models():
    from planner import Plan

    class Slower:
        def plan(self, ego, route, cars, walkers, model_target):
            return Plan(1.0, "stop", True, "a person ahead")

    class Recorder:
        def send(self, inputs, output, applied, model_driving, plan=None):
            self.sent = (output, applied, plan)

    alone = controls(make_driver(FakePolicy((0.2, 5.0))))
    driver = make_driver(FakePolicy((0.2, 5.0)))
    driver.planner, driver.telemetry = Slower(), Recorder()
    planned = controls(driver)
    assert planned[2] > alone[2] or planned[1] < alone[1]  # it brakes harder or opens the throttle less
    output, _, plan = driver.telemetry.sent
    assert tuple(output) == (0.2, 5.0) and plan.option == "stop"


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
