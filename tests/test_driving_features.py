"""The driving model's inputs and outputs. No CARLA needed."""
import math

import numpy as np
import pytest
from driving_features import (
    ACTOR_FEATURES,
    BRAKE_DEADBAND,
    BRAKE_RATE,
    CAR_RANGE,
    CRUISE_SPEED,
    CRUISE_THROTTLE,
    FULL_BRAKE,
    HOLD_SPEED,
    HOLD_TARGET,
    INPUT_SCALE,
    INPUT_SIZE,
    LANE_POINTS,
    LANE_SPACING,
    MAX_CARS,
    MAX_WALKERS,
    RISING_TARGET,
    ROUTES,
    SOFT_STOP_BRAKE,
    SOFT_STOP_SPEED,
    STOP_LINE_RANGE,
    THROTTLE_RATE,
    WALKER_RANGE,
    SpeedController,
    decode_inputs,
    encode_inputs,
    output_from_driving,
)
from waymo_predictor_node import STRAIGHT

SPEED = 0
ROUTE = 1
STOP_PRESENT = 4
STOP_GAP = 5
LANE = 6
CARS = LANE + 2 * (LANE_POINTS + 1)
WALKERS = CARS + MAX_CARS * ACTOR_FEATURES

EGO = (100.0, 50.0, 5.0, 0.0, 0.0)  # heading along world +x at 5 m/s
ROAD = [(100.0 + i, 50.0) for i in range(40)]


def encode(ego=EGO, path=ROAD, route=STRAIGHT, stop_gap=math.inf, cars=(), walkers=()):
    return encode_inputs(ego, path, route, stop_gap, cars, walkers)


def test_size_matches_the_scale_and_is_float32():
    inputs = encode()
    assert inputs.shape == (INPUT_SIZE,) == INPUT_SCALE.shape
    assert inputs.dtype == np.float32


def test_speed_is_the_size_of_the_velocity():
    assert encode(ego=(0.0, 0.0, 3.0, -4.0, 0.0), path=[(0.0, 0.0), (1.0, 0.0)])[SPEED] == pytest.approx(5.0)


@pytest.mark.parametrize("route", ROUTES)
def test_route_is_one_hot(route):
    one_hot = encode(route=route)[ROUTE:ROUTE + len(ROUTES)]
    assert one_hot[ROUTES.index(route)] == 1.0
    assert one_hot.sum() == 1.0


def test_an_unknown_route_is_refused():
    with pytest.raises(ValueError):
        encode(route=7)


def test_lane_points_are_spaced_along_the_road_ahead():
    lane = encode()[LANE:CARS].reshape(-1, 2)
    assert lane.shape == (LANE_POINTS + 1, 2)
    assert lane[:, 0] == pytest.approx(LANE_SPACING * np.arange(LANE_POINTS + 1))
    assert lane[:, 1] == pytest.approx(0.0)


def test_first_lane_point_shows_the_car_is_off_its_lane():
    # CARLA's y points right, so a car at y=52 has the lane at y=50 on its left
    ego = (100.0, 52.0, 5.0, 0.0, 0.0)
    assert encode(ego=ego)[LANE + 1] == pytest.approx(2.0)


def test_a_road_curving_right_in_carla_is_negative_y_in_the_ego_frame():
    right_bend = [(100.0 + i, 50.0 + 0.2 * i) for i in range(40)]
    assert encode(path=right_bend)[CARS - 1] < 0.0


def test_lane_points_stop_where_the_route_ends():
    lane = encode(path=[(100.0, 50.0), (101.0, 50.0), (102.0, 50.0)])[LANE:CARS].reshape(-1, 2)
    assert lane[-1, 0] == pytest.approx(2.0)


def test_lane_points_turn_with_the_car():
    # Facing world +y, the road along +y is straight ahead
    ego = (0.0, 0.0, 0.0, 5.0, 90.0)
    lane = encode(ego=ego, path=[(0.0, float(i)) for i in range(40)])[LANE:CARS].reshape(-1, 2)
    assert lane[-1] == pytest.approx((2.0 * LANE_POINTS, 0.0), abs=1e-4)


def test_a_stop_line_in_range_is_flagged_with_its_distance():
    inputs = encode(stop_gap=12.0)
    assert inputs[STOP_PRESENT] == 1.0
    assert inputs[STOP_GAP] == 12.0


@pytest.mark.parametrize("gap", (math.inf, STOP_LINE_RANGE, STOP_LINE_RANGE + 5.0))
def test_no_stop_line_in_range_stays_finite(gap):
    inputs = encode(stop_gap=gap)
    assert inputs[STOP_PRESENT] == 0.0
    assert inputs[STOP_GAP] == STOP_LINE_RANGE


def test_a_car_is_in_the_ego_frame():
    car = (110.0, 47.0, 0.0, 2.0, 90.0)  # 10 m ahead, 3 m to the left, heading to the right
    slot = encode(cars=[car])[CARS:CARS + ACTOR_FEATURES]
    assert slot == pytest.approx((1.0, 10.0, 3.0, 0.0, -2.0))


def test_cars_are_nearest_first_and_extras_are_dropped():
    cars = [(100.0 + 3.0 * i, 50.0, 0.0, 0.0, 0.0) for i in range(MAX_CARS + 3, 0, -1)]
    slots = encode(cars=cars)[CARS:WALKERS].reshape(MAX_CARS, ACTOR_FEATURES)
    assert slots[:, 1] == pytest.approx(3.0 * np.arange(1, MAX_CARS + 1))


def test_cars_out_of_range_leave_empty_slots():
    slots = encode(cars=[(100.0 + CAR_RANGE + 1.0, 50.0, 1.0, 0.0, 0.0)])[CARS:WALKERS]
    assert not slots.any()


def test_walkers_have_their_own_slots_and_range():
    near = (105.0, 50.0, 0.0, 1.0, 0.0)
    far = (100.0 + WALKER_RANGE + 1.0, 50.0, 0.0, 1.0, 0.0)
    inputs = encode(walkers=[far, near])
    assert not inputs[CARS:WALKERS].any()
    slots = inputs[WALKERS:WALKERS + MAX_WALKERS * ACTOR_FEATURES].reshape(MAX_WALKERS, ACTOR_FEATURES)
    assert slots[0] == pytest.approx((1.0, 5.0, 0.0, 0.0, -1.0))
    assert not slots[1:].any()


def test_decoding_gets_back_what_was_encoded():
    car = (110.0, 47.0, 0.0, 2.0, 90.0)
    walker = (105.0, 50.0, 0.0, 1.0)
    seen = decode_inputs(encode(stop_gap=12.0, cars=[car], walkers=[walker]))
    assert seen["speed"] == pytest.approx(5.0)
    assert seen["route"] == STRAIGHT
    assert seen["stop_gap"] == 12.0
    assert len(seen["lane"]) == LANE_POINTS + 1
    assert seen["cars"] == [pytest.approx((10.0, 3.0, 0.0, -2.0))]
    assert seen["walkers"] == [pytest.approx((5.0, 0.0, 0.0, -1.0))]


def test_decoding_shows_no_stop_line_when_there_is_none():
    assert decode_inputs(encode())["stop_gap"] is None


def run(controller, output, speed, ticks=1):
    for _ in range(ticks):
        result = controller.step(output, speed)
    return result


def test_steering_sign_stays_carlas():
    # Positive steer is to the right in CARLA, and nothing here may flip it
    assert output_from_driving(0.4, 5.0)[0] == pytest.approx(0.4)
    assert run(SpeedController(), (0.4, 5.0), 5.0, ticks=30)[0] == pytest.approx(0.4, abs=0.01)


def test_the_target_is_the_speed_later():
    assert output_from_driving(0.1, 3.5)[1] == pytest.approx(3.5)


def test_steering_is_smoothed_not_jumped():
    assert 0.0 < SpeedController().step((0.4, 5.0), 5.0)[0] < 0.4


def test_cruising_settles_near_the_cruising_throttle():
    _, throttle, brake = run(SpeedController(), (0.0, CRUISE_SPEED), CRUISE_SPEED, ticks=60)
    assert throttle == pytest.approx(CRUISE_THROTTLE, abs=0.01)
    assert brake == 0.0


def test_below_the_target_it_accelerates_without_a_lunge():
    controller = SpeedController()
    first = controller.step((0.0, 7.0), 3.0)[1]
    later = run(controller, (0.0, 7.0), 3.0, ticks=30)[1]
    assert first <= THROTTLE_RATE
    assert later > CRUISE_THROTTLE


def test_acceleration_is_damped():
    a, b = SpeedController(), SpeedController()
    run(a, (0.0, 5.0), 3.0, ticks=30)
    run(b, (0.0, 5.0), 3.0, ticks=30)
    assert a.step((0.0, 5.0), 3.5)[1] < b.step((0.0, 5.0), 3.0)[1]


def test_a_moving_car_doesnt_brake_when_the_controller_starts_or_resets():
    controller = SpeedController()
    assert controller.step((0.0, 5.0), 5.0)[2] == 0.0
    controller.hold()
    assert controller.step((0.0, 5.0), 5.0)[2] == 0.0


def test_a_lower_target_counts_at_once_and_a_higher_one_slowly():
    controller = SpeedController()
    run(controller, (0.0, 5.0), 5.0, ticks=60)
    controller.step((0.0, 1.0), 5.0)
    assert controller.target == pytest.approx(1.0)
    controller.step((0.0, 5.0), 5.0)
    assert controller.target == pytest.approx(1.0 + RISING_TARGET * 4.0)


def test_well_above_the_target_it_brakes_harder_the_further_over_it_is():
    gentle = run(SpeedController(), (0.0, 4.0), 5.0, ticks=10)
    hard = run(SpeedController(), (0.0, 0.0), 6.0, ticks=10)
    assert gentle[1] == 0.0
    assert 0.0 < gentle[2] < hard[2]


def test_the_brake_builds_rather_than_snaps():
    assert SpeedController().step((0.0, 0.0), 6.0)[2] == pytest.approx(BRAKE_RATE)


def test_just_over_the_target_it_eases_off_without_braking():
    _, throttle, brake = run(SpeedController(), (0.0, 5.0), 5.0 + BRAKE_DEADBAND / 2, ticks=30)
    assert throttle < CRUISE_THROTTLE
    assert brake == 0.0


def test_when_stopped_a_low_target_holds_it_rather_than_creeping_at_a_car():
    _, throttle, brake = run(SpeedController(), (0.0, HOLD_TARGET - 0.1), HOLD_SPEED / 2, ticks=5)
    assert throttle == 0.0
    assert brake == pytest.approx(SOFT_STOP_BRAKE)  # a gentle last moment
    assert run(SpeedController(), (0.0, HOLD_TARGET - 0.1), 0.0) == pytest.approx((0.0, 0.0, 1.0))  # then held


def test_at_a_crawl_the_brake_is_limited_so_it_doesnt_stop_dead():
    assert run(SpeedController(), (0.0, 0.0), SOFT_STOP_SPEED - 0.1, ticks=10)[2] == pytest.approx(SOFT_STOP_BRAKE)


def test_a_clear_road_pulls_away_from_a_stop():
    _, throttle, brake = run(SpeedController(), (0.0, 4.0), 0.0, ticks=30)
    assert throttle > 0.0
    assert brake == 0.0


def test_a_negative_target_means_stop_not_reverse():
    _, throttle, brake = run(SpeedController(), (0.0, -3.0), 4.0, ticks=10)
    assert throttle == 0.0
    assert brake > 0.0


def test_controls_stay_in_carlas_range():
    steer, throttle, brake = run(SpeedController(), (3.0, 100.0), 0.0, ticks=100)
    assert steer == pytest.approx(1.0, abs=0.01)
    assert throttle <= 1.0
    assert brake == 0.0


@pytest.mark.parametrize("output", ((math.nan, 5.0), (0.1, math.nan), (math.inf, 5.0), (0.1, math.inf)))
def test_a_broken_output_brakes_fully(output):
    assert SpeedController().step(output, 5.0) == FULL_BRAKE


def test_the_safety_brake_clears_the_controllers_memory():
    controller = SpeedController()
    run(controller, (0.3, 7.0), 3.0, ticks=30)
    assert controller.hold() == FULL_BRAKE
    assert controller.integral == 0.0
    assert controller.throttle == 0.0


def test_size_matches_the_documented_contract():
    assert INPUT_SIZE == 76  # models/README.md
