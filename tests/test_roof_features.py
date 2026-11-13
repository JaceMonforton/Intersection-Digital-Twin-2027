"""The roof LiDAR's frame: sweeps and boxes centred on the car, x forward and y left. No CARLA needed."""
import math

import numpy as np
import pytest
from detection_features import MODEL_GROUND_Z, PEDESTRIAN, VEHICLE
from roof_features import (
    ROOF_HEIGHT,
    actor_label,
    model_to_world,
    sweep_to_model,
    world_to_sensor,
)


def test_a_sweep_lands_with_y_left_the_road_at_the_ground_and_the_own_car_cut_out():
    raw = np.array([[10.0, 2.0, -ROOF_HEIGHT, 1.0],   # on the road 10 m ahead, 2 m to the right
                    [1.0, 0.2, -1.0, 1.0]], np.float32)  # the car's own bonnet
    points = sweep_to_model(raw.ravel())
    assert points.shape == (1, 3)
    assert points[0] == pytest.approx((10.0, -2.0, MODEL_GROUND_Z))


def test_a_car_ahead_and_right_of_a_car_heading_north_in_carla_is_labelled_ahead_and_right():
    sensor = (100.0, 50.0, ROOF_HEIGHT, 90.0)  # facing +y in CARLA, which is south on a map but "north" here
    other = (100.0 - 3.0, 50.0 + 20.0, 0.8, 2.3, 1.0, 0.8, 90.0, 0.0, 5.0, False)  # 20 m ahead, 3 m to its right
    row = actor_label(sensor, other)
    assert row[0] == pytest.approx(20.0) and row[1] == pytest.approx(-3.0)  # y left, so right is negative
    assert row[2] == pytest.approx(0.8 + MODEL_GROUND_Z)
    assert row[6] == pytest.approx(0.0, abs=1e-9) and row[7] == VEHICLE  # same heading as the sensor
    assert row[9] == pytest.approx(5.0) and row[10] == pytest.approx(0.0, abs=1e-9)  # moving ahead, as the sensor faces
    assert model_to_world(row, sensor) == pytest.approx((97.0, 70.0))


def test_walkers_are_pedestrians_and_far_actors_are_off_the_grid():
    sensor = (0.0, 0.0, ROOF_HEIGHT, 0.0)
    assert actor_label(sensor, (5.0, 0.0, 0.9, 0.3, 0.3, 0.9, 0.0, 0.0, 0.0, True))[7] == PEDESTRIAN
    assert actor_label(sensor, (61.0, 0.0, 0.9, 2.0, 1.0, 0.8, 0.0, 0.0, 0.0, False)) is None


def test_world_to_sensor_turns_by_the_heading():
    assert world_to_sensor(0.0, 10.0, 90.0) == pytest.approx((10.0, 0.0))
    assert world_to_sensor(10.0, 0.0, 90.0) == pytest.approx((0.0, -10.0))
    assert math.hypot(*world_to_sensor(3.0, 4.0, 33.0)) == pytest.approx(5.0)
