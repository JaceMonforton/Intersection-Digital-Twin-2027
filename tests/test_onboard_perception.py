"""Fusing the car's own tracks with the roadside unit's, and placing its detections in the world. No CARLA needed."""
import math

import pytest

pytest.importorskip("paho")

from detection_features import PEDESTRIAN, VEHICLE
from onboard_perception import detections_to_world, fuse_tracks, sensor_error

POLES = [(14.1, 17.2), (-14.1, -17.2)]


def test_one_actor_seen_by_both_becomes_one_track_leaning_on_the_nearer_sensor():
    own = [(100.0, 0.0, 5.0, 0.0, 0.0)]    # the car's sensor is right by it
    road = [(101.0, 0.0, 6.0, 0.0, 10.0)]  # the poles are about 85 m away
    fused = fuse_tracks(own, road, (98.0, 0.0), POLES, reach=3.0)
    assert len(fused) == 1
    x, _, vx, _, yaw = fused[0]
    assert 100.0 < x < 100.2 and 5.0 < vx < 5.2  # nearly the car's own measurement
    assert yaw == 0.0  # the heading is the car's own


def test_what_only_one_source_sees_is_kept_and_far_apart_tracks_stay_apart():
    own = [(0.0, 0.0, 0.0, 0.0, 0.0)]
    road = [(10.0, 0.0, 0.0, 0.0, 0.0), (0.5, 0.0, 0.0, 0.0, 0.0)]
    fused = fuse_tracks(own, road, (0.0, -5.0), POLES, reach=3.0)
    assert len(fused) == 2 and any(f[0] == 10.0 for f in fused)
    assert fuse_tracks([], road, (0.0, 0.0), POLES, 3.0) == road
    assert fuse_tracks(own, [], (0.0, 0.0), POLES, 3.0) == own


def test_each_track_is_used_once_even_when_two_are_close():
    own = [(0.0, 0.0, 0, 0), (2.0, 0.0, 0, 0)]
    road = [(1.0, 0.0, 0, 0)]
    assert len(fuse_tracks(own, road, (0.0, 0.0), POLES, 3.0)) == 2


def test_sensor_error_grows_with_range():
    assert sensor_error(50.0) > sensor_error(5.0) > 0.0


def test_detections_land_in_the_world_and_small_boxes_are_people():
    sensor = (10.0, 20.0, 1.9, 90.0)  # facing +y in CARLA
    rows = [(5.0, 0.0, -1.0, 4.5, 1.9, 1.5, 0.0, 0.9, VEHICLE),       # 5 m straight ahead
            (3.0, 2.0, -1.0, 0.6, 0.6, 1.8, 0.0, 0.8, PEDESTRIAN),
            (8.0, 0.0, -1.0, 0.8, 0.7, 1.7, 0.0, 0.7, VEHICLE)]       # scored as a car, but the size of a person
    cars, people = detections_to_world(rows, sensor)
    assert len(cars) == 1 and len(people) == 2
    assert cars[0][:2] == pytest.approx((10.0, 25.0)) and cars[0][2] == pytest.approx(90.0)
    assert people[0][:2] == pytest.approx((12.0, 23.0))  # 2 m to the left of a car facing +y is +x in CARLA
    assert math.isfinite(people[1][2])
