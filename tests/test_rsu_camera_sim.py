import math
import random
import statistics

from rsu_camera_sim import (
    CAMERA_RANGE,
    CAR_RECALL,
    camera_observe,
    depth_sigma,
    person_recall,
)

MOUNTS = [(0.0, 0.0)]


def test_the_same_seed_gives_the_same_tracks():
    actors = [(10.0, 5.0, 90.0, False), (20.0, -3.0, 0.0, True)]
    assert camera_observe(actors, MOUNTS, random.Random(4)) == camera_observe(actors, MOUNTS, random.Random(4))


def test_out_of_range_is_never_reported():
    rng = random.Random(0)
    assert all(not camera_observe([(CAMERA_RANGE + 1.0, 0.0, 0.0, False)], MOUNTS, rng) for _ in range(200))


def test_depth_error_grows_with_range_and_lies_along_the_line_of_sight():
    rng = random.Random(1)

    def errors(distance):
        reports = [r for _ in range(3000) for r in camera_observe([(distance, 0.0, 0.0, False)], MOUNTS, rng)]
        return [r[1] - distance for r in reports], [r[2] for r in reports]

    near_depth, _ = errors(10.0)
    far_depth, far_side = errors(50.0)
    assert statistics.pstdev(far_depth) > 5 * statistics.pstdev(near_depth)
    assert abs(statistics.pstdev(far_depth) - depth_sigma(50.0)) < 0.1 * depth_sigma(50.0)
    assert statistics.pstdev(far_side) < 0.2  # across the line, it stays accurate however far


def test_people_are_missed_more_than_cars_and_more_when_far():
    rng = random.Random(2)

    def recall(x, is_person):
        return sum(bool(camera_observe([(x, 0.0, 0.0, is_person)], MOUNTS, rng)) for _ in range(4000)) / 4000

    assert abs(recall(5.0, False) - CAR_RECALL) < 0.02
    assert abs(recall(50.0, True) - person_recall(50.0)) < 0.03
    assert recall(50.0, True) < recall(5.0, True)


def test_the_nearest_mount_sees_it_and_every_value_is_finite():
    rng = random.Random(3)
    mounts = [(0.0, 0.0), (100.0, 0.0)]
    reports = [r for _ in range(500) for r in camera_observe([(95.0, 0.0, 0.0, False), (0.0, 0.0, 0.0, True)], mounts, rng)]
    assert reports and all(math.isfinite(v) for r in reports for v in r[1:])
    assert all(abs(r[1] - 95.0) < 1.0 for r in reports if r[0] == 0)  # 5 m from the second mount, not 95 m from the first
