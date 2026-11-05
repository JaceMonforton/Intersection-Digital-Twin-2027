"""Unit tests for the LiDAR coverage check. No CARLA needed."""
import numpy as np
from check_lidar_coverage import (
    bev_image,
    corner_mounts,
    coverage_image,
    is_clear,
    rank_pairs,
)


def test_corners_sit_outside_the_junction_at_the_height_asked_for():
    corners = corner_mounts(10.0, 20.0, 3.0, 7.0)
    assert sorted(corners.values()) == [(-13.0, -23.0, 7.0), (-13.0, 23.0, 7.0), (13.0, -23.0, 7.0), (13.0, 23.0, 7.0)]


def test_a_ray_is_only_blocked_by_something_short_of_its_target():
    target = (10.0, 0.0, 1.0)
    assert is_clear([], target)
    assert is_clear([(10.2, 0.0, 1.0)], target)
    assert not is_clear([(5.0, 0.0, 1.0), (10.0, 0.0, 1.0)], target)


def test_pairs_rank_by_what_either_sees_then_by_what_both_see():
    seen = {"A": np.array([True, True, False, False]),
            "B": np.array([False, False, True, True]),
            "C": np.array([True, True, True, False])}
    ranked = rank_pairs(seen)
    assert ranked[0] == (1.0, 0.25, ("B", "C"))
    assert ranked[1] == (1.0, 0.0, ("A", "B"))
    assert ranked[2] == (0.75, 0.5, ("A", "C"))


def test_bev_has_x_up_and_carla_y_right_like_the_overhead_view():
    # In the model's frame y is left, so a point at model y = -20 is on the image's right
    points = np.array([[40.0, 0.0, -1.8, 0.0], [0.0, -20.0, -0.5, 0.0]], dtype=np.float32)
    image = bev_image(points, half_size=50.0, cell=1.0, ground_z=-1.8)
    assert image.shape == (100, 100, 3)
    assert image[9, 50].tolist() == [90, 90, 90]  # the road point, near the top
    assert image[49, 70].tolist() == [255, 150, 30]  # the raised point, to the right of centre


def test_coverage_colours_a_road_point_by_how_many_corners_see_it():
    road = [(40.0, 0.0), (0.0, 20.0), (-40.0, 0.0)]
    image = coverage_image(road, [True, True, False], [True, False, False], half_size=50.0, cell=1.0)
    scale = image.shape[0] // 100
    assert image[9 * scale, 50 * scale].tolist() == [40, 180, 70]
    assert image[49 * scale, 70 * scale].tolist() == [230, 200, 40]
    assert image[89 * scale, 50 * scale].tolist() == [220, 40, 40]
