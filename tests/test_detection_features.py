"""Unit tests for the roadside detector's grid and labels, and the recorder's files. No CARLA needed."""
import math
import queue
import random

import numpy as np
import pytest
from central_gpu_processor import Lidar, merged_sweep, model_box_to_world
from detection_features import (
    GRID_CELL,
    GRID_HALF,
    GRID_SIZE,
    MAX_HEIGHT,
    MODEL_GROUND_Z,
    OUTPUT_CHANNELS,
    OUTPUT_SIZE,
    PEDESTRIAN,
    VEHICLE,
    count_points,
    crop_to_grid,
    decode,
    encode_labels,
    label_row,
    points_in_box,
    rasterize,
)
from record_detection import (
    crossing_ends,
    load_sweep,
    next_route,
    save_sweep,
    steer,
    summarise,
    traffic_level,
)


def road(height=0.0):
    return MODEL_GROUND_Z + height


def test_crop_drops_what_the_grid_cannot_use():
    cloud = np.array([[1.0, 1.0, road(1.0)], [GRID_HALF + 1, 0.0, road(1.0)], [0.0, -GRID_HALF - 1, road(1.0)],
                      [1.0, 1.0, road(MAX_HEIGHT + 1)], [1.0, 1.0, road(-1.0)]], dtype=np.float32)
    assert crop_to_grid(cloud).tolist() == [pytest.approx([1.0, 1.0, road(1.0)])]


def test_rasterize_puts_points_in_the_right_cell_with_count_and_tallest_height():
    cloud = np.array([[10.0, -20.0, road(1.0)], [10.0, -20.0, road(2.5)], [10.0, -20.0, road(0.5)]], dtype=np.float32)
    grid = rasterize(cloud)
    assert grid.shape == (2, GRID_SIZE, GRID_SIZE)
    assert grid.dtype == np.float32
    row, column = round((10.0 + GRID_HALF) / GRID_CELL), round((-20.0 + GRID_HALF) / GRID_CELL)
    assert np.argwhere(grid[0] > 0).tolist() == [[row, column]]
    assert grid[1, row, column] == pytest.approx(2.5 / MAX_HEIGHT)
    assert 0.0 < grid[0, row, column] < 1.0


def test_rasterize_copes_with_an_empty_cloud_and_a_full_cell():
    assert rasterize(np.zeros((0, 3), dtype=np.float32)).sum() == 0.0
    grid = rasterize(np.tile(np.array([[0.1, 0.1, road(1.0)]], dtype=np.float32), (500, 1)))
    assert grid[0].max() == 1.0


def test_label_row_is_the_inverse_of_how_a_detection_becomes_a_box():
    # A car 3 m along CARLA's x and 4 m to the right of a junction at (100, 200), heading along +y
    row = label_row((3.0, 4.0, 0.7), (2.25, 1.0, 0.7), 90.0, (0.0, 5.0), VEHICLE)
    box = model_box_to_world(row, 100.0, 200.0, 0.0, ground_z=MODEL_GROUND_Z)
    assert (box.x, box.y, box.z) == pytest.approx((103.0, 204.0, 0.7))
    assert (box.length, box.width, box.height) == pytest.approx((4.5, 2.0, 1.4))
    assert box.yaw == pytest.approx(90.0)
    assert row[9:] == [0.0, -5.0]  # CARLA's +y velocity is -y in the model's frame


def test_points_on_a_box_are_counted_along_its_heading():
    # 2 m long each way and 1 m wide each way, at (10, 20) in the model's frame, heading along +y
    cloud = np.array([
        [10.0, 21.5, road(1.0)],  # 1.5 m ahead of its centre: counts
        [11.5, 20.0, road(1.0)],  # 1.5 m to its side: past the 1 m half-width
        [10.0, 20.0, road(0.0)],  # the road under it
        [10.0, 20.0, road(5.0)],  # high above it
    ], dtype=np.float32)
    assert points_in_box(cloud, 10.0, 20.0, math.pi / 2, 2.0, 1.0, 1.5) == 1


def test_count_points_reads_a_label_row():
    row = label_row((3.0, -4.0, 0.7), (2.0, 1.0, 0.7), 0.0, (0.0, 0.0), VEHICLE)
    cloud = np.array([[3.0, 4.0, road(1.0)], [3.0, -4.0, road(1.0)]], dtype=np.float32)  # the first is on it, as y flips
    assert count_points(row, cloud) == 1


def test_a_sweep_survives_saving(tmp_path):
    cloud = np.array([[1.234, -5.678, road(1.0), 0.5], [200.0, 0.0, road(1.0), 0.5]], dtype=np.float32)
    path = tmp_path / "1.npz"
    save_sweep(path, cloud, [label_row((1.0, 2.0, 0.5), (2.0, 1.0, 0.7), 0.0, (0.0, 0.0), VEHICLE)])
    points, labels = load_sweep(path)
    assert points.shape == (1, 3)  # the point outside the grid isn't kept
    assert points[0].tolist() == pytest.approx([1.23, -5.68, road(1.0)], abs=0.006)
    assert labels.shape == (1, 11)
    save_sweep(path, cloud, [])
    assert load_sweep(path)[1].shape == (0, 11)  # a sweep with no actors in it still loads


def test_summary_counts_only_the_actors_a_lidar_saw():
    rows = [label_row((0, 0, 0), (2, 1, 1), 0, (5.0, 0.0), VEHICLE), label_row((0, 0, 0), (2, 1, 1), 0, (0.0, 0.0), VEHICLE),
            label_row((0, 0, 0), (0.3, 0.3, 1), 0, (1.0, 0.0), PEDESTRIAN)]
    rows[0][8], rows[1][8], rows[2][8] = 40, 20, 1
    summary = summarise(rows)
    assert summary["vehicles"] == {"in_grid": 2, "seen": 2, "moving_share": 0.5, "median_points": 30.0}
    assert summary["pedestrians"]["in_grid"] == 1
    assert summary["pedestrians"]["seen"] == 0


def test_every_seed_gets_a_traffic_level():
    assert {traffic_level(seed) for seed in range(6)} == {(25, 15), (35, 30), (50, 45)}


def lidar_with(frame):
    sweeps = queue.Queue()
    sweeps.put((frame, np.array([[1.0, 2.0, -7.0, 0.5]], dtype=np.float32)))
    return Lidar(None, (0.0, 0.0, 7.0), sweeps)


def test_a_recording_only_takes_a_sweep_from_the_labels_own_frame():
    assert merged_sweep([lidar_with(100)], 100, exact=True) is not None
    assert merged_sweep([lidar_with(101)], 100, exact=True) is None
    assert merged_sweep([lidar_with(101)], 100) is not None  # live detection takes the newest instead


def test_a_pedestrian_steers_towards_its_target_until_it_arrives():
    assert steer((0.0, 0.0), (3.0, 4.0)) == pytest.approx((0.6, 0.8))
    assert steer((3.0, 3.0), (3.0, 4.0)) is None


def test_a_crossing_runs_between_the_middles_of_its_short_sides():
    # The shape of one from the map: 19 m long, and 2.5 m wide, starting on a short side
    assert crossing_ends([(0.0, 0.0), (0.0, 2.5), (-19.0, 2.5), (-19.0, 0.0)]) == ((0.0, 1.25), (-19.0, 1.25))
    # And starting on a long side
    assert crossing_ends([(0.0, 2.5), (-19.0, 2.5), (-19.0, 0.0), (0.0, 0.0)]) == ((-19.0, 1.25), (0.0, 1.25))


def test_a_pedestrian_only_goes_a_short_way_along_the_pavement():
    spots = [(3.0, 0.0), (100.0, 0.0)]
    assert next_route((0.0, 0.0), spots, [], random.Random(1)) == [(3.0, 0.0)]


def test_a_pedestrian_far_from_every_spot_heads_for_a_near_one():
    spots = [(50.0, 0.0), (60.0, 0.0), (70.0, 0.0), (900.0, 0.0)]
    assert next_route((0.0, 0.0), spots, [], random.Random(1))[0] != (900.0, 0.0)


def test_a_crossing_is_walked_from_the_nearer_kerb_and_only_when_in_reach():
    crossing = ((5.0, 0.0), (5.0, 19.0))
    picks = [next_route((0.0, 0.0), [(1.0, 0.0)], [crossing], random.Random(seed)) for seed in range(100)]
    assert [(5.0, 0.0), (5.0, 19.0)] in picks
    assert [(1.0, 0.0)] in picks
    assert sum(len(route) == 2 for route in picks) < 40  # most of the time they stay on the pavement
    far = ((500.0, 0.0), (500.0, 19.0))
    assert all(next_route((0.0, 0.0), [(1.0, 0.0)], [far], random.Random(seed)) == [(1.0, 0.0)] for seed in range(50))


def labelled(x, y, kind=VEHICLE, points=40, yaw_degrees=30.0):
    # Given in CARLA's frame, where y is right, as a recording would make it
    size = (2.25, 1.0, 0.75) if kind == VEHICLE else (0.3, 0.3, 0.9)
    row = label_row((x, y, size[2]), size, yaw_degrees, (0.0, 0.0), kind)
    row[8] = points
    return row


def test_decoding_perfect_targets_gives_the_labels_back():
    labels = np.array([labelled(10.3, -20.2), labelled(-30.1, 15.7, PEDESTRIAN)], dtype=np.float32)
    target, weight, centres = encode_labels(labels)
    assert target.shape == (OUTPUT_CHANNELS, OUTPUT_SIZE, OUTPUT_SIZE)
    assert centres.sum() == 2
    assert weight.min() == 1.0
    rows = decode(target, 0.5)
    assert len(rows) == 2
    for label in labels:
        found = rows[rows[:, 8] == label[7]][0]
        assert found[:6] == pytest.approx(label[:6], abs=1e-3)
        assert math.sin(2 * found[6]) == pytest.approx(math.sin(2 * label[6]), abs=1e-3)
        assert found[7] == pytest.approx(1.0)


def test_an_actor_too_hidden_to_see_is_neither_a_target_nor_a_mistake():
    target, weight, centres = encode_labels(np.array([labelled(10.0, 10.0, points=1)], dtype=np.float32))
    assert target.max() == 0.0
    assert centres.sum() == 0
    assert weight[VEHICLE].min() == 0.0
    assert weight[PEDESTRIAN].min() == 1.0  # and only for its own class


def test_a_seen_actor_beside_a_hidden_one_is_still_a_target():
    labels = np.array([labelled(10.0, 10.0), labelled(10.5, 10.5, points=0)], dtype=np.float32)
    target, weight, _ = encode_labels(labels)
    assert weight[VEHICLE][target[VEHICLE] > 0.5].min() == 1.0


def test_decode_takes_one_box_per_peak_and_respects_each_class_threshold():
    target, _, _ = encode_labels(np.array([labelled(0.0, 0.0), labelled(20.0, 0.0, PEDESTRIAN)], dtype=np.float32))
    target[PEDESTRIAN] *= 0.4
    assert len(decode(target, 0.5)) == 1
    assert len(decode(target, 0.3)) == 2
    assert len(decode(target, (0.5, 0.3))) == 2
    assert len(decode(target, (1.1, 0.3))) == 1


def test_an_empty_scene_decodes_to_nothing():
    assert decode(np.zeros((OUTPUT_CHANNELS, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32), 0.5).shape == (0, 9)
    assert encode_labels(np.zeros((0, 11), dtype=np.float32))[2].sum() == 0
