"""The LiDAR window's feed and sums. Needs nothing running, and no window opens."""
import sys
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import lidar_view
from central_gpu_processor import Lidar, _sweep_for, merged_sweep, sweeps_by_lidar
from detection_features import LABEL_SIZE, MIN_POINTS, MODEL_GROUND_Z, PEDESTRIAN
from lidar_view import SweepFeed, SweepRate, box_corners, points_on_boxes, to_screen

MOUNTS = ((14.1, -17.2), (-14.1, 17.2))


@pytest.fixture
def feed(monkeypatch):
    monkeypatch.setattr(lidar_view, "FEED_NAME", f"lidar_view_test_{id(monkeypatch)}")
    memory = SweepFeed.create()
    yield memory
    memory.destroy()


def sweep(count, x=5.0):
    return np.column_stack((np.full(count, x), np.zeros(count), np.full(count, MODEL_GROUND_Z + 1.0), np.zeros(count)))


def test_a_reader_gets_what_was_written_with_each_lidar_kept_apart(feed):
    detection = np.arange(9, dtype=np.float32)
    true_box = np.arange(LABEL_SIZE, dtype=np.float32)
    feed.write(100, 10.0, MOUNTS, [sweep(3, 5.0), sweep(2, -5.0)], detection[None], true_box[None])
    reader = SweepFeed.attach()
    state = reader.read()
    assert state["frame"] == 100 and state["time"] == 10.0
    assert [len(s) for s in state["sweeps"]] == [3, 2]
    assert state["sweeps"][0][0, 0] == pytest.approx(5.0) and state["sweeps"][1][0, 0] == pytest.approx(-5.0)
    assert np.array_equal(state["detections"][0], detection) and np.array_equal(state["true_boxes"][0], true_box)
    assert state["mounts"][0, 1] == pytest.approx(-17.2)
    assert reader.read() is None  # nothing new
    reader.close()


def test_a_lidar_that_stops_shows_up_as_falling_behind(feed):
    feed.write(100, 10.0, MOUNTS, [sweep(3), sweep(3)], [], [])
    feed.write(101, 10.1, MOUNTS, [sweep(3), None], [], [])
    reader = SweepFeed.attach()
    state = reader.read()
    assert state["last_frames"] == (101, 100) and state["totals"] == (2, 1)
    assert [len(s) for s in state["sweeps"]] == [3, 0]
    reader.close()


def test_a_read_during_a_write_is_thrown_away(feed):
    feed.write(100, 10.0, MOUNTS, [sweep(3), sweep(3)], [], [])
    reader = SweepFeed.attach()
    feed.memory.buf[0] = feed.memory.buf[0] | 1  # the mark a writer leaves while it's writing
    assert reader.read() is None
    reader.close()


def test_too_many_points_or_boxes_are_cut_rather_than_breaking_the_tick(feed, monkeypatch):
    feed.write(100, 10.0, MOUNTS, [sweep(lidar_view.MAX_POINTS + 5), sweep(1)],
               np.zeros((lidar_view.MAX_DETECTIONS + 5, 9)), np.zeros((lidar_view.MAX_TRUE + 5, LABEL_SIZE)))
    reader = SweepFeed.attach()
    state = reader.read()
    assert len(state["sweeps"][0]) == lidar_view.MAX_POINTS
    assert len(state["detections"]) == lidar_view.MAX_DETECTIONS and len(state["true_boxes"]) == lidar_view.MAX_TRUE
    reader.close()


def test_the_feed_is_there_only_once_made(monkeypatch):
    monkeypatch.setattr(lidar_view, "FEED_NAME", "lidar_view_test_never_made")
    assert SweepFeed.attach() is None


def test_a_second_processor_reuses_a_block_a_window_still_holds(feed):
    second = SweepFeed.create()
    second.write(7, 0.7, MOUNTS, [sweep(1), sweep(1)], [], [])
    assert feed.read()["frame"] == 7
    second.close()


def test_the_block_is_as_big_as_its_arrays_need():
    block = shared_memory.SharedMemory(create=True, size=SweepFeed.size())
    try:
        feed = SweepFeed(block)  # numpy refuses arrays that overrun their buffer
        feed.close()
    finally:
        block.unlink()


def test_screen_has_forward_up_and_left_left():
    assert to_screen(0, 0, 800) == (400, 400)
    across, down = to_screen(60, 60, 800)
    assert (across, down) == (0, 0)


def test_box_corners_follow_the_yaw():
    row = [10, 20, 0, 4, 2, 1.5, np.pi / 2]
    xs, ys = zip(*box_corners(row))
    assert (min(xs), max(xs)) == pytest.approx((9, 11))  # length now runs along y
    assert (min(ys), max(ys)) == pytest.approx((18, 22))
    grown = box_corners(row, 1.0)
    assert max(c[1] for c in grown) == pytest.approx(23)


def test_only_points_above_the_road_count_as_landing_on_a_box():
    box = np.zeros(LABEL_SIZE, np.float32)
    box[[3, 4, 5, 7]] = 2, 2, 2, PEDESTRIAN
    on_the_road = np.tile([0.0, 0.0, MODEL_GROUND_Z], (10, 1))
    on_the_person = np.tile([0.0, 0.0, MODEL_GROUND_Z + 1.0], (MIN_POINTS, 1))
    assert points_on_boxes(box[None], [on_the_road, on_the_person]).tolist() == [MIN_POINTS]


def test_sweep_rate_is_measured_over_its_window():
    rate = SweepRate(window=2.0)
    assert rate.per_second() is None
    for tick in range(11):
        rate.add(tick * 0.5, tick * 5)
    assert rate.per_second() == pytest.approx(10.0)
    rate.add(10.0, 50)  # stops arriving
    rate.add(12.5, 50)
    assert rate.per_second() == 0.0 or rate.per_second() is None


def lidar_with(*frames):
    import queue

    sweeps = queue.Queue()
    for frame in frames:
        sweeps.put((frame, np.zeros((1, 4), np.float32)))
    return Lidar(None, (0.0, 0.0, 7.0), sweeps)


def test_sweeps_by_lidar_says_which_one_is_late_but_merged_still_refuses(monkeypatch):
    monkeypatch.setattr("central_gpu_processor.LIDAR_TIMEOUT", 0.01)
    assert _sweep_for(lidar_with().sweeps, 100) is None
    assert [s is None for s in sweeps_by_lidar([lidar_with(100), lidar_with()], 100)] == [False, True]
    assert merged_sweep([lidar_with(100), lidar_with()], 100) is None
    assert merged_sweep([lidar_with(100), lidar_with(100)], 100) is not None
