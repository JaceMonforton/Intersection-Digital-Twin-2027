"""The ML car view's helpers. Skipped where carla or pygame isn't installed."""
import numpy as np
import pytest

pytest.importorskip("carla")
pytest.importorskip("pygame")

from ml_car_view import (
    VIEW_HEIGHT,
    VIEW_WIDTH,
    bgra_to_rgb,
    speed_from_positions,
    status_text,
    window_size,
)


def test_bgra_is_converted_to_rgb():
    raw = np.array([[[10, 20, 30, 255], [40, 50, 60, 255]]], dtype=np.uint8).tobytes()
    assert bgra_to_rgb(raw, 2, 1).tolist() == [[[30, 20, 10], [60, 50, 40]]]


def test_window_fits_both_views_side_by_side():
    width, height = window_size()
    assert width >= 2 * VIEW_WIDTH
    assert height > VIEW_HEIGHT


def test_speed_comes_from_how_far_the_car_moved():
    assert speed_from_positions((0.0, 0.0), (3.0, 4.0), 0.5) == pytest.approx(10.0)
    assert speed_from_positions(None, (3.0, 4.0), 0.5) == 0.0
    assert speed_from_positions((0.0, 0.0), (3.0, 4.0), 0.0) == 0.0


@pytest.mark.parametrize("speed,braking,expected", [
    (10.0, False, "driving"), (0.0, False, "stopped"), (10.0, True, "EMERGENCY BRAKE"),
])
def test_status_text(speed, braking, expected):
    assert expected in status_text(speed, braking)


def test_status_shows_km_per_hour():
    assert "36.0 km/h" in status_text(10.0, False)
