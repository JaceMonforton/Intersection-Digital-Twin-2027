"""Vehicle outlines and the gap between them, used to score how close cars came. No CARLA needed."""
import pytest
from driving_features import outline, rectangles_gap, rectangles_overlap


@pytest.mark.parametrize("b,expected", [
    ((0, 0, 0), True),        # same spot
    ((4, 0, 0), True),        # nose to tail, 0.8 m overlap
    ((6, 0, 0), False),       # nose to tail, clear
    ((0, 2.0, 0), True),      # side by side, 0.1 m overlap
    ((0, 2.3, 0), False),     # side by side, clear
    ((3, 3, 90), True),       # T-bone: b's rear corner reaches a's side
    ((5, 5, 45), False),      # corners would overlap as circles, not as rectangles
])
def test_rectangles_overlap(b, expected):
    assert rectangles_overlap(outline(0, 0, 0), outline(*b)) is expected


def test_rectangles_gap():
    assert rectangles_gap(outline(0, 0, 0), outline(0, 0, 0)) == 0.0
    # Nose to tail: 4.8 m long, centres 5.8 m apart
    assert rectangles_gap(outline(0, 0, 0), outline(5.8, 0, 0)) == pytest.approx(1.0)
    # Side by side: 2.1 m wide, centres 2.6 m apart
    assert rectangles_gap(outline(0, 0, 0), outline(0, 2.6, 0)) == pytest.approx(0.5)
    # At right angles: a's side at x=1.05, b's tail at x=1.6
    assert rectangles_gap(outline(0, 0, 90), outline(4.0, 0, 0)) == pytest.approx(0.55)

