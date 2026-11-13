"""The ML car window's model display. pygame runs with no screen, and no CARLA is needed."""
import math
import os
import socket

import numpy as np
import pytest
from driving_features import INPUT_SIZE, decode_inputs, encode_inputs
from model_panel import (
    STALE_SECONDS,
    TelemetryReceiver,
    TelemetrySender,
    control_status,
    is_alert,
    lane_to_pixels,
    safety_lines,
    seen_lines,
    wrap_text,
)
from waymo_predictor_node import LEFT

pytest.importorskip("pygame")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

EGO = (100.0, 50.0, 5.0, 0.0, 0.0)
ROAD = [(100.0 + i, 50.0) for i in range(40)]
THIRD_PERSON = {"location": {"x": -6.5, "y": 0.0, "z": 2.8}, "pitch": -12.0}
FIRST_PERSON = {"location": {"x": 0.15, "y": -0.35, "z": 1.25}, "pitch": -3.0}


def inputs(stop_gap=math.inf, cars=(), walkers=(), route=0):
    return encode_inputs(EGO, ROAD, route, stop_gap, cars, walkers)


def free_address():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()


def test_a_message_arrives_with_the_real_values_and_goes_stale():
    address = free_address()
    receiver = TelemetryReceiver(address)
    try:
        assert receiver.poll(now=0.0) is None
        TelemetrySender(address).send(inputs(), (0.2, 3.5), (0.0, 0.0, 1.0), False)
        message = None
        for _ in range(1000):
            message = receiver.poll(now=100.0)
            if message:
                break
        assert len(message["inputs"]) == INPUT_SIZE
        assert message["output"] == [0.2, 3.5]
        assert message["model_driving"] is False
        assert receiver.poll(now=100.0 + STALE_SECONDS + 0.1) is None
    finally:
        receiver.close()


def test_sending_to_nobody_never_stops_the_driver():
    TelemetrySender(free_address()).send(inputs(), (0.0, 0.0), (0.0, 0.0, 0.0), True)


@pytest.mark.parametrize("message, text, in_control", (
    ({"model_driving": True}, "MODEL DRIVING", True),
    ({"model_driving": True, "source": "fused", "fresh": True}, "MODEL DRIVING: OWN LIDAR + ROADSIDE", True),
    ({"model_driving": False, "source": "fused", "fresh": False}, "NO ROADSIDE DATA: BRAKING", False),
    ({"model_driving": False, "source": "truth", "fresh": True}, "SAFETY BRAKE OVERRIDE", False),
    ({"model_driving": False}, "SAFETY BRAKE OVERRIDE", False),
    (None, "NO MODEL DATA", False),
))
def test_control_status_never_claims_the_model_without_data(message, text, in_control):
    assert control_status(message) == (text, in_control)


def test_seen_lines_report_the_inputs():
    car = (110.0, 50.0, 0.0, 0.0, 0.0)
    walker = (105.0, 50.0, 0.0, 0.0)
    lines = dict(seen_lines(decode_inputs(inputs(stop_gap=12.0, cars=[car], walkers=[walker], route=LEFT))))
    assert lines["Speed"] == "18.0 km/h"
    assert lines["Route"] == "LEFT"
    assert lines["Red-light stop line"] == "12.0 m ahead"
    assert lines["Nearby cars"] == "1, nearest 10.0 m"
    assert lines["Nearby pedestrians"] == "1, nearest 5.0 m"


def test_seen_lines_with_nothing_around():
    lines = dict(seen_lines(decode_inputs(inputs())))
    assert lines["Route"] == "STRAIGHT"
    assert lines["Red-light stop line"] == "none in range"
    assert lines["Nearby cars"] == "0"


def test_a_lane_straight_ahead_is_in_the_middle_of_a_centred_camera():
    [(u, v)] = lane_to_pixels([(5.0, 0.0)], THIRD_PERSON, 800, 450, 90.0)
    assert u == pytest.approx(400.0)
    assert v > 225.0  # this close, the road is below the middle of the tilted camera's picture


def test_a_lane_on_the_left_appears_on_the_left():
    [(u, _)] = lane_to_pixels([(20.0, 3.0)], THIRD_PERSON, 800, 450, 90.0)
    assert u < 400.0


def test_a_driver_sitting_left_sees_the_lane_centre_to_the_right():
    [(u, _)] = lane_to_pixels([(20.0, 0.0)], FIRST_PERSON, 800, 450, 90.0)
    assert u > 400.0


def test_further_points_are_nearer_the_horizon():
    near, far = lane_to_pixels([(5.0, 0.0), (30.0, 0.0)], THIRD_PERSON, 800, 450, 90.0)
    assert far[1] < near[1]


def test_points_behind_the_camera_are_left_out():
    assert lane_to_pixels([(-20.0, 0.0)], THIRD_PERSON, 800, 450, 90.0) == []


@pytest.mark.parametrize("model_driving", (True, False))
def test_the_panel_draws_every_state(model_driving):
    import pygame
    from model_panel import PANEL_HEIGHT, ModelPanel

    pygame.init()
    surface = pygame.Surface((1604, PANEL_HEIGHT))
    panel = ModelPanel(1604)
    message = {"inputs": inputs(stop_gap=12.0, cars=[(110.0, 50.0, 0.0, 0.0, 0.0)]).tolist(), "output": [0.1, 2.0],
               "applied": [0.1, 0.0, 0.4], "model_driving": model_driving}
    panel.draw(surface, 0, message)
    panel.draw(surface, 0, None)
    panel.draw(surface, 0, {**message, "output": [float("nan"), 0.0]})
    panel.draw(surface, 0, message, {"ttc": 1.2, "engaged": True}, ["12:00:00 SAFETY GATE: a long reason " * 5] * 4)
    assert np.any(pygame.surfarray.array3d(surface))


def test_safety_lines_only_show_real_values():
    assert safety_lines(None) == [("Min TTC", "n/a"), ("Safety brake", "off")]
    assert safety_lines({"ttc": 1.234, "engaged": True}) == [("Min TTC", "1.23 s"), ("Safety brake", "ON")]
    assert safety_lines({"ttc": float("inf"), "engaged": False})[0] == ("Min TTC", "n/a")


def test_long_events_wrap_between_words():
    assert wrap_text("one two three four", len, 8) == ["one two", "three", "four"]
    assert wrap_text("averyveryverylongword x", len, 5) == ["averyveryverylongword", "x"]
    assert wrap_text("", len, 5) == []


@pytest.mark.parametrize("event, alert", (("SAFETY GATE: x", True), ("COLLISION with a car", True), ("TEST 2: FAIL", True), ("TEST 2: PASS", False)))
def test_only_trouble_is_an_alert(event, alert):
    assert is_alert(event) is alert
