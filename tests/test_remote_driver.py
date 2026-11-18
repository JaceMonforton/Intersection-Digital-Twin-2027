"""The remote driver's commands: when it drives, when it holds, and the units it sends. No CARLA, no broker."""
import json
import math

import numpy as np
import pytest
from lab_frame import LabFrame
from remote_driver import POSE_MAX_AGE, CommandStream, PoseTracker, next_command

FRAME = LabFrame(scale=10.0, rotation_deg=0.0, flip_y=True, offset_x=-47.8, offset_y=20.4)


class FakeClock:
    def __init__(self):
        self.now = 50.0

    def __call__(self):
        return self.now


def pose_payload(t, x=0.0, y=0.0, heading=0.0, speed=1.0):
    return json.dumps({"path": "/features/pose/properties",
                       "value": {"x": x, "y": y, "heading": heading, "speed": speed, "t": t, "source": "simulated stand-in"}}).encode()


def decision(steer=0.1, speed=8.0, allowed=True):
    """What ModelDriver.decide returns: inputs, the model's output, the drive (steer, CARLA m/s) and whether it may drive."""
    return np.zeros(76, dtype=np.float32), np.array((steer, speed)), np.array((steer, speed), dtype=np.float32), allowed


def tracked(clock, t=1.0):
    tracker = PoseTracker(FRAME, clock)
    tracker.feed(pose_payload(t))
    return tracker


# --- The pose ---
def test_a_fresh_pose_comes_out_in_carlas_frame_as_the_ego_state():
    ego = tracked(FakeClock()).ego()
    assert ego == pytest.approx((-47.8, 20.4, 10.0, 0.0, 0.0))  # x, y, vx, vy, yaw: 1 lab m/s is 10 in CARLA


def test_no_pose_yet_is_no_ego():
    assert PoseTracker(FRAME, FakeClock()).ego() is None


def test_a_pose_that_stops_arriving_goes_stale():
    clock = FakeClock()
    tracker = tracked(clock)
    clock.now += POSE_MAX_AGE
    assert tracker.ego() is not None
    clock.now += 0.01
    assert tracker.ego() is None


def test_an_old_pose_sent_again_doesnt_count_as_fresh():
    clock = FakeClock()
    tracker = tracked(clock, t=1.0)
    for _ in range(5):  # the car keeps sending the same pose, as a stuck camera or publisher would
        clock.now += 0.1
        tracker.feed(pose_payload(1.0))
    assert tracker.ego() is None
    tracker.feed(pose_payload(1.6))
    assert tracker.ego() is not None


@pytest.mark.parametrize("payload", [b"not json", b"[]", json.dumps({"path": "/features/pose/properties", "value": {"x": 1}}).encode(),
                                     pose_payload(float("nan")), json.dumps({"path": "/features/location/properties", "value": {}}).encode()])
def test_anything_that_isnt_a_usable_pose_is_ignored(payload):
    tracker = tracked(FakeClock())
    tracker.feed(payload)
    assert tracker.ego() == pytest.approx((-47.8, 20.4, 10.0, 0.0, 0.0))


# --- The commands ---
def test_each_stream_has_its_own_session_and_numbers_its_commands():
    a, b = CommandStream(), CommandStream()
    first, second = json.loads(a.hold()), json.loads(a.drive(0.1, 0.5))
    assert first["session"] == second["session"] != json.loads(b.hold())["session"]
    assert (first["seq"], second["seq"]) == (1, 2)
    assert first["hold"] is True and second == {"session": first["session"], "seq": 2, "steer": 0.1, "speed": 0.5}


def test_it_drives_at_the_lab_speed_and_keeps_the_steering_sign():
    command = json.loads(next_command(CommandStream(), tracked(FakeClock()), lambda ego: decision(steer=0.25, speed=8.0), FRAME))
    assert command["steer"] == 0.25  # positive is right in CARLA, and the y-flip leaves right as right
    assert command["speed"] == pytest.approx(0.8)  # 8 CARLA m/s is 0.8 lab m/s at x10


@pytest.mark.parametrize("why, tracker_fn, decide", [
    ("no pose yet", lambda clock: PoseTracker(FRAME, clock), lambda ego: decision()),
    ("the driver holds it", tracked, lambda ego: decision(allowed=False)),
    ("a non-finite steer", tracked, lambda ego: decision(steer=float("nan"))),
    ("a non-finite speed", tracked, lambda ego: decision(speed=float("inf"))),
])
def test_it_holds_on_any_doubt(why, tracker_fn, decide):
    command = json.loads(next_command(CommandStream(), tracker_fn(FakeClock()), decide, FRAME))
    assert command.get("hold") is True, why


def test_a_stale_pose_holds_without_even_asking_the_driver():
    clock = FakeClock()
    tracker = tracked(clock)
    clock.now += 1.0
    asked = []
    command = json.loads(next_command(CommandStream(), tracker, lambda ego: asked.append(ego) or decision(), FRAME))
    assert command.get("hold") is True and asked == []


def test_a_driver_that_raises_holds_rather_than_crashing():
    def broken(ego):
        raise ValueError("no route here")

    assert json.loads(next_command(CommandStream(), tracked(FakeClock()), broken, FRAME)).get("hold") is True


def test_it_never_asks_for_a_negative_speed():
    command = json.loads(next_command(CommandStream(), tracked(FakeClock()), lambda ego: decision(speed=-3.0), FRAME))
    assert command["speed"] == 0.0 and math.copysign(1.0, command["speed"]) == 1.0
