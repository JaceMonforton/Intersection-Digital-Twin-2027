"""The fake RC car: its motion, its faulty link, and the real safety gate behind it. No broker, no CARLA, a fake clock."""
import json
import math
import random

import pytest

pytest.importorskip("paho")

import itertools

from fake_lab_car import (
    MAX_SPEED,
    MAX_STEER_ANGLE,
    WHEELBASE,
    FakeCarMotors,
    FaultyLink,
    pose_message,
)
from physical_ml_subscriber import SafetyGateSubscriber


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def drive(seq, steer=0.0, speed=1.0, session="driver-a"):
    return json.dumps({"session": session, "seq": seq, "steer": steer, "speed": speed}).encode()


RELEASE = json.dumps({"path": "/features/SafetyGate/properties", "value": {"emergency_brake": False}}).encode()


def run(car, seconds, dt=0.05):
    for _ in range(round(seconds / dt)):
        car.step(dt)


# --- Motion ---
def test_it_lags_towards_the_commanded_speed_then_holds_it_in_a_straight_line():
    car = FakeCarMotors()
    car.drive(0.0, 1.0)
    run(car, 0.1)
    assert 0.0 < car.speed < 0.8  # an RC car doesn't jump to speed
    run(car, 3.0)
    assert car.speed == pytest.approx(1.0, abs=0.01)
    assert car.y == pytest.approx(0.0) and car.x > 2.5


def test_it_never_turns_tighter_than_its_steering_allows_or_goes_over_its_cap():
    car = FakeCarMotors()
    car.drive(1.0, 50.0)
    run(car, 2.0)
    assert car.speed <= MAX_SPEED + 1e-9
    radius = WHEELBASE / math.tan(MAX_STEER_ANGLE)
    assert car.y < 0.0  # positive steer is right, as in the driving model, and the lab's y is to the left
    assert math.hypot(car.x, car.y + radius) >= radius - 0.01  # it can't cut inside its tightest circle


def test_stop_brings_it_to_rest_and_it_stays_there():
    car = FakeCarMotors()
    car.drive(0.0, 1.0)
    run(car, 2.0)
    car.stop()
    run(car, 2.0)
    x = car.x
    run(car, 1.0)
    assert car.speed == 0.0 and car.x == x


def test_its_pose_message_says_its_a_simulated_stand_in():
    car = FakeCarMotors()
    message = json.loads(pose_message(car, 12.5))
    value = message["value"]
    assert message["path"] == "/features/pose/properties"
    assert set(value) == {"x", "y", "heading", "speed", "t", "source"}
    assert value["source"] == "simulated stand-in" and value["t"] == 12.5


# --- The faulty link ---
def test_a_clean_link_delivers_everything_in_order():
    link = FaultyLink(FakeClock(), random.Random(0))
    for n in range(5):
        link.send(n)
    assert link.receive() == [0, 1, 2, 3, 4]


def test_an_outage_delivers_nothing_while_it_lasts_and_no_backlog_after():
    clock = FakeClock()
    link = FaultyLink(clock, random.Random(0), outages=[(100.5, 101.5)])
    link.send("before")
    clock.now = 101.0
    link.send("during")
    assert link.receive() == ["before"]
    clock.now = 102.0
    link.send("after")
    assert link.receive() == ["after"]


def test_delay_holds_messages_back_until_they_are_due():
    clock = FakeClock()
    link = FaultyLink(clock, random.Random(0), delay=(0.2, 0.2))
    link.send("late")
    assert link.receive() == []
    clock.now += 0.25
    assert link.receive() == ["late"]


def test_drops_and_replays_happen_at_about_their_rates():
    link = FaultyLink(FakeClock(), random.Random(1), drop=0.2, replay=0.1)
    for n in range(2000):
        link.send(n)
    got = link.receive()
    fresh = set(got)
    assert 0.75 < len(fresh) / 2000 < 0.85
    assert len(got) > len(fresh)  # some arrived twice
    assert any(later < earlier for earlier, later in itertools.pairwise(got))  # a replay is an old one, out of order


# --- The real gate behind the faulty link ---
def drive_through(link, gate, car, clock, seconds, start_seq=1, dt=0.05):
    """The driver sends a command every 0.1 s through the link; the gate and the car run at 20 Hz."""
    seq = start_seq
    for tick in range(round(seconds / dt)):
        clock.now += dt
        if tick % 2 == 0:
            link.send(drive(seq))
            seq += 1
        for payload in link.receive():
            gate.handle_drive(payload)
        gate.check_watchdog()
        if gate.held:
            car.stop()
        car.step(dt)
    return seq


def test_a_silent_outage_stops_the_car_within_the_watchdog_and_it_stays_stopped_after():
    clock = FakeClock()
    car = FakeCarMotors()
    gate = SafetyGateSubscriber(car, clock=clock)
    gate.handle_payload(RELEASE)
    link = FaultyLink(clock, random.Random(0), outages=[(102.0, 103.0)])
    drive_through(link, gate, car, clock, 1.95)
    assert not gate.held and car.speed > 0.5
    drive_through(link, gate, car, clock, 0.4, start_seq=40)  # the last command got through just before 102.0
    assert gate.held and "watchdog" in gate.reason
    drive_through(link, gate, car, clock, 3.0, start_seq=60)  # commands are flowing again
    assert gate.held and car.speed == 0.0  # held until someone releases it


def test_replayed_commands_never_keep_a_silent_car_going():
    clock = FakeClock()
    car = FakeCarMotors()
    gate = SafetyGateSubscriber(car, clock=clock)
    gate.handle_payload(RELEASE)
    link = FaultyLink(clock, random.Random(2), replay=1.0)  # every message is followed by an old one
    seq = drive_through(link, gate, car, clock, 1.0)
    for _ in range(10):  # the driver has gone quiet; only replays of old commands arrive
        clock.now += 0.05
        link.send(drive(seq - 5))
        for payload in link.receive():
            gate.handle_drive(payload)
        gate.check_watchdog()
    assert gate.held
