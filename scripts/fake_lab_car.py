"""
A stand-in for a real RC car in the lab, until one is driving: the vehicle's own safety gate
(physical_ml_subscriber.SafetyGateSubscriber, unchanged) with a simulated motor board behind it, so everything that
reaches it goes through the same checks, watchdog and latch as on the car. It publishes where it is as telemetry,
the way the overhead camera will.

Its motion is a kinematic bicycle model of a 1/10 car: the speed and steering lag behind the command, acceleration
and braking are limited, and it can't turn tighter than its steering allows. Positive steer turns right, as in the
driving model. Its frame is the lab floor's: metres, x along its starting heading, y to its left, heading in degrees
counter-clockwise. Every pose it sends says "simulated stand-in", and nothing it measures is a result for the real car.

A faulty link sits between the broker and the gate, and between the car and its telemetry, so faults can be put in:
dropped and delayed messages, old ones replayed, and silent outages where nothing gets through but nothing
disconnects either.

    python scripts/fake_lab_car.py                                  # a clean link
    python scripts/fake_lab_car.py --outage 10,12 --replay 0.05      # 2 s of silence 10 s in, and some replays
    python scripts/fake_lab_car.py --drop 0.1 --delay 0.02,0.15

Needs only paho-mqtt and the broker. Drive it by publishing commands to device/vehicle_ml/drive, and release its
gate with python scripts/setup_entities.py --clear-brake, as for the real car.
"""
import argparse
import json
import math
import random
import signal
import threading
import time

from physical_ml_subscriber import (
    MAX_DRIVE_SPEED,
    MQTT_BROKER,
    MQTT_PORT,
    WATCHDOG_TIMEOUT,
    SafetyGateSubscriber,
)

# --- An RC-like car. shortcut: these are guesses for a 1/10 car; measure them on the Rosmaster and put them here ---
WHEELBASE = 0.25                     # m
MAX_STEER_ANGLE = math.radians(30.0)  # at steer = 1
MAX_SPEED = MAX_DRIVE_SPEED          # m/s; the gate clamps commands to this already
SPEED_LAG = 0.3                      # s; how quickly the speed follows the command
STEER_LAG = 0.1                      # s; and the steering
MAX_ACCEL = 2.0                      # m/s^2
MAX_BRAKE = 4.0                      # m/s^2
AT_REST = 0.01                       # m/s; slower than this while stopping is stopped
RATE = 20                            # Hz, of its motion and its telemetry
TELEMETRY_TOPIC = "telemetry/vehicle_ml"
THING = "org.example/vehicle_ml"
SOURCE = "simulated stand-in"
REPLAY_MEMORY = 50                   # messages a replay can be picked from


class FakeCarMotors:
    """The motor board's drive and stop, and the car they move. The gate's thread calls drive and stop; the car's
    own loop calls step."""
    name = "fake car"

    def __init__(self, x=0.0, y=0.0, heading=0.0):
        self.lock = threading.Lock()
        self.x, self.y, self.heading = x, y, math.radians(heading)
        self.speed = self.steer = 0.0
        self.target_speed = self.target_steer = 0.0

    def drive(self, steer, speed):
        with self.lock:
            self.target_steer = max(-1.0, min(1.0, steer))
            self.target_speed = max(0.0, min(MAX_SPEED, speed))

    def stop(self):
        with self.lock:
            self.target_speed = 0.0

    def step(self, dt):
        with self.lock:
            wanted = self.speed + (self.target_speed - self.speed) * min(1.0, dt / SPEED_LAG)
            self.speed = min(wanted, self.speed + MAX_ACCEL * dt) if wanted >= self.speed else max(wanted, self.speed - MAX_BRAKE * dt)
            if self.target_speed == 0.0 and self.speed < AT_REST:
                self.speed = 0.0
            self.steer += (self.target_steer - self.steer) * min(1.0, dt / STEER_LAG)
            self.heading -= self.speed / WHEELBASE * math.tan(self.steer * MAX_STEER_ANGLE) * dt  # right is clockwise
            self.x += self.speed * math.cos(self.heading) * dt
            self.y += self.speed * math.sin(self.heading) * dt

    def pose(self):
        with self.lock:
            return self.x, self.y, math.degrees(self.heading), self.speed


def pose_message(car, t):
    """Where the car is, as the Ditto merge the overhead camera will publish."""
    x, y, heading, speed = car.pose()
    return json.dumps({"topic": f"{THING}/things/twin/commands/merge", "headers": {"content-type": "application/merge-patch+json"},
                       "path": "/features/pose/properties",
                       "value": {"x": round(x, 4), "y": round(y, 4), "heading": round(heading, 2), "speed": round(speed, 3),
                                 "t": t, "source": SOURCE}})


class FaultyLink:
    """Messages in, the same messages out, with faults on the way: drop (a share lost), delay (a uniform range of
    seconds), replay (a share followed by an older message sent again) and outages ((start, end) clock times when
    nothing gets through, without any disconnect)."""

    def __init__(self, clock, rng, drop=0.0, delay=(0.0, 0.0), replay=0.0, outages=()):
        self.clock, self.rng = clock, rng
        self.drop, self.delay, self.replay, self.outages = drop, delay, replay, list(outages)
        self.lock = threading.Lock()
        self.queue = []   # (due, order, item)
        self.sent = []    # recent items, for replays
        self.order = 0

    def _silent(self, t):
        return any(start <= t < end for start, end in self.outages)

    def _enqueue(self, item, now):
        due = now + self.rng.uniform(*self.delay)
        if not self._silent(due):
            self.queue.append((due, self.order, item))
            self.order += 1

    def send(self, item):
        with self.lock:
            now = self.clock()
            if self._silent(now) or self.rng.random() < self.drop:
                return
            if self.sent and self.rng.random() < self.replay:
                self._enqueue(self.rng.choice(self.sent), now)
            self._enqueue(item, now)
            self.sent = [*self.sent[-(REPLAY_MEMORY - 1):], item]

    def receive(self):
        with self.lock:
            now = self.clock()
            due = sorted(entry for entry in self.queue if entry[0] <= now)
            self.queue = [entry for entry in self.queue if entry[0] > now]
        return [item for _, _, item in due]


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def pair(text):
    a, b = (float(v) for v in text.split(","))
    return a, b


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--broker", default=MQTT_BROKER)
    parser.add_argument("--port", type=int, default=MQTT_PORT)
    parser.add_argument("--watchdog", type=float, default=WATCHDOG_TIMEOUT)
    parser.add_argument("--start", type=lambda t: tuple(float(v) for v in t.split(",")), default=(0.0, 0.0, 0.0),
                        metavar="X,Y,HEADING", help="lab metres and degrees")
    parser.add_argument("--drop", type=float, default=0.0, help="share of messages lost, each way")
    parser.add_argument("--delay", type=pair, default=(0.0, 0.0), metavar="MIN,MAX", help="seconds each message is held back")
    parser.add_argument("--replay", type=float, default=0.0, help="share of messages followed by an old one sent again")
    parser.add_argument("--outage", type=pair, action="append", default=[], metavar="FROM,TO",
                        help="seconds after the start when nothing gets through either way; may be repeated")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not (0.0 <= args.drop < 1.0 and 0.0 <= args.replay <= 1.0 and 0.0 <= args.delay[0] <= args.delay[1]):
        parser.error("--drop must be in [0, 1), --replay in [0, 1] and --delay MIN,MAX with 0 <= MIN <= MAX")

    for name in ("SIGBREAK", "SIGTERM"):  # how full_loop_sim.py and check_physical_loop.py stop it on Windows
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _raise_keyboard_interrupt)
    started = time.monotonic()
    outages = [(started + a, started + b) for a, b in args.outage]
    rng = random.Random(args.seed)
    incoming = FaultyLink(time.monotonic, rng, args.drop, args.delay, args.replay, outages)
    outgoing = FaultyLink(time.monotonic, rng, args.drop, args.delay, args.replay, outages)
    car = FakeCarMotors(*args.start)
    gate = SafetyGateSubscriber(car, args.broker, args.port, watchdog_timeout=args.watchdog)
    gate.client.on_message = lambda client, userdata, msg: incoming.send(msg)  # everything in passes the faulty link
    print(f"[FAKE CAR] A {SOURCE}, not a real car. Faults: drop {args.drop}, delay {args.delay} s, replay {args.replay}, "
          f"outages {args.outage or 'none'}.", flush=True)
    gate.start()
    try:
        tick = 1.0 / RATE
        next_tick = time.monotonic()
        while True:
            for msg in incoming.receive():
                gate._on_message(None, None, msg)
            car.step(tick)
            outgoing.send(pose_message(car, round(time.monotonic() - started, 3)))
            for payload in outgoing.receive():
                gate.client.publish(TELEMETRY_TOPIC, payload)
            next_tick += tick
            time.sleep(max(0.0, next_tick - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        gate.stop()
        car.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
