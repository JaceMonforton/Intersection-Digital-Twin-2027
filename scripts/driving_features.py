"""
What the ML car's driving model sees and what it outputs, shared by the recorder, the trainer and the
driver so that they can't drift apart. models/README.md has the contract. No CARLA needed.

Positions and velocities go to the model in the car's own frame (x forward, y LEFT), in metres and
m/s. Scaling them lives inside the model, as it does for the trajectory model. Steering stays in
CARLA's convention, where positive is to the right.

The model chooses the steering and how fast to be going a moment from now. A small fixed controller here
turns that speed into throttle and brake, as cruise control would, and is not part of the model.
"""
import itertools
import math

import numpy as np
from waymo_predictor_node import LEFT, RIGHT, STRAIGHT, along_polyline, world_to_ego

# --- Configuration ---
LANE_POINTS = 10        # points along the route ahead, after the one nearest the car
LANE_SPACING = 2.0      # metres between them
STOP_LINE_RANGE = 35.0  # metres ahead that a stop line counts
MAX_CARS = 6
CAR_RANGE = 40.0        # metres
MAX_WALKERS = 3
WALKER_RANGE = 20.0     # metres
ACTOR_FEATURES = 5      # present, x, y, vx, vy
CROSSING_LOOK = 10.0    # s ahead that a car reaching the route counts; also the time-to-collision when there's none
TURN_GAP_LIMIT = 40.0   # m; distance to a left turn is capped at this

ROUTES = (LEFT, STRAIGHT, RIGHT)
STOP_PRESENT, STOP_GAP = 4, 5  # where the stop line sits in the input
CROSSING_PRESENT, CROSSING_TTC, TURN_GAP = -3, -2, -1  # where the left-turn inputs sit, at the very end
FULL_BRAKE = (0.0, 0.0, 1.0)  # steer, throttle, brake

# speed, route one-hot, stop line (present, distance), lane points, cars, walkers, crossing traffic
INPUT_SCALE = np.concatenate((
    [10.0],
    np.ones(len(ROUTES)),
    [1.0, STOP_LINE_RANGE],
    np.full(2 * (LANE_POINTS + 1), 20.0),
    np.tile([1.0, 20.0, 20.0, 10.0, 10.0], MAX_CARS + MAX_WALKERS),
    [1.0, CROSSING_LOOK, 20.0],
)).astype(np.float32)
INPUT_SIZE = len(INPUT_SCALE)
OUTPUT_SIZE = 2  # steer, target speed
TARGET_HORIZON = 1.0   # s ahead that the target speed is for
MAX_TARGET_SPEED = 15.0  # m/s; above this the model's output is treated as this

# The speed controller, tuned on the ML car in CARLA. Throttle is PID, on a smoothed target
CRUISE_THROTTLE = 0.4  # about what holds cruising speed
CRUISE_SPEED = 5.5     # m/s that CRUISE_THROTTLE holds; a lower target needs proportionally less
KP = 0.15              # throttle per m/s below the target
KI = 0.03              # throttle per m/s per second
KD = 0.03              # throttle per m/s^2 of acceleration, which damps the overshoot
INTEGRAL_LIMIT = 0.15  # most throttle the integral term may add or take
BRAKE_GAIN = 0.2       # per m/s above the target, which is about the deceleration it wants over TARGET_HORIZON
BRAKE_DEADBAND = 0.5   # m/s over the target before it brakes, so that it doesn't dab at the brake
HOLD_TARGET = 1.5      # m/s; when stopped, a target under this means stay stopped rather than creep at a car ahead
HOLD_SPEED = 0.5       # m/s
RISING_TARGET = 0.15   # of the way to a higher target each tick; a lower target counts at once, for safety
THROTTLE_RATE = 0.08   # most the throttle may rise per tick
BRAKE_RATE = 0.2       # most the brake may rise per tick, unless the target says stop now
STEER_SMOOTHING = 0.5  # of the way to the model's steering each tick
SOFT_STOP_SPEED = 2.0  # m/s; under this the brake is limited, because full brake at a crawl stops the car dead
SOFT_STOP_BRAKE = 0.35
STANDSTILL = 0.1       # m/s; once this slow, full brake holds the car
TICK = 0.1             # s


def _actor_slots(ego, actors, count, reach):
    """count slots of (present, x, y, vx, vy), nearest first. Actors are (x, y, vx, vy, ...) in the world."""
    x, y, _, _, yaw = ego
    near = sorted((a for a in actors if math.hypot(a[0] - x, a[1] - y) <= reach),
                  key=lambda a: math.hypot(a[0] - x, a[1] - y))[:count]
    slots = np.zeros((count, ACTOR_FEATURES), dtype=np.float32)
    if near:
        states = np.asarray([a[:4] for a in near], dtype=np.float32)
        slots[:len(near), 0] = 1.0
        slots[:len(near), 1:3] = world_to_ego(states[:, :2], x, y, yaw)
        slots[:len(near), 3:5] = world_to_ego(states[:, 2:4], 0.0, 0.0, yaw)  # vectors: rotate, don't translate
    return slots.ravel()


def encode_inputs(ego, path, route, stop_gap, cars, walkers, crossing_ttc=math.inf, turn_gap=0.0):
    """One INPUT_SIZE float32 vector.

    ego is (x, y, vx, vy, yaw) in the world. path is (x, y) points along the route, starting at the one
    nearest the car, so the first point also gives how far the car is off its lane. route is LEFT,
    STRAIGHT or RIGHT. stop_gap is metres along the route to a stop line whose light isn't green, or inf.
    On a left turn, crossing_ttc is seconds until the first car reaches the route, or inf, and turn_gap is
    metres along the route to where the turn starts, negative once inside it.
    """
    x, y, vx, vy, yaw = ego
    route_one_hot = np.zeros(len(ROUTES), dtype=np.float32)
    route_one_hot[ROUTES.index(route)] = 1.0
    points = np.asarray(path, dtype=np.float32)
    ahead = along_polyline(points, LANE_SPACING, steps=LANE_POINTS, step_seconds=1.0)
    lane = world_to_ego(np.vstack((points[:1], ahead)), x, y, yaw)
    return np.concatenate((
        [math.hypot(vx, vy)],
        route_one_hot,
        [1.0 if stop_gap < STOP_LINE_RANGE else 0.0, min(stop_gap, STOP_LINE_RANGE)],
        lane.ravel(),
        _actor_slots(ego, cars, MAX_CARS, CAR_RANGE),
        _actor_slots(ego, walkers, MAX_WALKERS, WALKER_RANGE),
        [1.0 if crossing_ttc < CROSSING_LOOK else 0.0, min(crossing_ttc, CROSSING_LOOK),
         max(-TURN_GAP_LIMIT, min(TURN_GAP_LIMIT, turn_gap))],
    )).astype(np.float32)


def decode_inputs(inputs):
    """An encode_inputs vector back in plain parts, for showing what the model is given."""
    lane_end = 6 + 2 * (LANE_POINTS + 1)
    cars_end = lane_end + MAX_CARS * ACTOR_FEATURES
    walkers_end = cars_end + MAX_WALKERS * ACTOR_FEATURES

    def present(slots):
        return [tuple(float(v) for v in row[1:]) for row in slots.reshape(-1, ACTOR_FEATURES) if row[0] == 1.0]

    return {
        "speed": float(inputs[0]),
        "route": ROUTES[int(np.argmax(inputs[1:1 + len(ROUTES)]))],
        "stop_gap": float(inputs[STOP_GAP]) if inputs[STOP_PRESENT] == 1.0 else None,
        "lane": [(float(x), float(y)) for x, y in inputs[6:lane_end].reshape(-1, 2)],
        "cars": present(inputs[lane_end:cars_end]),
        "walkers": present(inputs[cars_end:walkers_end]),
        "crossing_ttc": float(inputs[CROSSING_TTC]) if inputs[CROSSING_PRESENT] == 1.0 else None,
        "turn_gap": float(inputs[TURN_GAP]),
    }


def output_from_driving(steer, speed_later):
    """The training target for what a car did: its steering, and its speed TARGET_HORIZON later."""
    return np.array((steer, speed_later), dtype=np.float32)


class SpeedController:
    """Turns the model's steering and target speed into controls. It's a fixed controller, not the model.
    An output that isn't finite is a broken model: brake fully."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.target = None  # starts from the car's own speed, so that a moving car doesn't brake on a reset
        self.integral = 0.0
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.last_speed = None

    def hold(self):
        """The safety brake has the car, so nothing here winds up meanwhile."""
        self.reset()
        return FULL_BRAKE

    def step(self, output, speed):
        steer, target = (float(v) for v in output)
        if not (math.isfinite(steer) and math.isfinite(target) and math.isfinite(speed)):
            return self.hold()
        target = max(0.0, min(MAX_TARGET_SPEED, target))
        current = speed if self.target is None else self.target
        self.target = target if target < current else current + RISING_TARGET * (target - current)
        self.steer += STEER_SMOOTHING * (max(-1.0, min(1.0, steer)) - self.steer)
        acceleration = 0.0 if self.last_speed is None else (speed - self.last_speed) / TICK
        self.last_speed = speed
        error = self.target - speed

        if self.target < HOLD_TARGET and speed < HOLD_SPEED:
            self.integral, self.throttle = 0.0, 0.0
            self.brake = 1.0 if speed < STANDSTILL else min(self.brake + BRAKE_RATE, SOFT_STOP_BRAKE)
        elif error < -BRAKE_DEADBAND:
            self.integral = 0.0
            self.throttle = 0.0
            wanted = min(1.0, -BRAKE_GAIN * error)
            self.brake = min(self.brake + BRAKE_RATE, min(wanted, SOFT_STOP_BRAKE) if speed < SOFT_STOP_SPEED else wanted)
        else:
            self.integral = max(-INTEGRAL_LIMIT, min(INTEGRAL_LIMIT, self.integral + KI * error * TICK))
            wanted = CRUISE_THROTTLE * min(1.0, self.target / CRUISE_SPEED) + KP * error + self.integral - KD * acceleration
            wanted = max(0.0, min(1.0, wanted))
            self.throttle = min(self.throttle + THROTTLE_RATE, wanted)
            self.brake = 0.0
        return self.steer, self.throttle, self.brake


VEHICLE_LENGTH = 4.8      # m, Tesla Model 3
VEHICLE_WIDTH = 2.1       # m


def outline(x, y, yaw_deg):
    """The four corners of a vehicle seen from above."""
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    half_l, half_w = VEHICLE_LENGTH / 2, VEHICLE_WIDTH / 2
    return [(x + c * lx - s * ly, y + s * lx + c * ly)
            for lx, ly in ((half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w))]


def rectangles_overlap(a, b):
    """Separating axis test: two rectangles are apart if some edge's normal separates them."""
    for rect in (a, b):
        for (x1, y1), (x2, y2) in itertools.pairwise(rect):
            nx, ny = y2 - y1, x1 - x2
            pa = [nx * x + ny * y for x, y in a]
            pb = [nx * x + ny * y for x, y in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


def point_to_segment(p, a, b):
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def rectangles_gap(a, b):
    """Metres between two rectangles' nearest edges; 0 if they overlap."""
    if rectangles_overlap(a, b):
        return 0.0
    return min(point_to_segment(p, *edge)
               for corners, other in ((a, b), (b, a))
               for p in corners
               for edge in itertools.pairwise([*other, other[0]]))
