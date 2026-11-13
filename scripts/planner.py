"""
The planner: every tick, it decides how fast the ML car should go from what the car perceives, the way a Waymo-style
car plans rather than only reacting. The driving model still steers and proposes a speed; the planner can make the car
slower or more careful than that, and inside the junction it can choose to clear it rather than stop in a crossing
car's path. It's rules, not AI, and the window says so whenever it changed the model's speed.

Each tick:
  1. Every perceived car and person gets a few possible futures (Hypothesis), each with a weight: straight on; near
     the junction a car may also turn across, a person standing by the road may step out, and a car ahead may brake hard.
  2. A handful of options for the ML car (OPTIONS) are rolled out along its route for HORIZON s.
  3. Each option is scored: collisions and near misses in each future, weighted; progress; comfort. The cheapest wins.
  4. Without a roadside unit to see round the corner, the car approaches the junction no faster than it could stop
     for what might be hidden there.

numpy only, so that it's testable without CARLA. Everything is in CARLA's world frame (y right, a left turn makes the
yaw smaller), as the views give it.
"""
import math
from dataclasses import dataclass

import numpy as np

HORIZON, STEP = 4.0, 0.2       # s rolled out, and between checks
TARGET_HORIZON = 1.0           # s ahead that the speed handed to the controller is for, as the model's is
ACCEL, COMFORT_BRAKE, HARD_BRAKE = 1.5, 3.0, 6.0  # m/s^2 the ML car can manage; measured by training/measure_braking.py
# The brakes take about 0.2 s to bite. Rolling that out made it clear the junction into the red-light runner more
# often (4 collisions in 30 tune seeds against 1), so the roll-outs leave it out
CLEAR_SPEED = 5.0              # m/s that clearing the junction aims for
MAX_SPEED = 8.0                # m/s
JUNCTION_REACH = 35.0          # m from the junction centre where cars may turn and people may cross
INSIDE_JUNCTION = 18.0         # m from the centre that the ML car counts as in it, where clearing is an option
TURN_RADIUS = 11.0             # m, a car turning at this junction
STEP_OUT_SPEED = 1.4           # m/s a person standing by the road might step out at
STEP_OUT_REACH = 4.0           # m from the route that a standing person might step out from
# Weights of the futures: what's most likely counts most, but a turn across or a step out still has to be planned for
WEIGHTS = {"straight": 1.0, "turn_left": 0.5, "turn_right": 0.2, "step_out": 0.3, "standing": 1.0, "braking": 0.3}
LEAD_ANGLE = 45.0              # degrees: a car ahead heading within this of the ML car's way may brake hard in front of it
# The ML car and others as two circles along their length, so a car's corner can hit and not just its centre
CAR_CIRCLES, CAR_RADIUS = (-1.3, 1.3), 1.15
WALKER_RADIUS = 0.4
COLLISION_COST = 100.0         # per collision, times the future's weight, and more the sooner it is
NEAR_MISS, NEAR_MISS_COST = 2.0, 8.0  # m of clearance under which a pass costs, per m short: over the suite's 1.5 m safe gap, as
# waiting to turn at under it failed most left turns that failed
PROGRESS_COST = 6.0            # for standing still the whole horizon, less for getting further
COMFORT_COST = 0.15            # per (m/s^2)^2 of braking
BLIND_APPROACH = 4.0           # m/s; with no roadside unit, no faster than this into the junction
MOVING = 0.5                   # m/s


@dataclass(frozen=True)
class Hypothesis:
    """One possible future for one actor: where it is at each step, and how much it counts."""
    kind: str
    weight: float
    path: np.ndarray       # [steps, 2]
    yaws: np.ndarray       # [steps], radians, the way it faces
    radius: float
    circles: tuple


@dataclass(frozen=True)
class Plan:
    speed: float           # the target speed handed to the controller
    option: str            # which option won
    changed: bool          # whether it's different from the model's
    reason: str            # for the window and the logs


def times():
    return np.arange(STEP, HORIZON + STEP / 2, STEP)


def predicted_path(x, y, vx, vy, turning, at):
    """(x, y) at each time, keeping speed and turning at a steady rate (rad/s in CARLA's frame: negative turns left)."""
    speed, heading = math.hypot(vx, vy), math.atan2(vy, vx)
    if abs(turning) < 1e-3 or speed < 1e-3:
        return np.c_[x + vx * at, y + vy * at], np.full(len(at), heading)
    angles = heading + turning * at
    return (np.c_[x + speed / turning * (np.sin(angles) - math.sin(heading)), y - speed / turning * (np.cos(angles) - math.cos(heading))],
            angles)


def is_lead(x, y, vx, vy, route):
    """Is a car ahead of the start of the route and heading its way?"""
    way = route[1] - route[0]
    off = math.degrees(math.atan2(vy, vx) - math.atan2(way[1], way[0]))
    return abs((off + 180) % 360 - 180) < LEAD_ANGLE and (x - route[0][0]) * way[0] + (y - route[0][1]) * way[1] > 0


def braking_path(x, y, vx, vy, at):
    """Straight on, braking at HARD_BRAKE to a stop."""
    speed = math.hypot(vx, vy)
    gone = np.minimum(speed * at - HARD_BRAKE * at ** 2 / 2, speed ** 2 / (2 * HARD_BRAKE))
    return np.c_[x + vx / speed * gone, y + vy / speed * gone], np.full(len(at), math.atan2(vy, vx))


def hypotheses(cars, walkers, junction, route):
    """The futures of every actor. cars are (x, y, vx, vy, yaw in degrees), walkers (x, y, vx, vy)."""
    at, out = times(), []
    for x, y, vx, vy, *rest in cars:
        speed = math.hypot(vx, vy)
        near = math.hypot(x - junction[0], y - junction[1]) < JUNCTION_REACH
        if speed < MOVING:
            yaw = math.radians(rest[0]) if rest else 0.0
            out.append(Hypothesis("standing", WEIGHTS["standing"], np.tile((x, y), (len(at), 1)), np.full(len(at), yaw), CAR_RADIUS, CAR_CIRCLES))
            continue
        futures = [("straight", 0.0)]
        if near:
            futures += [("turn_left", -speed / TURN_RADIUS), ("turn_right", speed / TURN_RADIUS)]
        for kind, turning in futures:
            path, yaws = predicted_path(x, y, vx, vy, turning, at)
            out.append(Hypothesis(kind, WEIGHTS[kind], path, yaws, CAR_RADIUS, CAR_CIRCLES))
        if is_lead(x, y, vx, vy, route):
            out.append(Hypothesis("braking", WEIGHTS["braking"], *braking_path(x, y, vx, vy, at), CAR_RADIUS, CAR_CIRCLES))
    for x, y, vx, vy in walkers:
        speed = math.hypot(vx, vy)
        if speed >= MOVING:
            path, yaws = predicted_path(x, y, vx, vy, 0.0, at)
            out.append(Hypothesis("straight", WEIGHTS["straight"], path, yaws, WALKER_RADIUS, (0.0,)))
            continue
        out.append(Hypothesis("standing", WEIGHTS["standing"], np.tile((x, y), (len(at), 1)), np.zeros(len(at)), WALKER_RADIUS, (0.0,)))
        gaps = np.hypot(route[:, 0] - x, route[:, 1] - y)
        if math.hypot(x - junction[0], y - junction[1]) < JUNCTION_REACH and gaps.min() < STEP_OUT_REACH:
            nearest = route[int(np.argmin(gaps))]
            direction = np.array(nearest) - (x, y)
            direction = direction / max(np.hypot(*direction), 1e-6) * STEP_OUT_SPEED
            path, yaws = predicted_path(x, y, direction[0], direction[1], 0.0, at)
            out.append(Hypothesis("step_out", WEIGHTS["step_out"], path, yaws, WALKER_RADIUS, (0.0,)))
    return out


def along_route(route, distances):
    """(x, y) and heading at each distance along the route, carrying straight on past its end."""
    points = np.asarray(route, float)
    steps = np.diff(points, axis=0)
    lengths = np.hypot(steps[:, 0], steps[:, 1])
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    end = cumulative[-1]
    last = steps[-1] / max(lengths[-1], 1e-6)
    inside = np.minimum(distances, end)
    xy = np.c_[np.interp(inside, cumulative, points[:, 0]), np.interp(inside, cumulative, points[:, 1])]
    xy += np.outer(np.maximum(distances - end, 0.0), last)
    segment = np.clip(np.searchsorted(cumulative, inside, side="right") - 1, 0, len(steps) - 1)
    headings = np.arctan2(steps[segment, 1], steps[segment, 0])
    return xy, headings


def roll_out(speed, target, decel=0.0, accel=ACCEL):
    """Speeds at each step and distances covered, going for target (decel > 0 brakes at that rate down to it)."""
    speeds, distances, v, s = [], [], speed, 0.0
    for _ in times():
        if target < v:
            v = max(target, v - (decel or COMFORT_BRAKE) * STEP)
        else:
            v = min(target, v + accel * STEP)
        s += v * STEP
        speeds.append(v)
        distances.append(s)
    return np.array(speeds), np.array(distances)


def options(speed, model_target, inside):
    """(name, speeds, distances, braking) for each thing the ML car could do now."""
    model_target = min(max(model_target, 0.0), MAX_SPEED)
    out = [("model", *roll_out(speed, model_target), 0.0),
           ("ease", *roll_out(speed, min(model_target, 0.6 * speed)), COMFORT_BRAKE * 0.5),
           ("stop", *roll_out(speed, 0.0, COMFORT_BRAKE), COMFORT_BRAKE),
           ("hard_stop", *roll_out(speed, 0.0, HARD_BRAKE), HARD_BRAKE)]
    if inside:  # in the junction, stopping can leave the car in someone's path, so getting out of it is an option
        out.append(("clear", *roll_out(speed, max(model_target, CLEAR_SPEED)), 0.0))
    return out


def circles(xy, headings, offsets):
    """[steps, len(offsets), 2]: circle centres along a body's length."""
    return np.stack([xy + np.c_[np.cos(headings), np.sin(headings)] * o for o in offsets], axis=1)


def score(ego_xy, ego_headings, futures):
    """(collision cost, near-miss cost, first collision time or None) of one roll-out against every future."""
    mine = circles(ego_xy, ego_headings, CAR_CIRCLES)
    cost = near = 0.0
    first = None
    for h in futures:
        theirs = circles(h.path, h.yaws, h.circles)
        gaps = np.linalg.norm(mine[:, :, None, :] - theirs[:, None, :, :], axis=-1).min(axis=(1, 2)) - (CAR_RADIUS + h.radius)
        hit = np.flatnonzero(gaps < 0.0)
        if len(hit):
            when = (hit[0] + 1) * STEP
            cost += h.weight * COLLISION_COST * (1.0 + (HORIZON - when) / HORIZON)
            first = when if first is None else min(first, when)
        else:
            near += h.weight * NEAR_MISS_COST * max(0.0, NEAR_MISS - float(gaps.min()))
    return cost, near, first


class Planner:
    """Chooses the ML car's speed. junction is the junction centre's (x, y); has_roadside says whether a roadside unit
    sees round the corners for the car."""

    def __init__(self, junction, has_roadside=True):
        self.junction = junction
        self.has_roadside = has_roadside

    def plan(self, ego, route, cars, walkers, model_target):
        """ego is (x, y, vx, vy, yaw); route the (x, y) points ahead of the car along its route."""
        x, y, vx, vy = ego[:4]
        speed = math.hypot(vx, vy)
        route = np.asarray(route, float)
        if len(route) < 2:
            return Plan(model_target, "model", False, "no route ahead")
        to_centre = math.hypot(x - self.junction[0], y - self.junction[1])
        inside = to_centre < INSIDE_JUNCTION
        futures = hypotheses(cars, walkers, self.junction, route)
        scored = []
        for name, speeds, distances, braking in options(speed, model_target, inside):
            xy, headings = along_route(route, distances)
            collide, near, first = score(xy, headings, futures)
            progress = PROGRESS_COST * (1.0 - distances[-1] / (CLEAR_SPEED * HORIZON))
            scored.append((collide + near + progress + COMFORT_COST * braking ** 2, name, speeds, first, collide))
        if scored[0][4] == 0.0:  # clearing is an escape: when the model's own speed is collision-free, it never overrides a wait
            scored = [s for s in scored if s[1] != "clear"]
        _, name, speeds, first, _ = min(scored, key=lambda s: s[0])
        target = float(np.interp(TARGET_HORIZON, times(), speeds))
        reason = f"{name}: {'a collision in ' + format(first, '.1f') + ' s otherwise' if first else 'clearest way on'}"
        if not self.has_roadside and to_centre < JUNCTION_REACH and not inside and target > BLIND_APPROACH:
            target, name, reason = BLIND_APPROACH, "blind_approach", "nothing sees round the corner, so slow into the junction"
        if name == "model":
            return Plan(float(model_target), name, False, "the model's speed is clear")
        return Plan(target, name, abs(target - model_target) > 0.3, reason)

