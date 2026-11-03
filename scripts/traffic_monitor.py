"""
Puts vehicle_ml into live traffic and reports how the safety system copes.

Its prediction comes from the trained model, in the processor. A simple lane driver here only moves
it along the road, because the model predicts and can't steer, and because the Traffic Manager keeps
overriding the brake once a car has been handed back to it. The car stops when the processor engages
its SafetyGate. The brake latches, so after a short hold this script releases it, standing in for an
operator, and the car drives on. If the hazard is still there, the processor brakes again.

Normally started by run_simulation.py. Needs the processor and test_traffic.py running.
    python scripts/traffic_monitor.py --duration 60
"""
import argparse
import json
import math
import random
import re
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import requests
from central_gpu_processor import ML_PAINT
from left_turn_scenario import outline, rectangles_gap
from waymo_predictor_node import LEFT, RIGHT, STRAIGHT, turn_class, yaw_difference

# --- Configuration ---
CARLA_HOST = "localhost"
CARLA_PORT = 2000
GATE_URL = "http://localhost:8080/api/2/things/org.example:vehicle_ml/features/SafetyGate/properties"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

BRAKE_HOLD = 3.0          # s the car stays stopped before the brake is released
VIEW_RADIUS = 40.0        # m around the intersection counted as "in view"
JUNCTION_RADIUS = 18.0    # m around the intersection used to classify a pass through it
TURN_DEGREES = 45.0       # yaw change through the junction that counts as a turn
CLOSE_CALL_GAP = 1.0      # m between outlines
CLOSE_CALL_SPEED = 2.0    # m/s closing speed needed for a close call
# Autopilot routinely pulls up behind a stopped car at about 3 m/s, so in-line needs more to count
CLOSE_CALL_SPEED_IN_LINE = 5.0
IN_LINE_HEADING = 20.0    # degrees between two cars that counts as in line
IN_LINE_OFFSET = 1.5      # m to one side that counts as in line
CLOSE_CALL_COOLDOWN = 5.0 # s before the same pair can count again
PAIR_RANGE = 8.0          # m between centres; further apart, outlines can't be close
REPORT_INTERVAL = 15.0    # s between progress lines

# The ML car's lane driver
CRUISE_SPEED = 7.0        # m/s, about 25 km/h
DRIVE_ACCEL = 2.5         # m/s^2
DRIVE_DECEL = 3.0         # m/s^2, ordinary slowing for traffic
BRAKE_DECEL = 6.0         # m/s^2, the emergency brake
WAYPOINT_STEP = 1.0       # m
LOOK_AHEAD = 45.0         # m of lane planned ahead
BRANCH_LOOK = 15          # waypoints walked down a branch to see where it heads
SIGNAL_DISTANCE = 30.0    # m before a turn that the signal comes on
FOLLOW_GAP = 7.0          # m between centres to keep from the car ahead when stopped
FOLLOW_OFFSET = 1.6       # m to the side that still counts as in the lane ahead
Z_CLEARANCE = 0.3         # m above the road

# Red lights
STOP_LINE_WIDTH = 1.75    # m either side of a stop line's lane centre
STOP_SHORT = 3.0          # m before the stop line that the ML car aims to stop
RED_LOOK = 35.0           # m ahead that the ML car looks for a red light
RED_RUN_SPEED = 1.5       # m/s; crossing a stop line slower than this is creeping, not running the light
RED_RUN_DEPTH = 2.5       # m around a stop line where a crossing is detected
RED_RUN_HEADING = 45.0    # degrees from the lane's direction; otherwise it's crossing traffic
STOP_LINE_CLEARANCE = 25.0  # m back from a stop line that a car may be placed
RED_RUN_CONFIRM = 6.0     # m past the stop line before it counts: autopilot often stops hard right at the line
RED_RUN_WINDOW = 5.0      # s after reaching the line to get that far; otherwise it stopped
RED_RUN_COOLDOWN = 10.0   # s before the same car at the same light can count again

# Autopilot routes wander off across the map and rarely come back, which empties the intersection.
# So cars that have gone far enough to be out of sight are moved back to a road near it.
RECYCLE_DISTANCE = 90.0   # m from the intersection before a car can be moved back
RECYCLE_NEAR = (35.0, 80.0)  # m from the intersection that they reappear, just out of the overhead view
RECYCLE_CLEARANCE = 20.0  # m clear around a reappearing car, or a moving one can't stop for it in time
RECYCLE_INTERVAL = 1.0    # s between moves, so traffic returns as a flow and not all at once


# ---------------------------------------------------------------------------
# Pure logic (no CARLA needed)
# ---------------------------------------------------------------------------
def classify_pass(entry_yaw, exit_yaw):
    """straight, left or right, from a vehicle's yaw entering and leaving the junction."""
    change = (exit_yaw - entry_yaw + 180.0) % 360.0 - 180.0  # negative is left in CARLA's frame
    if change < -TURN_DEGREES:
        return "left"
    return "right" if change > TURN_DEGREES else "straight"


class JunctionCounter:
    """Counts vehicles passing through the junction, by what they did there."""

    def __init__(self, center):
        self.center = center
        self.inside = {}  # actor id -> yaw on entry
        self.counts = {"straight": 0, "left": 0, "right": 0}

    def update(self, actor_id, x, y, yaw):
        near = math.hypot(x - self.center[0], y - self.center[1]) <= JUNCTION_RADIUS
        if near and actor_id not in self.inside:
            self.inside[actor_id] = yaw
        elif not near and actor_id in self.inside:
            self.counts[classify_pass(self.inside.pop(actor_id), yaw)] += 1


def relative_position(me, other):
    """(metres ahead, metres to the left) of other, as seen from me. Both are (x, y, vx, vy, yaw)."""
    yaw = math.radians(me[4])
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = other[0] - me[0], other[1] - me[1]
    return c * dx + s * dy, s * dx - c * dy  # CARLA's y points right, so left is negated


def is_close_call_speed(a, b):
    """Is this pair closing fast enough to count, given how they're lined up?"""
    closing = closing_speed(a, b)
    _, left = relative_position(a, b)
    heading = abs((b[4] - a[4] + 180.0) % 360.0 - 180.0)
    in_line = heading < IN_LINE_HEADING and abs(left) < IN_LINE_OFFSET
    return closing >= (CLOSE_CALL_SPEED_IN_LINE if in_line else CLOSE_CALL_SPEED)


def closing_speed(a, b):
    """How fast two (x, y, vx, vy) vehicles are approaching each other; negative if separating."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        return 0.0
    return -(dx * (b[2] - a[2]) + dy * (b[3] - a[3])) / distance


def speed_for_gap(gap):
    """The fastest this car may go and still stop within gap metres at ordinary braking."""
    return math.sqrt(2.0 * DRIVE_DECEL * max(gap, 0.0)) if math.isfinite(gap) else math.inf


def next_speed(speed, target, dt, braked):
    """One step of the driver's speed."""
    if braked:
        return max(0.0, speed - BRAKE_DECEL * dt)
    if speed > target:
        return max(target, speed - DRIVE_DECEL * dt)
    return min(target, speed + DRIVE_ACCEL * dt)


def lead_gap(me, others):
    """Metres to the nearest car ahead in my lane, or inf."""
    nearest = math.inf
    for other in others:
        ahead, left = relative_position(me, other)
        if ahead > 0.0 and abs(left) <= FOLLOW_OFFSET:
            nearest = min(nearest, ahead)
    return nearest


def red_light_gap(path_points, stop_lines):
    """Metres along my planned path to the first stop line of a light that isn't green, or inf.
    path_points are (x, y) a metre apart, starting here; stop_lines are (x, y) of the red ones."""
    for distance, (x, y) in enumerate(path_points):
        for sx, sy in stop_lines:
            if math.hypot(x - sx, y - sy) <= STOP_LINE_WIDTH:
                return float(distance)
    return math.inf


def is_running_red(vehicle, stop_x, stop_y, stop_yaw):
    """Is this (x, y, vx, vy, yaw) vehicle driving across this stop line, in the lane's direction?"""
    x, y, vx, vy, yaw = vehicle
    if math.hypot(x - stop_x, y - stop_y) > RED_RUN_DEPTH or math.hypot(vx, vy) < RED_RUN_SPEED:
        return False
    return abs(yaw_difference(stop_yaw, yaw)) <= RED_RUN_HEADING


def distance_past_line(x, y, stop_x, stop_y, stop_yaw):
    """Metres beyond a stop line, measured along its lane's direction."""
    yaw = math.radians(stop_yaw)
    return (x - stop_x) * math.cos(yaw) + (y - stop_y) * math.sin(yaw)


def clear_of_stop_lines(points, stop_lines):
    """Spawn points far enough back from every stop line. A car only learns about a light by entering
    its trigger zone, so one placed inside the zone ignores that light."""
    return [p for p in points if all(math.hypot(p.location.x - x, p.location.y - y) >= STOP_LINE_CLEARANCE
                                     for x, y in stop_lines)]


class RedLightWatch:
    """Records every vehicle that drives across a stop line while its light is red."""

    def __init__(self, carla, world):
        self.red = carla.TrafficLightState.Red
        self.green = carla.TrafficLightState.Green
        self.lines = []  # (light, x, y, yaw) for the stop line of every lane controlled by a light
        for light in world.get_actors().filter("traffic.traffic_light"):
            for stop in light.get_stop_waypoints():
                transform = stop.transform
                self.lines.append((light, transform.location.x, transform.location.y, transform.rotation.yaw))
        self.seen = {}
        self.suspects = {}  # (vehicle, stop line index) -> time it reached the line on red
        self.runs = []

    def stop_line_positions(self):
        return [(x, y) for _, x, y, _ in self.lines]

    def not_green(self):
        return [(x, y) for light, x, y, _ in self.lines if light.get_state() != self.green]

    def update(self, states, now, ml_id=None, label=""):
        """Reaching a red stop line at speed only makes a car a suspect. It has run the light if it then
        carries on past the line, whatever colour the light has turned by then."""
        for index, (light, x, y, yaw) in enumerate(self.lines):
            if light.get_state() != self.red:
                continue  # amber: legal to clear the junction
            for vehicle_id, state in states.items():
                key = (vehicle_id, index)
                if key not in self.suspects and is_running_red(state, x, y, yaw) and distance_past_line(
                        state[0], state[1], x, y, yaw) <= RED_RUN_CONFIRM / 2:
                    self.suspects[key] = now

        for key, reached in list(self.suspects.items()):
            vehicle_id, index = key
            state = states.get(vehicle_id)
            _, x, y, yaw = self.lines[index]
            if state is None or now - reached > RED_RUN_WINDOW:
                del self.suspects[key]
                continue
            past = distance_past_line(state[0], state[1], x, y, yaw)
            if RED_RUN_CONFIRM <= past <= 3 * RED_RUN_CONFIRM:
                del self.suspects[key]
                if now - self.seen.get(key, -math.inf) < RED_RUN_COOLDOWN:
                    continue
                self.seen[key] = now
                speed = math.hypot(state[2], state[3])
                is_ml = vehicle_id == ml_id
                self.runs.append({"t": round(now, 1), "ml": is_ml, "vehicle": vehicle_id, "speed": round(speed, 1)})
                who = "vehicle_ml" if is_ml else f"carla:{vehicle_id}"
                print(f"[{label}] t={now:.1f} s: {who} RAN A RED LIGHT at {speed:.1f} m/s", flush=True)


class LaneDriver:
    """Moves a car along lanes from a starting waypoint, turning at random where the road branches.
    It predicts nothing: that's the trained model's job, in the processor."""

    def __init__(self, waypoint, rng):
        self.rng = rng
        self.points = [waypoint]
        self.turns = {}  # index into points -> LEFT or RIGHT, for a branch chosen there
        self.s = 0.0
        self.speed = 0.0
        self._extend()

    def _branch_heading(self, waypoint):
        end = waypoint
        for _ in range(BRANCH_LOOK):
            following = end.next(WAYPOINT_STEP)
            if not following:
                break
            end = following[0]
        return end.transform.rotation.yaw

    def _extend(self):
        while (len(self.points) - 1) * WAYPOINT_STEP - self.s < LOOK_AHEAD:
            last = self.points[-1]
            following = last.next(WAYPOINT_STEP)
            if not following:
                return
            chosen = following[0]
            if len(following) > 1:
                chosen = self.rng.choice(following)
                change = yaw_difference(last.transform.rotation.yaw, self._branch_heading(chosen))
                turn = turn_class(change)
                if turn != STRAIGHT:
                    self.turns[len(self.points)] = turn
            self.points.append(chosen)

    def pose(self):
        """(x, y, z, yaw) at the current distance along the lane."""
        index = min(int(self.s / WAYPOINT_STEP), len(self.points) - 1)
        a = self.points[index].transform
        b = self.points[min(index + 1, len(self.points) - 1)].transform
        f = self.s / WAYPOINT_STEP - index
        x = a.location.x + (b.location.x - a.location.x) * f
        y = a.location.y + (b.location.y - a.location.y) * f
        z = a.location.z + (b.location.z - a.location.z) * f
        yaw = a.rotation.yaw + yaw_difference(a.rotation.yaw, b.rotation.yaw) * f
        return x, y, z, yaw

    def path_ahead(self, metres):
        """(x, y) points a metre apart along the planned lane, starting here."""
        start = int(self.s / WAYPOINT_STEP)
        return [(w.transform.location.x, w.transform.location.y) for w in self.points[start:start + int(metres)]]

    def signal(self):
        """LEFT, RIGHT or STRAIGHT: what the turn signal should show right now."""
        here = self.s / WAYPOINT_STEP
        upcoming = [(i, turn) for i, turn in self.turns.items() if here - BRANCH_LOOK <= i <= here + SIGNAL_DISTANCE]
        return min(upcoming)[1] if upcoming else STRAIGHT

    def step(self, dt, target, braked):
        self.speed = next_speed(self.speed, target, dt, braked)
        self.s += self.speed * dt
        self._extend()
        return self.speed


@dataclass
class Report:
    duration: float = 0.0
    ticks: int = 0
    wall_seconds: float = 0.0
    vehicles: int = 0
    mean_in_view: float = 0.0
    junction_passes: dict = field(default_factory=dict)
    close_calls: int = 0
    contacts: int = 0
    close_call_log: list = field(default_factory=list)
    closest_gap: float = math.inf
    cars_recycled: int = 0
    brakes: list = field(default_factory=list)
    ml_contacts: list = field(default_factory=list)
    red_light_runs: list = field(default_factory=list)
    ml_red_stops: int = 0
    ml_distance: float = 0.0


def print_report(r):
    minutes = max(r.duration / 60.0, 1e-6)
    passes = r.junction_passes
    print("\n" + "=" * 60)
    print("LIVE TRAFFIC REPORT")
    print("=" * 60)
    print(f"Simulated            {r.duration:.0f} s, at {r.duration / max(r.wall_seconds, 1e-6):.2f}x real time")
    print(f"Vehicles             {r.vehicles} total, {r.mean_in_view:.1f} near the intersection on average")
    print(f"Junction passes      {passes.get('straight', 0)} straight, {passes.get('left', 0)} left, "
          f"{passes.get('right', 0)} right")
    gap = f"{r.closest_gap:.2f} m" if math.isfinite(r.closest_gap) else "none"
    print(f"Close calls          {r.close_calls} (closing fast within {CLOSE_CALL_GAP} m), of which {r.contacts} "
          f"overlapped; closest moving gap {gap}")
    print(f"Cars moved back      {r.cars_recycled}, off-screen, to keep the intersection busy")
    print(f"vehicle_ml           drove {r.ml_distance:.0f} m; touched another car {len(r.ml_contacts)} times")
    ml_runs = sum(run["ml"] for run in r.red_light_runs)
    print(f"Red lights           run {len(r.red_light_runs) - ml_runs} times by traffic, {ml_runs} by vehicle_ml; "
          f"vehicle_ml stopped for {r.ml_red_stops}")
    print(f"Brake events         {len(r.brakes)} ({len(r.brakes) / minutes:.1f} per minute)")
    for brake in r.brakes[:10]:
        print(f"  t={brake['t']:6.1f} s  {brake['reason']}  [{brake.get('context', '')}]")
    if len(r.brakes) > 10:
        print(f"  ...and {len(r.brakes) - 10} more")


# ---------------------------------------------------------------------------
# CARLA
# ---------------------------------------------------------------------------
class Monitor:
    def __init__(self, carla, client):
        self.carla = carla
        self.world = client.get_world()
        self.session = requests.Session()
        self.world.wait_for_tick(10.0)  # until the first tick, a new client reads the spectator as (0, 0)
        view = self.world.get_spectator().get_transform().location
        self.center = (view.x, view.y)
        self.counter = JunctionCounter(self.center)
        self.report = Report()
        self.rng = random.Random(0)
        self.now = 0.0
        self.braked_at = None
        self.pair_seen = {}
        self.ml_touching = set()
        self.states = {}
        self.actors = {}
        self.last_recycle = 0.0
        self.shown_signal = None
        points = self.world.get_map().get_spawn_points()
        self.recycle_points = [p for p in points if RECYCLE_NEAR[0] <= math.hypot(
            p.location.x - self.center[0], p.location.y - self.center[1]) <= RECYCLE_NEAR[1]]
        self.ml, self.driver = self._spawn_ml(view, points)
        self.lights = RedLightWatch(carla, self.world)
        self.recycle_points = clear_of_stop_lines(self.recycle_points, self.lights.stop_line_positions())
        self.waiting_at_red = False

    def _spawn_ml(self, view, points):
        bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        bp.set_attribute("role_name", "vehicle_ml")
        bp.set_attribute("color", ML_PAINT)
        cars = list(self.world.get_actors().filter("vehicle.*"))
        for point in sorted(points, key=lambda p: p.location.distance(view)):
            if any(c.get_location().distance(point.location) < 10.0 for c in cars):
                continue
            ml = self.world.try_spawn_actor(bp, point)
            if ml is not None:
                ml.set_simulate_physics(False)  # LaneDriver owns the pose
                waypoint = self.world.get_map().get_waypoint(point.location)
                print(f"[TRAFFIC] vehicle_ml joined traffic at x={point.location.x:.0f} y={point.location.y:.0f}.",
                      flush=True)
                return ml, LaneDriver(waypoint, self.rng)
        raise RuntimeError("No free spawn point for vehicle_ml.")

    def _gate(self):
        try:
            return self.session.get(GATE_URL, timeout=0.5).json()
        except (requests.RequestException, ValueError):
            return {}

    def _release_gate(self):
        body = {"emergency_brake": False, "reason": None, "ttc": None,
                "updated_at": datetime.now().astimezone().isoformat()}
        self.session.put(GATE_URL, json=body, timeout=2.0).raise_for_status()

    def _brake_context(self, reason, me):
        """Where the other vehicle was, which shows whether this was a crossing or just a car ahead."""
        match = re.search(r"carla:(\d+)", reason)
        other = self.states.get(int(match.group(1))) if match else None
        if other is None:
            return "other vehicle unknown"
        ahead, left = relative_position(me, other)
        heading = abs((other[4] - me[4] + 180.0) % 360.0 - 180.0)
        return (f"other {ahead:.1f} m ahead, {left:.1f} m left, heading {heading:.0f} deg off, "
                f"at {math.hypot(other[2], other[3]):.1f} m/s; me at {self.driver.speed:.1f} m/s")

    def _recycle(self):
        if self.now - self.last_recycle < RECYCLE_INTERVAL or not self.recycle_points:
            return
        self.last_recycle = self.now
        far = [i for i, s in self.states.items()
               if i != self.ml.id and math.hypot(s[0] - self.center[0], s[1] - self.center[1]) > RECYCLE_DISTANCE]
        if not far:
            return
        point = self.rng.choice(self.recycle_points)
        for s in self.states.values():
            if math.hypot(s[0] - point.location.x, s[1] - point.location.y) < RECYCLE_CLEARANCE:
                return  # that road is occupied; try another next time
        self.actors[self.rng.choice(far)].set_transform(point)
        self.report.cars_recycled += 1

    def _bring_ml_back_if_far(self, x, y):
        """The ML car wanders too. Bring it back while it's out of sight, so it stays in the action."""
        if math.hypot(x - self.center[0], y - self.center[1]) <= RECYCLE_DISTANCE or not self.recycle_points:
            return
        point = self.rng.choice(self.recycle_points)
        if any(math.hypot(s[0] - point.location.x, s[1] - point.location.y) < RECYCLE_CLEARANCE
               for i, s in self.states.items() if i != self.ml.id):
            return
        self.driver = LaneDriver(self.world.get_map().get_waypoint(point.location), self.rng)

    def _drive_ml(self, dt):
        x, y, z, yaw = self.driver.pose()
        me = (x, y, 0.0, 0.0, yaw)
        others = [s for i, s in self.states.items() if i != self.ml.id]

        if self.braked_at is None:
            gate = self._gate()
            if gate.get("emergency_brake") is True:
                self.braked_at = self.now
                context = self._brake_context(gate.get("reason") or "", me)
                self.report.brakes.append({"t": round(self.now, 1), "reason": gate.get("reason"), "context": context})
                print(f"[TRAFFIC] t={self.now:.1f} s: BRAKE: {gate.get('reason')} [{context}]", flush=True)
        elif self.now - self.braked_at >= BRAKE_HOLD:
            self._release_gate()
            self.braked_at = None

        target = min(CRUISE_SPEED, speed_for_gap(lead_gap(me, others) - FOLLOW_GAP))
        gap = red_light_gap(self.driver.path_ahead(RED_LOOK), self.lights.not_green())
        if gap < RED_LOOK:
            target = min(target, speed_for_gap(gap - STOP_SHORT))
            if target < 0.5 and not self.waiting_at_red:
                self.waiting_at_red = True
                self.report.ml_red_stops += 1
                print(f"[TRAFFIC] t={self.now:.1f} s: vehicle_ml stopping for a red light.", flush=True)
        else:
            self.waiting_at_red = False
        self.driver.step(dt, target, self.braked_at is not None)
        x, y, z, yaw = self.driver.pose()
        self.ml.set_transform(self.carla.Transform(self.carla.Location(x=x, y=y, z=z + Z_CLEARANCE),
                                                   self.carla.Rotation(yaw=yaw)))
        self._show_signal()
        self._note_ml_contact(x, y, yaw)
        self._bring_ml_back_if_far(x, y)

    def _show_signal(self):
        wanted = self.driver.signal()
        if wanted == self.shown_signal:
            return
        self.shown_signal = wanted
        lights = self.carla.VehicleLightState
        self.ml.set_light_state({LEFT: lights.LeftBlinker, RIGHT: lights.RightBlinker}.get(wanted, lights.NONE))

    def _note_ml_contact(self, x, y, yaw):
        mine = outline(x, y, yaw)
        touching = set()
        for i, s in self.states.items():
            if (i != self.ml.id and math.hypot(s[0] - x, s[1] - y) <= PAIR_RANGE
                    and rectangles_gap(mine, outline(s[0], s[1], s[4])) == 0.0):
                touching.add(i)
        for i in touching - self.ml_touching:  # once per touch, not once per tick
            self.report.ml_contacts.append({"t": round(self.now, 1), "with": self.actors[i].type_id})
            print(f"[TRAFFIC] t={self.now:.1f} s: vehicle_ml TOUCHED {self.actors[i].type_id}", flush=True)
        self.ml_touching = touching

    def _measure(self, states):
        moving_gap = self.report.closest_gap
        ids = list(states)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                sa, sb = states[a], states[b]
                if math.hypot(sa[0] - sb[0], sa[1] - sb[1]) > PAIR_RANGE:
                    continue
                if not is_close_call_speed(sa, sb):
                    continue
                gap = rectangles_gap(outline(sa[0], sa[1], sa[4]), outline(sb[0], sb[1], sb[4]))
                moving_gap = min(moving_gap, gap)
                key = (a, b)
                if gap < CLOSE_CALL_GAP and self.now - self.pair_seen.get(key, -math.inf) > CLOSE_CALL_COOLDOWN:
                    self.pair_seen[key] = self.now
                    self.report.close_calls += 1
                    self.report.contacts += gap == 0.0
                    ahead, left = relative_position(sa, sb)
                    self.report.close_call_log.append({
                        "t": round(self.now, 1), "gap": round(gap, 2), "ids": [a, b],
                        "closing": round(closing_speed(sa, sb), 1), "ahead": round(ahead, 1), "left": round(left, 1),
                        "speeds": [round(math.hypot(sa[2], sa[3]), 1), round(math.hypot(sb[2], sb[3]), 1)],
                        "heading_off": round(abs((sb[4] - sa[4] + 180.0) % 360.0 - 180.0)),
                        "from_center": round(math.hypot(sa[0] - self.center[0], sa[1] - self.center[1])),
                    })
        self.report.closest_gap = moving_gap

    def run(self, duration):
        started_wall = time.time()
        started_sim = None
        in_view_total = 0
        last_log = 0.0
        last_ml = None
        ever_seen = set()
        while duration <= 0 or self.now < duration:
            snapshot = self.world.wait_for_tick(10.0)
            sim = snapshot.timestamp.elapsed_seconds
            started_sim = sim if started_sim is None else started_sim
            self.now = sim - started_sim
            self.report.ticks += 1

            states = {}
            actors = {}
            in_view = 0
            for actor in self.world.get_actors().filter("vehicle.*"):
                actors[actor.id] = actor
                tf, v = actor.get_transform(), actor.get_velocity()
                x, y = tf.location.x, tf.location.y
                states[actor.id] = (x, y, v.x, v.y, tf.rotation.yaw)
                ever_seen.add(actor.id)
                self.counter.update(actor.id, x, y, tf.rotation.yaw)
                if math.hypot(x - self.center[0], y - self.center[1]) <= VIEW_RADIUS:
                    in_view += 1
            in_view_total += in_view
            self.states, self.actors = states, actors
            self.lights.update(states, self.now, self.ml.id, "TRAFFIC")
            self._measure({i: s for i, s in states.items() if i != self.ml.id})
            self._drive_ml(snapshot.timestamp.delta_seconds)
            self._recycle()

            x, y, _, _ = self.driver.pose()
            if last_ml is not None and math.hypot(x - last_ml[0], y - last_ml[1]) < 5.0:
                self.report.ml_distance += math.hypot(x - last_ml[0], y - last_ml[1])
            last_ml = (x, y)

            if self.now - last_log >= REPORT_INTERVAL:
                last_log = self.now
                print(f"[TRAFFIC] t={self.now:5.0f} s: {len(states)} vehicles, {in_view} near the intersection, "
                      f"{sum(self.counter.counts.values())} junction passes, {len(self.report.brakes)} brakes",
                      flush=True)

        r = self.report
        r.duration = self.now
        r.wall_seconds = time.time() - started_wall
        r.vehicles = len(ever_seen)
        r.mean_in_view = in_view_total / max(r.ticks, 1)
        r.junction_passes = dict(self.counter.counts)

    def finish(self):
        self.report.red_light_runs = self.lights.runs
        r = self.report
        if r.duration == 0.0:
            r.duration = self.now
            r.junction_passes = dict(self.counter.counts)
        self.ml.destroy()
        try:
            self._release_gate()
        except requests.RequestException:
            pass
        print_report(r)
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / f"traffic_{datetime.now():%Y%m%d_%H%M%S}.json"
        data = asdict(r)
        if not math.isfinite(data["closest_gap"]):
            data["closest_gap"] = None
        path.write_text(json.dumps(data, indent=2))
        print(f"\nReport: {path}", flush=True)


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=60.0, help="simulated seconds; 0 runs until Ctrl+C")
    args = parser.parse_args()

    for name in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _raise_keyboard_interrupt)

    import carla

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(30.0)
    monitor = Monitor(carla, client)
    try:
        monitor.run(args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
