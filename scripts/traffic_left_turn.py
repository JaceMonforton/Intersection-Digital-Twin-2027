"""
Safe vs unsafe left turn across real traffic.

vehicle_ml waits at a red light to turn left across autopilot cars that have the green:
    unsafe: it runs its red light into the oncoming traffic, and needs the brake to avoid contact
    safe:   it waits for its light to turn green and the oncoming traffic to stop, then turns

Unlike left_turn_scenario.py, the other cars here are real: they accelerate, queue and react to the
ML car, so results vary from run to run. Run it several times before drawing conclusions.

Normally started by run_simulation.py. Needs the processor and test_traffic.py running.
    python scripts/traffic_left_turn.py --case both
"""
import argparse
import json
import math
import random
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import requests
from central_gpu_processor import (
    COLLISION_RADIUS,
    LABEL_LIFE_TIME,
    ML_PAINT,
    Track,
    approaching_from_behind,
    deceleration_to_avoid_crash,
    time_to_collision,
)
from left_turn_scenario import (
    DRAC_MIN_GAP,
    SAFE_COLOR,
    SIGNAL_OFF_AFTER,
    STOP_LINE_SETBACK,
    TURN_ACCEL,
    TURN_SPEED,
    UNSAFE_COLOR,
    UNSAFE_GAP,
    ZONE_RADIUS,
    Result,
    Turner,
    classify,
    conflict_point,
    find_left_turn_lanes,
    measured_pet,
    outline,
    rectangles_gap,
    release_gate,
    time_to_cover,
    twin_transform,
    yaw_difference,
)
from traffic_monitor import RedLightWatch, clear_of_stop_lines

# --- Configuration ---
CARLA_HOST = "localhost"
CARLA_PORT = 2000
GATE_URL = "http://localhost:8080/api/2/things/org.example:vehicle_ml/features/SafetyGate/properties"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

SAFE_MARGIN = 3.0         # s after the turner would arrive that the lane must stay clear for a safe turn
UNSAFE_WINDOW = 0.6       # s of tolerance around the target gap; outside it, wait for the next car
ETA_MIN_SPEED = 4.0       # m/s assumed for a slow or stopped car, so it isn't treated as never arriving
ON_PATH_OFFSET = 2.5      # m from the lane centre that counts as being in the lane
ON_PATH_HEADING = 40.0    # degrees from the lane's direction that counts as following it
WATCH_DISTANCE = 90.0     # m up the oncoming lane that the turner looks
FEED_DISTANCE = 55.0      # m before the conflict point that a car is placed if the lane is empty
FEED_INTERVAL = 6.0       # s between placing cars, so oncoming traffic arrives as a flow
CLEAR_RADIUS = 9.0        # m around the stop line cleared before the turner appears
PASSED_HOLD = 1.0         # s the brake is held at least, before checking whether it can be released
RED_HOLD = 6.0            # s the safe run waits at red while oncoming traffic has the green
AMBER_TIME = 2.5          # s of amber before the oncoming light turns red and the turner's turns green
LIGHT_RADIUS = 45.0       # m around the junction that counts as this junction's lights
MOVING_SPEED = 1.0        # m/s; slower than this, a car has stopped to yield and isn't a reason to wait
PATH_WIDTH = 1.7          # m either side of the lane centre that a car has to be in to block the turner
BLOCK_LOOK = 12.0         # m ahead along its path that the turner checks
RESUME_MARGIN = 0.5       # s spare that a stopped turner wants, on top of the time it needs to clear
GAP_MEASURE_RANGE = 12.0  # m between centres; further apart, outlines can't be close
WAIT_LIMIT = 75.0         # s the turner waits for a suitable gap before giving up on the run
PAST_CONFLICT_DONE = 20.0 # m past the conflict point at which the turn is complete
RUN_LIMIT = 45.0          # s from leaving the stop line before a run is cut off
STATUS_INTERVAL = 5.0     # s between progress lines while the turner is held
SETTLE_BETWEEN = 4.0      # s between runs


# ---------------------------------------------------------------------------
# Gap acceptance (no CARLA needed)
# ---------------------------------------------------------------------------
def light_role(light_yaw, turner_yaw):
    """Whose light is this, judging by the direction of the lane it controls?"""
    change = abs(yaw_difference(light_yaw, turner_yaw))
    if change < 30.0:
        return "turner"
    return "oncoming" if change > 150.0 else "side"


def oncoming_etas(path, conflict_s, vehicles, moving_only=False):
    """Maps vehicle id -> seconds until it reaches the conflict point, for vehicles in the oncoming
    lane and still short of it. vehicles maps id -> (x, y, vx, vy, yaw)."""
    etas = {}
    for vehicle_id, (x, y, vx, vy, yaw) in vehicles.items():
        s = path.nearest_s(x, y)
        distance = conflict_s - s
        if distance <= 0.0 or distance > WATCH_DISTANCE:
            continue
        px, py = path.position(s)
        if math.hypot(px - x, py - y) > ON_PATH_OFFSET:
            continue
        if abs(yaw_difference(path.yaw(s), yaw)) > ON_PATH_HEADING:
            continue
        if moving_only and math.hypot(vx, vy) < MOVING_SPEED:
            continue
        etas[vehicle_id] = distance / max(math.hypot(vx, vy), ETA_MIN_SPEED)
    return etas


def should_go(case, etas, turn_time, zone_occupied):
    """Is this the gap this run is looking for? turn_time is how long the turner needs to reach the
    conflict point from the stop line."""
    if case == "unsafe":
        if not etas:
            return False
        # Aim to arrive UNSAFE_GAP after the next car: too soon to clear it
        target = turn_time - UNSAFE_GAP
        return abs(min(etas.values()) - target) <= UNSAFE_WINDOW
    return not zone_occupied and all(eta >= turn_time + SAFE_MARGIN for eta in etas.values())


def path_blocked(path, s, vehicles):
    """Is a car actually standing on the turner's path just ahead? It doesn't drive into it."""
    for step in range(3, int(BLOCK_LOOK) + 1):
        x, y = path.position(s + step)
        for vx_state in vehicles.values():
            if math.hypot(vx_state[0] - x, vx_state[1] - y) <= PATH_WIDTH:
                return True
    return False


def can_resume(etas, zone_occupied, clear_time):
    """Is it clear to carry on after a brake? Moving off just because one car has passed would only
    bring the brake straight back for the next, so it needs a gap long enough to clear the lane."""
    return not zone_occupied and all(eta >= clear_time + RESUME_MARGIN for eta in etas.values())


class ZoneLog:
    """Times each vehicle was inside the conflict zone."""

    def __init__(self, center):
        self.center = center
        self.times = {}

    def update(self, vehicle_id, x, y, t):
        if math.hypot(x - self.center[0], y - self.center[1]) <= ZONE_RADIUS:
            first = self.times.get(vehicle_id, (t, t))[0]
            self.times[vehicle_id] = (first, t)

    def occupied(self, t, except_id=None):
        return any(last >= t - 0.11 for vid, (_, last) in self.times.items() if vid != except_id)

    def smallest_pet(self, turner_id):
        """The tightest measured gap between the turner and any vehicle sharing the zone."""
        mine = self.times.get(turner_id)
        pets = [measured_pet(mine, zone) for vid, zone in self.times.items() if vid != turner_id]
        pets = [p for p in pets if p is not None]
        return min(pets) if pets else None


class RunResult:
    """Accumulates the safety measures of one run against every other vehicle."""

    def __init__(self, case):
        self.result = Result(case)
        self.waited = 0.0
        self.gave_up = False
        self.ran_red = False
        self.brake_reason = None

    def observe(self, me, me_yaw, others):
        r = self.result
        my_track = Track("vehicle_ml", me[0], me[1], me[2], me[3], yaw=me_yaw)
        my_outline = outline(me[0], me[1], me_yaw)
        for vehicle_id, (x, y, vx, vy, yaw) in others.items():
            other = Track(f"carla:{vehicle_id}", x, y, vx, vy, yaw=yaw)
            if approaching_from_behind(my_track, other):
                continue  # a car queueing behind the turner says nothing about the turn
            separation = math.hypot(x - me[0], y - me[1])
            ttc = time_to_collision(my_track, other)
            r.min_ttc = min(r.min_ttc, ttc)
            r.min_separation = min(r.min_separation, separation)
            if separation <= GAP_MEASURE_RANGE:
                gap = rectangles_gap(my_outline, outline(x, y, yaw))
                r.min_gap = min(r.min_gap, gap)
                r.contact = r.contact or gap == 0.0
            if ttc < math.inf and gap_is_meaningful(separation):
                r.max_drac = max(r.max_drac, deceleration_to_avoid_crash(my_track, other))

    def finish(self, zones, turner_id):
        r = self.result
        r.pet = zones.smallest_pet(turner_id)
        r.turner_entered_zone = turner_id in zones.times
        r.verdict, r.reasons = classify(r)
        if self.ran_red:
            r.verdict = "UNSAFE"
            r.reasons.insert(0, "ran a red light into oncoming traffic")
        if self.gave_up:
            r.verdict = "NO GAP"
            r.reasons = [f"no suitable gap in {WAIT_LIMIT:.0f} s"]
        return r


def gap_is_meaningful(separation):
    return separation - COLLISION_RADIUS >= DRAC_MIN_GAP


def format_value(v, unit=" s"):
    return "never" if v is None or not math.isfinite(v) else f"{v:.2f}{unit}"


def print_comparison(runs):
    rows = [
        ("Verdict", lambda r: r.result.verdict),
        ("Waited at stop line", lambda r: f"{r.waited:.1f} s"),
        ("Min TTC", lambda r: format_value(r.result.min_ttc)),
        ("Measured PET", lambda r: format_value(r.result.pet)),
        ("Max DRAC", lambda r: f"{r.result.max_drac:.2f} m/s2"),
        ("Closest gap", lambda r: format_value(r.result.min_gap, " m")),
        ("Contact", lambda r: "YES" if r.result.contact else "no"),
        ("Brake fired", lambda r: format_value(r.result.brake_fired_at).replace("never", "no")),
    ]
    print("\n" + "=" * 60)
    print("LEFT TURN IN REAL TRAFFIC")
    print("=" * 60)
    print(f"{'':<22}" + "".join(f"{case:<18}" for case in runs))
    for label, read in rows:
        print(f"{label:<22}" + "".join(f"{read(run):<18}" for run in runs.values()))
    for case, run in runs.items():
        print(f"\n{case.upper()} run: {run.result.verdict}")
        for reason in run.result.reasons:
            print(f"  - {reason}")
        if run.brake_reason:
            print(f"  - brake reason: {run.brake_reason}")


# ---------------------------------------------------------------------------
# CARLA
# ---------------------------------------------------------------------------
class TrafficScenario:
    def __init__(self, carla, client):
        self.carla = carla
        self.world = client.get_world()
        self.session = requests.Session()
        self.rng = random.Random()
        self.world.wait_for_tick(10.0)  # until the first tick, a new client reads the spectator as (0, 0)
        self.turner_path, entry_s, oncoming_paths = find_left_turn_lanes(carla, self.world)
        self.oncoming_path = oncoming_paths[0]  # the first lane it crosses; there may be more
        self.oncoming_lanes = [(path, conflict_point(self.turner_path, path)[3]) for path in oncoming_paths]
        self.start_s = max(0.0, entry_s - STOP_LINE_SETBACK)
        self.conflict = conflict_point(self.turner_path, self.oncoming_path)
        self.center = (self.conflict[0], self.conflict[1])
        self.turn_time = time_to_cover(self.conflict[2] - self.start_s, TURN_SPEED, TURN_ACCEL)
        self.far_points = [p for p in self.world.get_map().get_spawn_points()
                           if p.location.distance(carla.Location(x=self.center[0], y=self.center[1])) > 120.0]
        print(f"[SCENARIO] The turn takes {self.turn_time:.1f} s to reach the conflict point.", flush=True)
        self.lights = self._junction_lights()
        self.red_watch = RedLightWatch(carla, self.world)
        self.far_points = clear_of_stop_lines(self.far_points, self.red_watch.stop_line_positions())

    def _junction_lights(self):
        """This junction's lights, grouped by whose they are, and held under this script's control."""
        turner_yaw = self.turner_path.yaw(self.start_s)
        groups = {"turner": [], "oncoming": [], "side": []}
        for light in self.world.get_actors().filter("traffic.traffic_light"):
            stops = light.get_stop_waypoints()
            if not stops:
                continue
            location = stops[0].transform.location
            if math.hypot(location.x - self.center[0], location.y - self.center[1]) > LIGHT_RADIUS:
                continue
            groups[light_role(stops[0].transform.rotation.yaw, turner_yaw)].append(light)
            light.freeze(True)
        print(f"[SCENARIO] Lights at this junction: {', '.join(f'{len(v)} {k}' for k, v in groups.items())}.", flush=True)
        return groups

    def set_lights(self, turner, oncoming):
        """Set the turner's and oncoming lights; the side road stays red throughout."""
        for group, state in ((self.lights["turner"], turner), (self.lights["oncoming"], oncoming),
                             (self.lights["side"], self.carla.TrafficLightState.Red)):
            for light in group:
                light.set_state(state)

    def release_lights(self):
        for group in self.lights.values():
            for light in group:
                light.freeze(False)

    def _vehicles(self, exclude=None):
        states, actors = {}, {}
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.id == exclude:
                continue
            tf, v = actor.get_transform(), actor.get_velocity()
            states[actor.id] = (tf.location.x, tf.location.y, v.x, v.y, tf.rotation.yaw)
            actors[actor.id] = actor
        return states, actors

    def _etas(self, states, moving_only=False):
        """The soonest arrival of each vehicle, across every oncoming lane the turner has to cross."""
        etas = {}
        for path, conflict_s in self.oncoming_lanes:
            for vehicle_id, eta in oncoming_etas(path, conflict_s, states, moving_only).items():
                etas[vehicle_id] = min(eta, etas.get(vehicle_id, math.inf))
        return etas

    def _move_far_away(self, actor):
        if self.far_points:
            actor.set_transform(self.rng.choice(self.far_points))

    def _clear_stop_line(self):
        """Autopilot cars may be sitting where the turner needs to appear, or on the path it needs."""
        spots = [self.turner_path.position(self.start_s + step) for step in range(0, 45, 3)]
        states, actors = self._vehicles()
        for vehicle_id, state in states.items():
            if any(math.hypot(state[0] - x, state[1] - y) <= CLEAR_RADIUS for x, y in spots[:1]) or any(
                    math.hypot(state[0] - x, state[1] - y) <= PATH_WIDTH + 1.0 for x, y in spots):
                self._move_far_away(actors[vehicle_id])
        self.world.wait_for_tick(5.0)

    def _feed_oncoming(self, states, actors, etas, turner_id):
        """Autopilot routes are random, so make sure someone is coming down the oncoming lane."""
        if etas:
            return
        far = [i for i, s in states.items() if math.hypot(s[0] - self.center[0], s[1] - self.center[1]) > 100.0]
        if not far:
            far = [i for i, s in states.items() if math.hypot(s[0] - self.center[0], s[1] - self.center[1]) > 45.0]
        far = [i for i in far if i != turner_id]
        if not far:
            return
        s = max(0.0, self.conflict[3] - FEED_DISTANCE)
        x, y = self.oncoming_path.position(s)
        if any(math.hypot(o[0] - x, o[1] - y) < 12.0 for o in states.values()):
            return
        actors[self.rng.choice(far)].set_transform(twin_transform(self.carla, self.oncoming_path, s))

    def _spawn_ml(self):
        bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        bp.set_attribute("role_name", "vehicle_ml")
        bp.set_attribute("color", ML_PAINT)
        ml = self.world.spawn_actor(bp, twin_transform(self.carla, self.turner_path, self.start_s))
        ml.set_simulate_physics(False)  # the plan owns the pose; its speed comes from Turner
        ml.set_light_state(self.carla.VehicleLightState.LeftBlinker)
        return ml

    def _gate(self):
        try:
            return self.session.get(GATE_URL, timeout=0.5).json()
        except (requests.RequestException, ValueError):
            return {}

    def run_case(self, case):
        release_gate(self.session)
        self._clear_stop_line()
        ml = self._spawn_ml()
        try:
            return self._drive_case(case, ml)
        finally:
            ml.destroy()
            self.world.wait_for_tick(5.0)
            release_gate(self.session)

    def _drive_case(self, case, ml):
        carla = self.carla
        run = RunResult(case)
        zones = ZoneLog(self.center)
        turner = Turner(math.inf, self.start_s)
        color = carla.Color(*(UNSAFE_COLOR if case == "unsafe" else SAFE_COLOR))
        label_at = carla.Location(x=self.center[0], y=self.center[1], z=4.0)
        conflict_s = self.conflict[2]
        lights = carla.TrafficLightState
        self.set_lights(turner=lights.Red, oncoming=lights.Green)
        print(f"\n[{case.upper()}] vehicle_ml is stopped at a RED light; oncoming traffic has the green.", flush=True)

        start, departed, braked_at, last_feed, signalling = None, None, None, -math.inf, True
        phase = "red"
        blocked_since = None
        settle_until = None
        last_status = 0.0
        while True:
            snapshot = self.world.wait_for_tick(10.0)
            now = snapshot.timestamp.elapsed_seconds
            start = now if start is None else start
            t = now - start
            dt = snapshot.timestamp.delta_seconds

            states, actors = self._vehicles(exclude=ml.id)
            for vehicle_id, state in states.items():
                zones.update(vehicle_id, state[0], state[1], t)
            self.red_watch.update(states, t, label=case.upper())
            etas = self._etas(states)
            # A stopped car in the zone is handled by path_blocked; only moving ones mean "wait"
            zone_occupied = any(math.hypot(a[2], a[3]) >= MOVING_SPEED
                                and math.hypot(a[0] - self.center[0], a[1] - self.center[1]) <= ZONE_RADIUS
                                for a in states.values())

            if case == "safe" and phase == "red" and t >= RED_HOLD:
                phase = "amber"
                self.set_lights(turner=lights.Red, oncoming=lights.Yellow)
                print(f"[SAFE] t={t:.1f} s: oncoming light turns AMBER.", flush=True)
            elif case == "safe" and phase == "amber" and t >= RED_HOLD + AMBER_TIME:
                phase = "green"
                self.set_lights(turner=lights.Green, oncoming=lights.Red)
                print(f"[SAFE] t={t:.1f} s: oncoming light turns RED; vehicle_ml's light turns GREEN.", flush=True)

            if departed is None:
                if phase == "red" and t - last_feed >= FEED_INTERVAL:
                    last_feed = t
                    self._feed_oncoming(states, actors, etas, ml.id)
                if case == "unsafe":
                    go = should_go(case, etas, self.turn_time, zone_occupied)
                else:
                    go = phase == "green" and should_go(case, self._etas(states, moving_only=True),
                                                        self.turn_time, zone_occupied)
                if go:
                    run.ran_red = case == "unsafe"
                    departed = t
                    turner.depart_at = t
                    run.waited = t
                    nearest = min(etas.values()) if etas else None
                    action = "RUNNING THE RED LIGHT" if case == "unsafe" else "turning on green"
                    print(f"[{case.upper()}] t={t:.1f} s: {action}; next oncoming car "
                          f"{f'{nearest:.1f} s away' if nearest is not None else 'none in sight'}.", flush=True)
                elif t >= WAIT_LIMIT:
                    run.gave_up = True
                    run.waited = t
                    print(f"[{case.upper()}] Gave up: no suitable gap in {WAIT_LIMIT:.0f} s.", flush=True)
                    break

            gate = self._gate() if departed is not None else {}
            if braked_at is None and gate.get("emergency_brake") is True:
                braked_at = t
                if run.result.brake_fired_at is None:
                    run.result.brake_fired_at = t - departed
                    run.brake_reason = gate.get("reason")
                print(f"[{case.upper()}] t={t:.1f} s: BRAKE: {gate.get('reason')}", flush=True)
            elif braked_at is not None and t - braked_at >= PASSED_HOLD and can_resume(
                    self._etas(states, moving_only=True), zone_occupied, time_to_cover(max(conflict_s + ZONE_RADIUS - turner.s, 0.1), TURN_SPEED, TURN_ACCEL)):
                release_gate(self.session)
                braked_at = None
                print(f"[{case.upper()}] t={t:.1f} s: oncoming traffic has passed, brake released, finishing the turn.",
                      flush=True)

            if braked_at is not None and t - last_status >= STATUS_INTERVAL:
                last_status = t
                moving = self._etas(states, moving_only=True)
                print(f"[{case.upper()}] t={t:.1f} s: still holding; zone "
                      f"{'occupied' if zone_occupied else 'clear'}, next moving car "
                      f"{f'{min(moving.values()):.1f} s away' if moving else 'none'}.", flush=True)

            blocked = departed is not None and path_blocked(self.turner_path, turner.s, states)
            blocked_since = (blocked_since if blocked_since is not None else t) if blocked else None
            if blocked_since is not None and t - blocked_since >= STATUS_INTERVAL:
                blocked_since = t
                print(f"[{case.upper()}] t={t:.1f} s: waiting for a car standing on its path.", flush=True)
            s, v = turner.step(t, dt, braked_at is not None or blocked)
            ml.set_transform(twin_transform(carla, self.turner_path, s))
            x, y = self.turner_path.position(s)
            yaw = self.turner_path.yaw(s)
            zones.update(ml.id, x, y, t)
            vx, vy = v * math.cos(math.radians(yaw)), v * math.sin(math.radians(yaw))
            if departed is not None:
                run.observe((x, y, vx, vy), yaw, states)

            if signalling and s > conflict_s + SIGNAL_OFF_AFTER:
                ml.set_light_state(carla.VehicleLightState.NONE)
                signalling = False

            self.world.debug.draw_string(label_at, f"{case.upper()} LEFT TURN  t={t:.1f}s", color=color, life_time=LABEL_LIFE_TIME)

            if settle_until is None and (s >= self.turner_path.length or s >= conflict_s + PAST_CONFLICT_DONE):
                settle_until = t + 1.0
            if (settle_until is not None and t >= settle_until) or (departed is not None and t - departed >= RUN_LIMIT):
                break

        run.finish(zones, ml.id)
        self.world.debug.draw_string(label_at, f"{case.upper()} LEFT TURN: {run.result.verdict}",
                                     color=color, life_time=SETTLE_BETWEEN)
        time.sleep(SETTLE_BETWEEN)
        return run


def write_report(runs):
    OUTPUT_DIR.mkdir(exist_ok=True)
    report = {}
    for case, run in runs.items():
        data = {k: (v if not isinstance(v, float) or math.isfinite(v) else None) for k, v in asdict(run.result).items()}
        data.update(waited=round(run.waited, 1), brake_reason=run.brake_reason)
        report[case] = data
    path = OUTPUT_DIR / f"traffic_left_turn_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {path}", flush=True)


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=("safe", "unsafe", "both"), default="both")
    args = parser.parse_args()

    for name in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _raise_keyboard_interrupt)

    import carla

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(30.0)
    if not client.get_world().get_settings().synchronous_mode:
        print("[SCENARIO] CARLA isn't synchronous. Start central_gpu_processor.py first.")
        return 1

    scenario = TrafficScenario(carla, client)
    runs = {}
    try:
        for case in ("unsafe", "safe") if args.case == "both" else (args.case,):
            runs[case] = scenario.run_case(case)
    except KeyboardInterrupt:
        pass
    finally:
        scenario.release_lights()
    if runs:
        print_comparison(runs)
        print(f"\nRed lights run by traffic during the turns: {len(scenario.red_watch.runs)}", flush=True)
        write_report(runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
