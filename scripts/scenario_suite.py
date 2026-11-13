"""
The scenarios' machinery around Town 10's junction 189, with the ML car driven by the trained model: Suite places the
ML car, runs a stage tick by tick and scores it from CARLA's true positions. Its two built-in stages are the left turn
across oncoming traffic (test_left_turn) and the blind-corner red-light runner (test_blind_corner); hidden_hazards.py
adds the hidden pedestrian and the oncoming turner, and run_bench.py runs them on many seeds.

Other cars are CARLA's autopilot. Between stages the brake is released and every car is removed. The brake release
here stands in for an operator and only ever happens inside CARLA.
"""
import csv
import json
import math
import os
import queue
import random
import shutil
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from central_gpu_processor import (
    ML_PAINT,
    TRAFFIC_MANAGER_PORT,
    Track,
    approaching_from_behind,
    time_to_collision,
)
from driving_features import STOP_LINE_RANGE, outline, rectangles_gap
from ml_driver import (
    ModelDriver,
    actor_state,
    crossing_ttc,
    is_new_contact,
    smoothness,
)
from model_panel import SUITE_ADDRESS, SafetyReceiver, TelemetrySender
from roadside_link import (
    ROADSIDE_MODES,
    RoadsideReceiver,
    TickAck,
    make_view,
)
from traffic_monitor import (
    GATE_URL,
    LOOK_AHEAD,
    OUTPUT_DIR,
    RED_LOOK,
    LaneDriver,
    RedLightWatch,
    red_light_gap,
    release_gate,
)
from waymo_predictor_node import LEFT, STRAIGHT, pick_branch, yaw_difference

# --- Configuration ---
ROOT = Path(__file__).resolve().parent.parent
CARLA_HOST = "localhost"
CARLA_PORT = 2000
MQTT_HOST = "localhost"
MQTT_PORT = 1883
COMMAND_TOPIC = "device/vehicle_ml/commands"
CONTROL_MODE = "AI TARGET SPEED MODEL + PID"

JUNCTION = (-47.8, 20.4)      # junction 189, as the processor finds it
LIGHT_RADIUS = 40.0           # m from the junction that its lights' stop lines lie within
APPROACH = 55.0               # m before the stop line that the ML car starts
TICK_TIMEOUT = 30.0           # s; CARLA can stall for a few seconds while cars come and go
SETTLE_TICKS = 25             # ticks after spawning before a test's clock starts
BRAKE_HOLD = 3.0              # s a latched brake stays on before the stand-in operator may release it
TRUCK = "vehicle.carlamotors.carlacola"
MOVING = 0.5                  # m/s
ASSUMED_BRAKING = 6.0         # m/s^2, for the stopping distance in the alert

T2_LIMIT = 150.0
T2_ONCOMING = 12.0            # m/s
T2_TURNED = 60.0              # degrees of heading change
T2_START_BACK = 15.0          # m before the stop line that the ML car starts; further back is another junction
T2_FEED_FROM = 100.0          # m before the oncoming stop line that cars start, on a road straight enough for a fixed heading
T2_REMOVE_BEYOND = 45.0       # m past the junction that an oncoming car is removed
T2_PREFILL_TO = 0.0           # m before the oncoming stop line that traffic is already on the road down to
T2_FIRST_WAVE = 30.0          # s of steady traffic at the start, so that the ML car always meets it
T2_HEADWAY = (2.0, 6.0)       # s between cars in a wave, too short to turn through
T2_GAP = (11.0, 14.0)         # s between waves, long enough to turn through
T2_WAVE_CARS = (3, 6)         # cars in each later wave
T5_WAITED = 3.0               # s of giving way before the light changes
T5_YELLOW = 3.0               # s of yellow before red
T5_CAN_STOP = 3.5             # m/s^2; oncoming cars that would need more than this carry on through
T2_CLEAR = 15.0               # m from any other car for the stand-in operator to release the brake
T2_FACED = 25.0               # m; oncoming cars must come this close at the junction or the test tested nothing
T2_WAITED = 1.0               # s stopped, with a car due within T2_TOO_SOON, that counts as giving way
T2_TOO_SOON = 7.0             # s; about what the turn takes from a stop plus a margin
T2_SAFE_GAP = 1.5             # m; the closest any car may come
T3_RUNUP = 45.0               # m before its stop line that the speeding car starts
T3_LIMIT = 50.0
T3_SPEED = 10.0               # m/s
T3_LEAD = 0.0                 # s the speeding car is launched before arriving together would need
T3_REST = 6.0                 # s after stopping before scoring
T3_AFTER = 15.0               # s after the launch that the hazard has passed
T3_TRUCK_BACK = (10.0, 14.0, 18.0, 22.0)  # m before the speeding car's stop line to try parking the truck
T3_TRUCK_SIDE = (4.0, 3.5, 5.0, -4.0)     # m to the right of its lane
OFF_LANE_LIMIT = 1.0          # m; normal driving stays within about 0.6 m
JUNCTION_ID = 189
TRAFFIC_CLEARANCE = 30.0      # m from the ML car's start that no background car is spawned within
# A safety brake counts as a false alarm when CARLA's true TTC would have stayed at or above FALSE_SAFE_TTC from FALSE_BEFORE s
# before it to FALSE_AFTER s after it had the car carried on without braking, with no collision then. See false_brakes.
FALSE_SAFE_TTC, FALSE_BEFORE, FALSE_AFTER = 3.0, 1.0, 3.0

TAGS = {"SCENARIO": "96", "MQTT / DITTO": "94", "ALERT": "91"}
STATUS_INTERVAL = 0.2  # seconds between redraws of the live table
CSV_COLUMNS = ("wall_time", "test", "sim_t", "x", "y", "yaw", "speed", "target_speed", "steer", "throttle", "brake",
               "speed_error", "model_driving", "gate_engaged", "min_ttc", "path_ttc", "pet", "stop_gap", "min_gap", "off_lane")


# ---------------------------------------------------------------------------
# Pure logic (no CARLA needed)
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    test: int
    name: str
    passed: bool = False
    checks: dict = field(default_factory=dict)
    collisions: list = field(default_factory=list)
    red_light_violations: int = 0
    hazard_gap: float = None  # m; how close the test's hazard came, where it has one
    stopping_margin: float = None
    min_gap: float = None
    min_ttc: float = None
    brake_events: int = 0
    brake_at: float = None
    brake_ttc: float = None
    mqtt_latency_ms: float = None
    gate_poll_ms: float = None
    worst_off_lane: float = 0.0
    skipped_ticks: int = 0  # ticks the driver never saw, because a loop took longer than one; above 0 the run isn't the one designed
    monitor: dict = field(default_factory=dict)  # per rule of the car's own safety monitor: {"ticks": n, "times": n}
    distance: float = 0.0
    duration: float = 0.0
    smoothness: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    # From CARLA's true positions, not what the gate perceived, so they measure the outcome whatever the perception
    true_min_ttc: float = None    # s; None if nothing was ever on a collision course
    false_brakes: int = 0
    clearance_s: float = None     # s from first entering junction 189 to first leaving it; None if it never got through
    traffic: int = 0              # background cars


def true_ttc(ego, others, walkers):
    """CARLA's true straight-line TTC from the ML car to the nearest threat, as the gate measures it, but leaving out
    a car coming up from behind as the gate does. ego and others as (x, y, vx, vy, yaw), walkers as (x, y, vx, vy)."""
    me = Track("ego", *ego[:4], yaw=ego[4])
    threats = [Track("car", *o[:4], yaw=o[4]) for o in others] + [Track("person", *w[:4]) for w in walkers]
    return min((time_to_collision(me, t) for t in threats if not approaching_from_behind(me, t)), default=math.inf)


def false_brakes(engagements, states, collisions, safe=FALSE_SAFE_TTC, before=FALSE_BEFORE, after=FALSE_AFTER):
    """How many brake engagements had no true danger behind them. A brake that works stops the danger coming, so after
    it the ML car is judged where it would have been had it carried on at the speed and heading it had as the brake
    came on: a brake is false only if even then the true TTC stays at or above safe. engagements and collisions are
    times in s, states (time, ego, others, walkers) for each tick, as true_ttc takes them. Straight-line TTC still
    can't see a car turning across, so a brake for one can still count as false."""
    count = 0
    for at in engagements:
        if not any(s[0] >= at for s in states):
            continue
        # The last tick before the brake came on, when the car still had its own speed
        start = max((s for s in states if s[0] < at), key=lambda s: s[0], default=states[0])
        x, y, vx, vy, yaw = start[1]
        window = [true_ttc(ego if t < at else (x + vx * (t - start[0]), y + vy * (t - start[0]), vx, vy, yaw), others, walkers)
                  for t, ego, others, walkers in states if at - before <= t <= at + after]
        if all(ttc >= safe for ttc in window) and not any(at - before <= c <= at + after for c in collisions):
            count += 1
    return count


def finish(result, **checks):
    """Fills in the checks and the result. A test passes only if every check does."""
    result.checks = {name: bool(value) for name, value in checks.items()}
    result.passed = bool(result.checks) and all(result.checks.values())
    return result


def grade_left_turn(result, turned, faced_traffic, gave_way):
    return finish(result, faced_oncoming_traffic=faced_traffic, gave_way_to_traffic=gave_way, completed_the_turn=turned,
                  no_collision=not result.collisions, kept_a_safe_distance=result.min_gap is None or result.min_gap >= T2_SAFE_GAP,  # None: no car it could have avoided came near
                  no_red_light_violation=result.red_light_violations == 0)


def grade_blind_corner(result, heard_on_mqtt, stopped):
    return finish(result, safety_gate_fired=result.brake_at is not None, command_seen_on_mqtt=heard_on_mqtt,
                  car_stopped=stopped, no_collision=not result.collisions)


def traffic_schedule(rng, until=T2_LIMIT, first_wave=None, headway=None, gap=None):
    """The seconds at which oncoming cars set off: a steady wave, then gaps long enough to turn through and more waves."""
    first_wave = T2_FIRST_WAVE if first_wave is None else first_wave
    headway, gap = headway or T2_HEADWAY, gap or T2_GAP
    t, times = 0.0, []
    while t < first_wave:
        times.append(t)
        t += rng.uniform(*headway)
    while t < until:
        t += rng.uniform(*gap)
        for _ in range(rng.randint(*T2_WAVE_CARS)):
            times.append(t)
            t += rng.uniform(*headway)
    return times


def grade_light_change(result, turned, faced_traffic, gave_way, light_changed):
    grade_left_turn(result, turned, faced_traffic, gave_way)
    return finish(result, light_changed_while_waiting=light_changed, **result.checks)


def locations_array(locations):
    return np.array([(loc.x, loc.y) for loc in locations], dtype=float)


def paths_cross(mine, theirs, within=2.0):
    """(index along mine, index along theirs) where two paths of (x, y) points come closest, if they meet."""
    distance = np.hypot(mine[:, None, 0] - theirs[None, :, 0], mine[:, None, 1] - theirs[None, :, 1])
    i, j = np.unravel_index(int(np.argmin(distance)), distance.shape)
    return (int(i), int(j)) if distance[i, j] <= within else None


def launch_now(my_distance, my_speed, their_distance, their_speed, lead=0.0):
    """Should the other car go now, for both to reach the crossing together?"""
    return my_distance / my_speed <= their_distance / their_speed + lead


def straight_back(waypoint, metres, tolerance=3.0):
    """The waypoint this far back along a lane, or as far as it stays straight and clear of a junction. None if under 15 m."""
    yaw = waypoint.transform.rotation.yaw
    found = None
    for distance in range(5, int(metres) + 1, 5):
        behind = waypoint.previous(float(distance))
        if not behind or behind[0].is_junction or abs(yaw_difference(yaw, behind[0].transform.rotation.yaw)) > tolerance:
            break
        found = behind[0] if distance >= 15 else found
    return found


def along_and_across(ego, other):
    """(metres ahead, metres to the side) of other, seen from ego. Both are (x, y, ..., yaw)."""
    yaw = math.radians(ego[4])
    dx, dy = other[0] - ego[0], other[1] - ego[1]
    return dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)


LANE_PARALLEL = 30.0  # deg off the ML car's heading, either way, that counts as travelling along its road
OTHER_LANE = 2.5      # m to the side, centre to centre, that puts a parallel car in another lane


def counts_toward_gap(ego, other):
    """Whether other's distance from the ML car says anything about how it drove. A car in the next lane going either
    way along the road, or one coming up from behind, passes as close in normal driving, and the ML car didn't choose it."""
    ahead, across = along_and_across(ego, other)
    turn = abs((other[4] - ego[4] + 180.0) % 360.0 - 180.0)
    parallel = turn <= LANE_PARALLEL or turn >= 180.0 - LANE_PARALLEL
    return not (parallel and abs(across) >= OTHER_LANE) and not (turn <= LANE_PARALLEL and ahead < 0.0)


def follow_path(state, path, speed, look=6.0):
    """(steer, throttle, brake) for a simple scripted car following (x, y) points. state is (x, y, vx, vy, yaw)."""
    x, y, vx, vy, yaw = state
    remaining = [p for p in path if along_and_across(state, p)[0] > look * 0.5]
    target = remaining[0] if remaining else (x + math.cos(math.radians(yaw)) * look, y + math.sin(math.radians(yaw)) * look)
    # CARLA's y points right, so a target with a larger bearing means steering right, which is positive
    error = yaw_difference(yaw, math.degrees(math.atan2(target[1] - y, target[0] - x)))
    steer = max(-1.0, min(1.0, math.radians(error) * 1.2))
    gap = speed - math.hypot(vx, vy)
    if gap < -0.5:
        return steer, 0.0, min(1.0, -gap * 0.2)
    return steer, max(0.0, min(1.0, 0.4 + 0.3 * gap)), 0.0


def walk_lane(waypoint, metres):
    points, current = [waypoint], waypoint
    for _ in range(int(metres)):
        following = current.next(1.0)
        if not following:
            break
        current = following[0]
        points.append(current)
    return points


def stopping_distance(speed, braking=ASSUMED_BRAKING):
    return speed ** 2 / (2.0 * braking)


def alert_banner(cause, ttc, speed):
    shown = f"{ttc:.2f}s" if ttc is not None else "n/a"
    return (f"[ALERT - SAFETY GATE TRIGGERED] Cause: {cause} | TTC: {shown} | Actuation: MQTT Brake 100% | "
            f"Target Stopping Distance: {stopping_distance(speed):.1f}m")


def latency_ms(updated_at, received):
    """ms from the processor's timestamp on a brake command to now, or None if there's no usable timestamp."""
    try:
        return (received - datetime.fromisoformat(updated_at).timestamp()) * 1000.0
    except (TypeError, ValueError):
        return None


def yaw_class(yaw, mine):
    """A stop line's direction compared with mine: same, oncoming or side."""
    difference = abs(yaw_difference(mine, yaw))
    return "same" if difference < 45.0 else "oncoming" if difference > 135.0 else "side"


def shown(value, digits=2):
    return "n/a" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.{digits}f}"


def next_car_text(ttc):
    """How far the next car was when the ML car set off."""
    return "no car due" if ttc is None or not math.isfinite(ttc) else f"the next car {ttc:.1f} s away"


def live_table(title, t, pairs):
    """Lines of a two-column table of (name, value) pairs, with the title and time on top."""
    half = (len(pairs) + 1) // 2
    body = [f"| {a[0]:<15}{a[1]:>13} | {b[0]:<15}{b[1]:>13} |" for a, b in zip(pairs[:half], pairs[half:])]
    rule = "+" + "-" * (len(body[0]) - 2) + "+"
    clock = f"t={t:.1f} s"
    room = len(body[0]) - 4 - len(clock) - 1
    return [rule, f"| {title[:room]:<{room}} {clock} |", rule, *body, rule]


class TurnChooser:
    """Stands in for LaneDriver's random choice: takes this turn at the next junction, then goes straight."""

    def __init__(self, turn):
        self.turn = turn

    def choice(self, branches):
        changes = [yaw_difference(b.transform.rotation.yaw, _yaw_past_junction(b)) for b in branches]
        chosen = branches[pick_branch(changes, self.turn)]
        self.turn = STRAIGHT
        return chosen


def _yaw_past_junction(waypoint, longest=80):
    end = waypoint
    for step in range(longest):
        if step >= 15 and not end.is_junction:
            break
        following = end.next(1.0)
        if not following:
            break
        end = following[0]
    return end.transform.rotation.yaw


class Console:
    """Events scroll past as tagged lines, above one table of live values that redraws in place.

    The table only shows in a real terminal, and not with --quiet, so redirected output stays plain lines.
    Every tick's values are saved to output/live_telemetry.csv either way.
    """

    def __init__(self, quiet=False, stream=None):
        self.stream = stream or sys.stdout
        self.live = not quiet and self.stream.isatty()
        self.rows = []
        self.drawn = 0
        self.last_draw = 0.0
        self.lock = threading.Lock()  # MQTT events arrive on another thread
        os.system("")  # turns on ANSI escapes in a Windows console

    def _write(self, event=""):
        """Replaces the table in one write, so it doesn't flicker. Lines are cut to fit, since a wrapped line would break the redraw."""
        size = shutil.get_terminal_size()
        rows = self.rows if len(self.rows) < size.lines else []  # too short a terminal can't redraw it either
        erase = f"\033[{self.drawn}F\033[J" if self.drawn else ""
        self.stream.write(erase + event + "".join(row[:size.columns - 1] + "\n" for row in rows))
        self.stream.flush()
        self.drawn = len(rows)

    def plain(self, text):
        with self.lock:
            self._write(text + "\n")

    def line(self, tag, text):
        self.plain(f"\033[{TAGS.get(tag, '0')}m[{tag}]\033[0m {text}")

    def status(self, rows):
        """Shows these lines as the live table, at most five times a second."""
        if not self.live or time.monotonic() - self.last_draw < STATUS_INTERVAL:
            return
        with self.lock:
            self.last_draw = time.monotonic()
            self.rows = rows
            self._write()

    def end_status(self):
        with self.lock:
            self.rows = []
            self._write()


class MqttWatch:
    """Listens to the topic that a real vehicle's motors listen to, and notes each brake command it hears."""

    def __init__(self, console):
        self.console = console
        self.connected = False
        self.engaged = None
        self.heard_at = None
        self.latency = None
        self.client = None
        try:
            import paho.mqtt.client as mqtt

            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"test-suite-{os.getpid()}")
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=5)
            self.client.loop_start()
        except (ImportError, OSError) as error:
            console.line("MQTT / DITTO", f"Not listening: {error}")

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        self.connected = True
        client.subscribe(COMMAND_TOPIC, qos=1)
        self.console.line("MQTT / DITTO", f"Connected to {MQTT_HOST}:{MQTT_PORT}, listening to {COMMAND_TOPIC}")

    def _on_message(self, client, userdata, message):
        received = time.time()
        try:
            value = json.loads(message.payload).get("value", {})
        except ValueError:
            return
        if not isinstance(value, dict) or "emergency_brake" not in value:
            return
        engaged = value.get("emergency_brake") is True
        if engaged != self.engaged and not message.retain:
            self.heard_at = received if engaged else self.heard_at
            self.latency = latency_ms(value.get("updated_at"), received) if engaged else self.latency
            shown_latency = f"{self.latency:.0f} ms" if engaged and self.latency is not None else "n/a"
            self.console.line("MQTT / DITTO", f"Published on {COMMAND_TOPIC}: emergency_brake={engaged}, latency {shown_latency}")
        self.engaged = engaged

    def clear(self):
        self.heard_at = None
        self.latency = None

    def stop(self):
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()


class HudSender:
    """Tells the ML car's window which test is running and what has just happened."""

    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.title = "STARTING"
        self.events = deque(maxlen=4)

    def event(self, text):
        self.events.append(f"{datetime.now():%H:%M:%S} {text}")
        self.send()

    def send(self):
        message = {"title": self.title, "mode": CONTROL_MODE, "events": list(self.events)}
        try:
            self.socket.sendto(json.dumps(message).encode(), SUITE_ADDRESS)
        except OSError:
            pass


class RememberingSender(TelemetrySender):
    """Sends the model's values to the window, and keeps the newest for the logs."""

    def __init__(self):
        super().__init__()
        self.last = None

    def send(self, inputs, output, applied, model_driving, plan=None):
        self.last = {"target": float(output[1]), "applied": applied, "model_driving": bool(model_driving), "plan": plan}
        super().send(inputs, output, applied, model_driving, plan)


# ---------------------------------------------------------------------------
# CARLA
# ---------------------------------------------------------------------------
class Tick:
    """What one tick of a test sees."""

    def __init__(self, t, ego, speed, others, stop_gap, gate_on):
        self.t, self.ego, self.speed, self.others, self.stop_gap, self.gate_on = t, ego, speed, others, stop_gap, gate_on


class Suite:
    def __init__(self, carla, client, policy, console, seed=0, perception="truth", traffic=0):
        self.seed = seed  # picks the oncoming traffic in test 2
        self.perception = perception  # what the ML car is told about the actors around it: see roadside_link.py
        self.traffic = traffic  # background autopilot cars in every test, on top of each test's own
        self.variation = None  # a scenario_variation.Variation for the next run, or None for the fixed scenario
        self.planner = None    # a planner.Planner the ML car plans its speed with, or None to drive on the model's alone
        self.monitor = False   # whether the ML car drives with its own safety monitor, safety_monitor.py
        self.link_drop = None  # (from, to) s into each run that the roadside link is cut, to test what the car does then
        self.receiver = RoadsideReceiver() if perception in ROADSIDE_MODES else None
        self.carla = carla
        self.client = client
        self.world = client.get_world()
        self.map = self.world.get_map()
        self.policy = policy
        self.console = console
        self.session = requests.Session()
        self.traffic_manager = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
        self.traffic_manager.set_synchronous_mode(True)
        self.mqtt = MqttWatch(console)
        self.safety = SafetyReceiver()
        self.ack = TickAck()  # for a processor run with --lockstep
        self.hud = HudSender()
        self.world.wait_for_tick(10.0)
        # Only in sight, where the roof LiDAR is all the car has; in rsu and camera the roadside windows show more
        self.lights = RedLightWatch(carla, self.world)
        self.controlled = []  # (stop waypoint, light) of each lane at the junction
        for light in self.world.get_actors().filter("traffic.traffic_light"):
            for stop in light.get_stop_waypoints():
                location = stop.transform.location
                if math.hypot(location.x - JUNCTION[0], location.y - JUNCTION[1]) <= LIGHT_RADIUS:
                    self.controlled.append((stop, light))
        console.line("SCENARIO", f"Junction 189 has {len(self.controlled)} lanes with lights.")
        OUTPUT_DIR.mkdir(exist_ok=True)
        self.csv_file = open(OUTPUT_DIR / "live_telemetry.csv", "w", newline="", encoding="utf-8")  # noqa: SIM115
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(CSV_COLUMNS)
        self.actors = []
        self.sensors = []

    # --- set-up and teardown ---
    def believed(self, cars):
        """Which of the cars it's been told about the driver should act on."""
        return cars

    def clear(self):
        """Removes every car and their sensors, and releases the brake, so no test inherits anything."""
        for sensor in self.sensors:
            try:
                sensor.stop()
            except RuntimeError:
                pass
        self.client.apply_batch_sync([self.carla.command.DestroyActor(a) for a in self.sensors])
        self.sensors = []
        cars = [*self.world.get_actors().filter("vehicle.*"), *self.world.get_actors().filter("walker.pedestrian.*")]
        self.client.apply_batch_sync([self.carla.command.DestroyActor(a) for a in cars])
        self.actors = []
        for _ in range(3):
            self.world.wait_for_tick(TICK_TIMEOUT)
        ghosts = len(self.world.get_actors().filter("vehicle.*"))
        if ghosts:
            self.console.line("SCENARIO", f"WARNING: {ghosts} cars survived the clear-up.")
        try:
            release_gate(self.session)
        except requests.RequestException as error:
            self.console.line("MQTT / DITTO", f"Could not release the brake: {error}")
        self.mqtt.clear()
        return ghosts

    def set_lights(self, mine_yaw, same, oncoming, side):
        states = {"same": same, "oncoming": oncoming, "side": side}
        for stop, light in self.controlled:
            light.set_state(states[yaw_class(stop.transform.rotation.yaw, mine_yaw)])
            light.freeze(True)

    def release_lights(self):
        for _, light in self.controlled:
            light.freeze(False)

    def lane_with_turn(self, turn):
        """A stop line and route that really offer this turn."""
        for stop, _ in sorted(self.controlled, key=lambda c: (round(c[0].transform.rotation.yaw / 10.0), c[0].lane_id)):
            behind = stop.previous(APPROACH)
            if not behind or behind[0].is_junction:
                continue
            lane = LaneDriver(stop, TurnChooser(turn))
            if turn == STRAIGHT:
                if not lane.turns:
                    return stop
            elif turn in lane.turns.values():
                return stop
        raise RuntimeError(f"No lane at junction 189 offers turn {turn}.")

    def spawn_car(self, blueprint, waypoint, role="autopilot", lift=0.5, paint=None):
        bp = self.world.get_blueprint_library().find(blueprint)
        bp.set_attribute("role_name", role)
        if paint and bp.has_attribute("color"):
            bp.set_attribute("color", paint)
        tf = waypoint.transform
        place = self.carla.Transform(self.carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z + lift), tf.rotation)
        car = self.world.try_spawn_actor(bp, place)
        if car is None:
            raise RuntimeError(f"Couldn't place {blueprint} at x={tf.location.x:.0f} y={tf.location.y:.0f}.")
        self.actors.append(car)
        return car

    def find_waypoint(self, x, y):
        return self.map.get_waypoint(self.carla.Location(x=float(x), y=float(y), z=0.0), project_to_road=True,
                                     lane_type=self.carla.LaneType.Driving)

    def straight_path(self, waypoint, metres=120, every=5, turn=STRAIGHT):
        """Locations through the junction on this turn, straight on unless told otherwise, for the Traffic Manager to follow."""
        chooser, current, points = TurnChooser(turn), waypoint, []
        for step in range(metres):
            following = current.next(1.0)
            if not following:
                break
            current = chooser.choice(following) if len(following) > 1 else following[0]
            if step % every == 0:
                points.append(current.transform.location)
        return points

    def spawn_traffic(self, rng, away_from):
        """self.traffic autopilot cars at the map's spawn points, none near the ML car's start. Vehicles only: AI-driven
        people are invisible to LiDAR, and their controllers mustn't start with LiDARs running. A taken point is skipped."""
        command = self.carla.command
        points = [p for p in self.map.get_spawn_points() if p.location.distance(away_from) > TRAFFIC_CLEARANCE]
        rng.shuffle(points)
        blueprints = [bp for bp in self.world.get_blueprint_library().filter("vehicle.*") if bp.get_attribute("number_of_wheels").as_int() == 4]
        batch = []
        for place in points[:self.traffic]:
            bp = rng.choice(blueprints)
            bp.set_attribute("role_name", "background")
            batch.append(command.SpawnActor(bp, place).then(command.SetAutopilot(command.FutureActor, True, TRAFFIC_MANAGER_PORT)))
        spawned = sum(not response.error for response in self.client.apply_batch_sync(batch))
        self.console.line("SCENARIO", f"{spawned} background cars on autopilot.")
        return spawned

    def autopilot(self, car, speed, path=None, ignore_rules=False):
        self.traffic_manager.auto_lane_change(car, False)
        car.set_autopilot(True, TRAFFIC_MANAGER_PORT)
        self.traffic_manager.set_desired_speed(car, speed * 3.6)
        self.traffic_manager.ignore_lights_percentage(car, 100.0 if ignore_rules else 0.0)
        self.traffic_manager.ignore_vehicles_percentage(car, 100.0 if ignore_rules else 0.0)
        if path:
            self.traffic_manager.set_path(car, path)

    # --- running a test ---
    def run_stage(self, test, name, title, stop, turn, limit, on_tick, build=None, start=None):
        """Places the ML car at start, or before stop, on a route with this turn, runs on_tick until it says it's done."""
        carla = self.carla
        result = StageResult(test, name)
        self.hud.title = title
        self.hud.events.clear()
        self.console.line("SCENARIO", f"=== {title} ===")
        self.hud.event(f"{name}: started")
        self.clear()
        rng = random.Random(test)
        start = start or stop.previous(APPROACH)[0]
        ml = self.spawn_car("vehicle.tesla.model3", start, role="vehicle_ml", paint=ML_PAINT)
        telemetry = RememberingSender()
        telemetry.source = self.perception
        driver = ModelDriver(self.policy, LaneDriver(start, TurnChooser(turn)), self.find_waypoint, rng, telemetry)
        driver.planner = self.planner
        if self.monitor:
            from safety_monitor import SafetyMonitor

            driver.monitor = SafetyMonitor()
        hits = []
        sensor = self.world.spawn_actor(self.world.get_blueprint_library().find("sensor.other.collision"), carla.Transform(), attach_to=ml)
        sensor.listen(lambda event: hits.append((event.other_actor.id, event.other_actor.type_id, event.other_actor.attributes.get("role_name", ""))))
        self.sensors.append(sensor)
        if build is not None:
            build(ml)
        if self.traffic:
            result.traffic = self.spawn_traffic(random.Random(self.seed * 100 + test), start.transform.location)
        view = make_view(self.perception, carla, self.world, ml, 0.1, self.receiver)
        for _ in range(SETTLE_TICKS):
            self.world.wait_for_tick(TICK_TIMEOUT)

        contacts = {}
        driven = ([], [], [], [])
        lights = self.lights
        lights.runs, lights.suspects, lights.seen = [], {}, {}
        started = now = frame = None
        last_position = None
        was_braking = False
        held_since = None
        polls = []
        closest = None
        min_ttc = math.inf
        states, ttcs, engagements, entered = [], [], [], None
        self.mqtt.clear()
        done = False
        # Every tick, not wait_for_tick's next one: in lockstep the processor ticks the moment it hears the ack, and a tick
        # that arrives before wait_for_tick is called is missed, so both sides waited out the gate's timeout
        ticks = queue.Queue()
        listening = self.world.on_tick(ticks.put)
        try:
            while not done:
                try:
                    snapshot = ticks.get(timeout=TICK_TIMEOUT)
                except queue.Empty:
                    raise RuntimeError(f"no tick from CARLA within {TICK_TIMEOUT} s") from None
                while not ticks.empty():  # a driver slower than the clock acts on the newest, counting the rest as skipped
                    snapshot = ticks.get_nowait()
                if started is not None:
                    result.skipped_ticks += max(0, snapshot.frame - frame - 1)
                frame = snapshot.frame
                now = snapshot.timestamp.elapsed_seconds
                started = now if started is None else started
                t = now - started
                ego = actor_state(ml)
                speed = math.hypot(ego[2], ego[3])
                others = {a.id: actor_state(a) for a in self.world.get_actors().filter("vehicle.*") if a.id != ml.id}
                walkers = {w.id: (w.get_location().x, w.get_location().y, w.get_velocity().x, w.get_velocity().y)
                           for w in self.world.get_actors().filter("walker.pedestrian.*")}
                not_green = lights.not_green()
                if self.link_drop and self.link_drop[0] <= t < self.link_drop[1] and hasattr(view, "own"):
                    seen_cars, seen_walkers, fresh = view.own.see(ego, others, walkers, now)  # the roadside unit's tracks never arrive
                    link_up = False
                else:
                    seen_cars, seen_walkers, fresh = view.see(ego, others, walkers, now)
                    link_up = getattr(view, "link_up", None)  # None with no roadside unit
                seen_cars = self.believed(seen_cars)
                telemetry.fresh = fresh

                asked = time.perf_counter()
                try:
                    gate = self.session.get(GATE_URL, timeout=0.5).json()
                    polls.append((time.perf_counter() - asked) * 1000.0)
                except (requests.RequestException, ValueError):
                    gate = {}
                # In the simulation no config obeys the roadside gate, so all three drive by the same rules; it's still recorded
                released = fresh
                steer, throttle, brake = driver.controls(ego, not_green, seen_cars, seen_walkers, released, link_up)
                ml.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
                for series, value in zip(driven, (steer, throttle, brake, speed)):
                    series.append(value)

                safety = self.safety.poll() or {}
                ttc = safety.get("ttc")
                if ttc is not None:
                    min_ttc = min(min_ttc, ttc)
                gap, nearest = min(((rectangles_gap(outline(ego[0], ego[1], ego[4]), outline(o[0], o[1], o[4])), o)
                                    for o in others.values() if math.hypot(o[0] - ego[0], o[1] - ego[1]) < 15.0 and counts_toward_gap(ego, o)),
                                   default=(None, None), key=lambda pair: math.inf if pair[0] is None else pair[0])
                if gap is not None and (result.min_gap is None or gap < result.min_gap):
                    result.min_gap = gap
                    ahead, across = along_and_across(ego, nearest)
                    turn = (nearest[4] - ego[4] + 180.0) % 360.0 - 180.0
                    closest = (f"closest car {gap:.1f} m at t={t:.1f} s: {ahead:+.1f} m ahead, {across:+.1f} m right, "
                               f"heading {turn:+.0f} deg, {math.hypot(nearest[2], nearest[3]):.1f} m/s, ML car {speed:.1f} m/s")
                result.worst_off_lane = max(result.worst_off_lane, driver.off_lane)
                if last_position is not None:
                    result.distance += math.hypot(ego[0] - last_position[0], ego[1] - last_position[1])
                last_position = ego[:2]
                lights.update({ml.id: ego}, t, ml.id, "SCENARIO")
                stop_gap = red_light_gap(driver.lane.path_ahead(RED_LOOK), not_green)

                states.append((t, ego, list(others.values()), list(walkers.values())))
                ttcs.append(true_ttc(*states[-1][1:]))
                here = self.find_waypoint(ego[0], ego[1])
                inside = here.is_junction and here.get_junction().id == JUNCTION_ID
                if inside and entered is None:
                    entered = t
                elif not inside and entered is not None and result.clearance_s is None:
                    result.clearance_s = round(t - entered, 2)

                while hits:
                    other_id, kind, role = hits.pop()
                    if is_new_contact(contacts, other_id, t):
                        result.collisions.append({"t": round(t, 1), "with": kind, "role": role, "speed": round(speed, 1)})
                        self.console.line("ALERT", f"COLLISION with {kind} at {speed:.1f} m/s, t={t:.1f} s")
                        self.hud.event(f"COLLISION with {kind.split('.')[-1]} at {speed:.1f} m/s")

                engaged = gate.get("emergency_brake") is True
                if engaged and not was_braking:
                    result.brake_events += 1
                    engagements.append(t)
                    reason = gate.get("reason") or ""
                    if result.brake_at is None:
                        result.brake_at, result.brake_ttc = round(t, 2), gate.get("ttc")
                    banner = alert_banner(name, gate.get("ttc"), speed)
                    self.console.plain(f"\033[1;91m{banner} | {reason}\033[0m")
                    self.hud.event(f"SAFETY GATE: {reason}")
                elif was_braking and not engaged:
                    self.hud.event("Safety brake released")
                was_braking = engaged
                held_since = (held_since if held_since is not None else t) if engaged else None

                state = telemetry.last or {"target": math.nan, "model_driving": False}
                error = state["target"] - speed
                self.console.status(live_table(title, t, (
                    ("Target speed", f"{shown(state['target'] * 3.6, 1)} km/h"), ("Actual speed", f"{speed * 3.6:.1f} km/h"), ("Steering", f"{steer:+.2f}"),
                    ("Throttle", f"{throttle * 100:.0f} %"), ("Brake pedal", f"{brake * 100:.0f} %"), ("Off lane", f"{driver.off_lane:.2f} m"),
                    ("Min TTC", f"{shown(ttc)} s"), ("Path TTC", f"{shown(safety.get('path_ttc'))} s"), ("PET", f"{shown(safety.get('pet'))} s"),
                    ("Safety brake", "ON" if engaged else "off"), ("Closest car", f"{shown(gap, 1)} m"), ("MQTT", "connected" if self.mqtt.connected else "NOT CONNECTED"))))
                self.csv.writerow((f"{time.time():.3f}", test, f"{t:.2f}", f"{ego[0]:.2f}", f"{ego[1]:.2f}", f"{ego[4]:.1f}", f"{speed:.3f}",
                                   f"{state['target']:.3f}", f"{steer:.4f}", f"{throttle:.3f}", f"{brake:.3f}", f"{error:.3f}",
                                   int(state["model_driving"]), int(engaged), shown(ttc, 3), shown(safety.get("path_ttc"), 3),
                                   shown(safety.get("pet"), 3), f"{min(stop_gap, STOP_LINE_RANGE):.1f}", shown(gap, 2), f"{driver.off_lane:.2f}"))
                self.hud.send()

                tick = Tick(t, ego, speed, others, stop_gap, engaged)
                tick.walkers, tick.throttle, tick.brake = walkers, throttle, brake
                tick.held_for = 0.0 if held_since is None else t - held_since
                tick.turn_yaw = ego[4]
                done = on_tick(tick, result, driver)
                self.ack.send(frame)  # after on_tick too, which moves the test's own actors
                if t >= limit:
                    result.notes.append(f"ran out of time at {limit:.0f} s")
                    self.hud.event("Ran out of time")
                    done = True
        finally:
            self.world.remove_on_tick(listening)
            self.ack.release()  # first, so the processor doesn't wait out the clean-up below
            view.close()
            result.duration = round(now - started, 1) if started is not None else 0.0
            if closest is not None:
                result.notes.append(closest)
            if driver.monitor is not None:
                result.monitor = {rule: {"ticks": n, "times": driver.monitor.times[rule]} for rule, n in driver.monitor.ticks.items()}
            result.red_light_violations = len(lights.runs)
            result.min_ttc = None if math.isinf(min_ttc) else round(min_ttc, 3)
            lowest = min(ttcs, default=math.inf)
            result.true_min_ttc = None if math.isinf(lowest) else round(lowest, 3)
            result.false_brakes = false_brakes(engagements, states, [c["t"] for c in result.collisions])
            result.mqtt_latency_ms = self.mqtt.latency
            result.gate_poll_ms = float(np.median(polls)) if polls else None
            result.smoothness = {k: round(v, 3) for k, v in smoothness(*driven).items()}
            self.csv_file.flush()
        return result


    def left_turn_lanes(self):
        """(stop line with a left turn, the oncoming lanes), for an approach with a long straight view of them: the
        variation's, counted round, or the first."""
        found = []
        for stop, _ in sorted(self.controlled, key=lambda c: (round(c[0].transform.rotation.yaw / 10.0), c[0].lane_id)):
            behind = stop.previous(T2_START_BACK)
            if not behind or behind[0].is_junction or LEFT not in LaneDriver(stop, TurnChooser(LEFT)).turns.values():
                continue
            yaw = stop.transform.rotation.yaw
            ahead = stop.transform.get_forward_vector()
            here = stop.transform.location
            oncoming = [o for o, _ in self.controlled if yaw_class(o.transform.rotation.yaw, yaw) == "oncoming"
                        and (o.transform.location.x - here.x) * ahead.x + (o.transform.location.y - here.y) * ahead.y > 0.0]
            far_back = [o.previous(T2_FEED_FROM) for o in oncoming]
            if oncoming and all(b and abs(yaw_difference(o.transform.rotation.yaw, b[0].transform.rotation.yaw)) < 3.0 for o, b in zip(oncoming, far_back)):
                found.append((stop, oncoming))
        if not found:
            raise RuntimeError("No left turn at junction 189 has a long straight view of oncoming traffic.")
        return found[(self.variation.approach if self.variation else 0) % len(found)]


    def test_left_turn(self, light_change=False):
        stop, oncoming = self.left_turn_lanes()
        yaw = stop.transform.rotation.yaw
        red, green, yellow = self.carla.TrafficLightState.Red, self.carla.TrafficLightState.Green, self.carla.TrafficLightState.Yellow
        self.set_lights(yaw, green, green, red)
        rng = random.Random(self.seed)
        v = self.variation
        speed, headway = (v.oncoming_speed, v.headway) if v else (T2_ONCOMING, T2_HEADWAY)
        schedule = deque(traffic_schedule(rng, first_wave=v.first_wave, headway=v.headway, gap=v.gap) if v else traffic_schedule(rng))
        starts = [o.previous(T2_FEED_FROM)[0] for o in oncoming]
        cars = []
        heading = math.radians(yaw)
        line = stop.transform.location
        state = {"turned": False, "turned_at": 0.0, "start_yaw": None, "closest": math.inf, "waited": 0.0, "released": 0,
                 "stopped": False, "go_ttc": None, "changed_at": None, "red": False, "stopping": set()}
        self.console.line("SCENARIO", f"Oncoming traffic seed {self.seed}, {speed:.0f} m/s")

        def launch(start=None):
            start = start or rng.choice(starts)
            bp = self.world.get_blueprint_library().find("vehicle.audi.a2")
            bp.set_attribute("role_name", "autopilot")
            tf = start.transform
            car = self.world.try_spawn_actor(bp, self.carla.Transform(self.carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z + 0.5), tf.rotation))
            if car is None:
                return False
            car.enable_constant_velocity(self.carla.Vector3D(x=speed))
            cars.append(car)
            self.actors.append(car)
            return True

        def build(ml):
            # The road already has traffic on it when the ML car sets off
            metres = T2_FEED_FROM
            while metres > T2_PREFILL_TO:
                launch(rng.choice(oncoming).previous(metres)[0])
                metres -= speed * rng.uniform(*headway)

        def on_tick(tick, result, driver):
            state["start_yaw"] = tick.ego[4] if state["start_yaw"] is None else state["start_yaw"]
            if light_change and state["changed_at"] is None and state["waited"] >= T5_WAITED:
                state["changed_at"] = tick.t
                schedule.clear()
                self.set_lights(yaw, yellow, yellow, red)
                for car in cars:
                    room = self._metres_to_lines(tick.others.get(car.id), oncoming)
                    if room is not None and room > T2_ONCOMING ** 2 / (2.0 * T5_CAN_STOP):
                        state["stopping"].add(car.id)
                        car.disable_constant_velocity()
                self.console.line("SCENARIO", f"Light turns YELLOW with the ML car inside the junction, t={tick.t:.1f} s")
                self.hud.event("Light turns yellow: car may finish its turn")
            if state["changed_at"] is not None and not state["red"] and tick.t - state["changed_at"] >= T5_YELLOW:
                state["red"] = True
                self.set_lights(yaw, red, red, red)
                self.console.line("SCENARIO", f"Light turns RED, t={tick.t:.1f} s")
            for car in cars:
                if car.id in state["stopping"] and car.id in tick.others:
                    other = tick.others[car.id]
                    room = self._metres_to_lines(other, oncoming)
                    needed = math.hypot(other[2], other[3]) ** 2 / (2.0 * max(room or 0.0, 0.5))
                    car.apply_control(self.carla.VehicleControl(throttle=0.0, brake=min(1.0, needed / 8.0)))
            if schedule and tick.t >= schedule[0] and launch():
                schedule.popleft()
            past = (tick.ego[0] - line.x) * math.cos(heading) + (tick.ego[1] - line.y) * math.sin(heading)
            for car in list(cars):
                if car.id not in tick.others:
                    continue
                x, y = tick.others[car.id][:2]
                if past > -5.0:  # how close oncoming traffic really came once the ML car reached the junction
                    state["closest"] = min(state["closest"], math.hypot(x - tick.ego[0], y - tick.ego[1]))
                beyond = (x - JUNCTION[0]) * -math.cos(heading) + (y - JUNCTION[1]) * -math.sin(heading)
                if beyond > T2_REMOVE_BEYOND:
                    cars.remove(car)
                    car.destroy()
            ttc = crossing_ttc(driver.lane.path_ahead(LOOK_AHEAD), [tick.others[c.id] for c in cars if c.id in tick.others], self.lights.not_green())
            if tick.speed < MOVING:
                state["stopped"] = True
                if ttc < T2_TOO_SOON and past > -5.0:
                    state["waited"] += 0.1
            elif state["stopped"]:
                state["stopped"] = False
                state["go_ttc"] = ttc
                if past > -5.0:  # not when it first pulls away, far from the turn
                    self.hud.event(f"Set off with {next_car_text(ttc)}")
            clear = all(math.hypot(o[0] - tick.ego[0], o[1] - tick.ego[1]) > T2_CLEAR for o in tick.others.values())
            if tick.gate_on and tick.held_for >= BRAKE_HOLD and clear:
                release_gate(self.session)
                state["released"] += 1
                self.console.line("SCENARIO", f"Gap is clear: brake released, t={tick.t:.1f} s")
            change = abs(yaw_difference(state["start_yaw"], tick.ego[4]))
            if change >= T2_TURNED and not state["turned"]:
                state["turned"] = True
                state["turned_at"] = tick.t
                self.console.line("SCENARIO", f"Turned left {change:.0f} degrees, t={tick.t:.1f} s")
                self.hud.event("Left turn completed")
            return state["turned"] and tick.t - state["turned_at"] > 5.0

        number, name, title = ((5, "Left turn as the light changes", "[TEST 5/5] LEFT TURN AS THE LIGHT TURNS YELLOW") if light_change
                               else (2, "Unprotected left turn and gap acceptance", "[TEST 2/5] UNPROTECTED LEFT TURN & GAP ACCEPTANCE"))
        result = self.run_stage(number, name, title, stop, LEFT, T2_LIMIT, on_tick,
                                build, start=stop.previous(T2_START_BACK)[0])
        result.notes.append(f"seed {self.seed}; waited {state['waited']:.1f} s with a car due; set off with {next_car_text(state['go_ttc'])}; "
                            f"closest oncoming car {shown(state['closest'], 1)} m; {result.brake_events} safety brakes")
        gave_way = state["waited"] >= T2_WAITED
        if light_change:
            return grade_light_change(result, state["turned"], state["closest"] <= T2_FACED, gave_way, state["changed_at"] is not None)
        return grade_left_turn(result, state["turned"], state["closest"] <= T2_FACED, gave_way)

    def _metres_to_lines(self, car, lines):
        """Metres ahead of this (x, y, vx, vy, yaw) car to the nearest of these stop waypoints in its lane; None if it's past them."""
        if car is None:
            return None
        ahead = [along_and_across(car, (w.transform.location.x, w.transform.location.y)) for w in lines]
        ahead = [a for a, side in ahead if a > 0.0 and abs(side) < 2.0]
        return min(ahead) if ahead else None

    def test_blind_corner(self, lead=T3_LEAD):
        stop = self.lane_with_turn(STRAIGHT)
        yaw = stop.transform.rotation.yaw
        red, green = self.carla.TrafficLightState.Red, self.carla.TrafficLightState.Green
        self.set_lights(yaw, green, red, red)
        my_path = locations_array(self.straight_path(stop.previous(APPROACH)[0], every=1))
        side, runner_start, crossing = None, None, None
        for candidate, _ in self.controlled:
            if yaw_class(candidate.transform.rotation.yaw, yaw) != "side" or not candidate.previous(T3_RUNUP):
                continue
            start = candidate.previous(T3_RUNUP)[0]
            if start.is_junction:
                continue
            their_path = self.straight_path(start, every=1)
            found = paths_cross(my_path, locations_array(their_path))
            if found is not None:
                side, runner_start, crossing = candidate, start, found
                break
        if side is None:
            raise RuntimeError("No side road's straight route crosses the ML car's route.")
        my_index, their_index = crossing
        state = {"runner": None, "launched_at": None, "stopped_at": None, "heard": False, "closest_runner": math.inf}
        notes = []

        def build(ml):
            truck_bp = self.world.get_blueprint_library().find(TRUCK)
            truck_bp.set_attribute("role_name", "truck")
            truck = None
            for back in T3_TRUCK_BACK:
                for across in T3_TRUCK_SIDE:
                    corner = side.previous(back)[0]
                    right = corner.transform.get_right_vector()
                    location = corner.transform.location
                    place = self.carla.Transform(self.carla.Location(x=location.x + right.x * across, y=location.y + right.y * across, z=location.z + 1.5), corner.transform.rotation)
                    truck = truck or self.world.try_spawn_actor(truck_bp, place)
            if truck is None:
                notes.append("the truck couldn't be placed at the corner")
            else:
                truck.set_simulate_physics(False)
                self.actors.append(truck)
                self.hud.event("Box truck parked at the corner")
            state["runner"] = self.spawn_car("vehicle.dodge.charger_2020", runner_start)
            state["runner"].apply_control(self.carla.VehicleControl(hand_brake=True))

        def on_tick(tick, result, driver):
            runner = state["runner"]
            nearest = int(np.argmin(np.hypot(my_path[:, 0] - tick.ego[0], my_path[:, 1] - tick.ego[1])))
            if state["launched_at"] is None:
                if launch_now(my_index - nearest, max(tick.speed, 4.0), their_index, T3_SPEED, lead):
                    state["launched_at"] = tick.t
                    # Autopilot won't hold a set speed through a junction, so this speed is enforced
                    runner.enable_constant_velocity(self.carla.Vector3D(x=T3_SPEED))
                    self.console.line("SCENARIO", f"Speeding car launched through its red light, ML car {my_index - nearest} m from the crossing, t={tick.t:.1f} s")
                    self.hud.event("Speeding car launched through its red light")
            elif runner.id in tick.others:
                other = tick.others[runner.id]
                state["closest_runner"] = min(state["closest_runner"], rectangles_gap(outline(tick.ego[0], tick.ego[1], tick.ego[4]), outline(other[0], other[1], other[4])))
            if self.mqtt.heard_at is not None:
                state["heard"] = True
            if tick.gate_on and tick.speed < 0.1:
                if state["stopped_at"] is None:
                    state["stopped_at"] = tick.t
                    if my_index > nearest:
                        result.stopping_margin = round(float(my_index - nearest), 2)
                    self.hud.event(f"Stopped {my_index - nearest} m before the crossing")
                if tick.t - state["stopped_at"] >= T3_REST:
                    return True
            return bool(state["launched_at"] is not None and tick.t - state["launched_at"] > T3_AFTER and not tick.gate_on)

        result = self.run_stage(3, "Blind corner and MQTT safety gate", "[TEST 3/5] BLIND CORNER HAZARD - MQTT SAFETY GATE ACTIVE", stop, STRAIGHT, T3_LIMIT, on_tick, build)
        result.notes += notes
        result.notes.append(f"closest the speeding car came: {shown(state['closest_runner'], 1)} m")
        result.hazard_gap = None if math.isinf(state["closest_runner"]) else state["closest_runner"]
        return grade_blind_corner(result, state["heard"], state["stopped_at"] is not None)


    def close(self):
        self.clear()
        self.release_lights()
        self.mqtt.stop()
        self.safety.close()
        self.ack.close()
        if self.receiver is not None:
            self.receiver.close()
        self.csv_file.close()


# ---------------------------------------------------------------------------
# Running the suite
# ---------------------------------------------------------------------------


