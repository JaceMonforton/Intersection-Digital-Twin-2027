"""
The five-test demonstration around Town 10's junction 189, with the ML car driven by the trained model.

    TEST 1  Stopping at a red light, waiting, pulling away on green, holding speed
    TEST 2  Turning left across oncoming traffic
    TEST 3  A speeding car from behind a truck runs its red light: the safety gate must stop the ML car
    TEST 4  A car in the next lane cuts in front of the ML car
    TEST 5  The light turns yellow while the ML car waits inside the junction to turn left: it finishes the
            turn once no oncoming car is still coming

Each test runs once and is scored on what was measured, with no retries, so a failure stays a failure.
Other cars are CARLA's autopilot. Between tests the brake is released and every car is removed. The brake
release here stands in for an operator and only ever happens inside CARLA.

If the simulation isn't running this starts it through run_simulation.py.
    python scripts/run_professor_test_suite.py
    python scripts/run_professor_test_suite.py --only 3 --quiet
"""
import argparse
import csv
import json
import math
import os
import random
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from central_gpu_processor import ML_PAINT, TRAFFIC_MANAGER_PORT
from driving_features import STOP_LINE_RANGE
from left_turn_scenario import outline, rectangles_gap
from ml_driver import DrivingPolicy as Policy
from ml_driver import (
    ModelDriver,
    actor_state,
    brake_is_off,
    crossing_ttc,
    is_new_contact,
    smoothness,
)
from model_panel import SUITE_ADDRESS, SafetyReceiver, TelemetrySender
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

T1_WAIT = 6.0                 # s to wait at the red
T1_LIMIT = 90.0
T1_CRUISE = 4.5               # m/s that counts as back up to speed
T1_RECOVER = 20.0             # s to get back up to speed after the green
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
T4_LIMIT = 60.0
T4_STRAIGHT = 70.0            # m of straight two-lane road needed
T4_REAL_CUT_AHEAD = 20.0      # m ahead of the ML car that counts as cutting in front of it
T4_IN_LANE = 1.2              # m to the side of the ML car that counts as in its lane
T4_SPEED = 7.0                # m/s, the scripted cut-in car before it moves over
T4_THEN = 3.0                 # m/s, the cut-in car after it has moved over, so the ML car must slow
T4_CUT_AHEAD = 6.0            # m ahead of the ML car, centre to centre, that it moves over
T4_MERGE = 8.0                # m further on that its path joins the ML car's lane
T4_RUN = 25.0                 # s after the cut-in before scoring
OFF_LANE_LIMIT = 1.0          # m; normal driving stays within about 0.6 m

TAGS = {"SCENARIO": "96", "MODEL": "92", "SAFETY GATE": "93", "MQTT / DITTO": "94", "ALERT": "91"}
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
    stopping_margin: float = None
    min_gap: float = None
    min_ttc: float = None
    brake_events: int = 0
    brake_at: float = None
    brake_ttc: float = None
    mqtt_latency_ms: float = None
    gate_poll_ms: float = None
    worst_off_lane: float = 0.0
    distance: float = 0.0
    duration: float = 0.0
    smoothness: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def finish(result, **checks):
    """Fills in the checks and the result. A test passes only if every check does."""
    result.checks = {name: bool(value) for name, value in checks.items()}
    result.passed = bool(result.checks) and all(result.checks.values())
    return result


def grade_lights(result, stopped, waited, went_on_green, reached_speed):
    return finish(result, stopped_before_the_line=stopped and (result.stopping_margin or -1.0) >= 0.0, waited_at_red=waited,
                  no_red_light_violation=result.red_light_violations == 0, pulled_away_on_green=went_on_green,
                  back_up_to_speed=reached_speed, no_collision=not result.collisions)


def grade_left_turn(result, turned, faced_traffic, gave_way):
    return finish(result, faced_oncoming_traffic=faced_traffic, gave_way_to_traffic=gave_way, completed_the_turn=turned,
                  no_collision=not result.collisions, kept_a_safe_distance=result.min_gap is not None and result.min_gap >= T2_SAFE_GAP,
                  safety_gate_stayed_off=result.brake_events == 0, no_red_light_violation=result.red_light_violations == 0)


def grade_blind_corner(result, heard_on_mqtt, stopped):
    return finish(result, safety_gate_fired=result.brake_at is not None, command_seen_on_mqtt=heard_on_mqtt,
                  car_stopped=stopped, no_collision=not result.collisions)


def grade_cut_in(result, cut_in_happened):
    return finish(result, cut_in_happened=cut_in_happened, no_collision=not result.collisions,
                  stayed_in_lane=result.worst_off_lane <= OFF_LANE_LIMIT)


def traffic_schedule(rng, until=T2_LIMIT):
    """The seconds at which oncoming cars set off: a steady wave, then gaps long enough to turn through and more waves."""
    t, times = 0.0, []
    while t < T2_FIRST_WAVE:
        times.append(t)
        t += rng.uniform(*T2_HEADWAY)
    while t < until:
        t += rng.uniform(*T2_GAP)
        for _ in range(rng.randint(*T2_WAVE_CARS)):
            times.append(t)
            t += rng.uniform(*T2_HEADWAY)
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


def matrix_rows(results):
    rows = []
    for r in results:
        rows.append(f"| {r['test']}. {r['name']} | {'PASS' if r['passed'] else '**FAIL**'} | {len(r['collisions'])} | "
                    f"{r['red_light_violations']} | {shown(r['stopping_margin'])} m | {shown(r['min_gap'])} m | "
                    f"{shown(r['mqtt_latency_ms'], 0)} ms | {shown(r['gate_poll_ms'], 0)} ms |")
    return rows


def smoothness_rows(smooth):
    names = {"steer_change_rms": "Steering change per tick (rms)", "pedal_switches_per_min": "Throttle/brake switches per minute",
             "jerk_p95": "Jerk, 95th percentile (m/s3)", "harsh_stops_per_min": "Decelerations over 6 m/s2 per minute",
             "hardest_brake": "Hardest one-tick deceleration (m/s2)"}
    return [f"| {label} | {shown(smooth['model'].get(key), 3)} | {shown(smooth['autopilot'].get(key), 3)} |" for key, label in names.items()]


def round_rows(rounds, limit=0.11):
    rows = []
    for name, data in rounds.items():
        collisions, reds = data.get("collisions_per_km"), data.get("red_lights_per_km")
        met = collisions is not None and reds is not None and collisions <= limit and reds <= limit
        rows.append(f"| {name} | {data.get('what', '')} | {shown(data.get('km'), 1)} | {shown(collisions)} | {shown(reds)} | "
                    f"{'yes' if met else 'no'} |")
    return rows


def write_summary(path, results, rounds, smooth_note, smooth=None):
    passed = sum(r["passed"] for r in results)
    lines = [
        "# Professor Demo Summary", "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. Every figure is measured. Each test ran once, with no retries.", "",
        f"## Scenario tests: {passed} of {len(results)} passed", "",
        "| Test | Result | Collisions | Red light violations | Stopping margin | Closest gap | MQTT latency | Gate read latency |",
        "|---|---|---|---|---|---|---|---|",
        *matrix_rows(results), "",
    ]
    for r in results:
        failed = [name.replace("_", " ") for name, ok in r["checks"].items() if not ok]
        if failed or r["notes"]:
            lines.append(f"- **Test {r['test']}:** " + "; ".join(([f"failed: {', '.join(failed)}"] if failed else []) + r["notes"]))
    lines += ["", "## Driving in traffic, against the 0.11 per km target", "",
              ("Measured with the safety brake ignored, so these are the model's own mistakes. Autopilot's rate in the same traffic "
               "was about 0.09 per km, and the target is 1.25 times that."), "",
              "| Round | What changed | km | Collisions per km | Red lights per km | Within target so far |", "|---|---|---|---|---|---|",
              *round_rows(rounds), "",
              ("The last two rounds use the same general driver, so together that's 1 collision in 11.2 km, in which the ML car was stopped when "
               "an Audi hit it on a drive with no left turns; the cause wasn't examined. That's too little driving to show the target is met, "
               "and the approved 10-run pass test against autopilot hasn't been run on this model."), "",
              *(["## Smoothness over those drives", "", "| Measure | Model | Autopilot |", "|---|---|---|", *smoothness_rows(smooth), ""] if smooth else []),
              "## How to read this", "",
              "- The model chooses steering and target speed. A fixed PID controller works the pedals, and it isn't AI.",
              "- It learned by copying CARLA's autopilot, and sees positions from CARLA rather than cameras or LiDAR.",
              "- Top speed is about 20 km/h, because that's how autopilot drove in the recordings.",
              "- The brake release between hazards stands in for an operator, in CARLA only.",
              "- The hazard cars in tests 2, 3 and 4 are scripted rather than autopilot, since autopilot wouldn't hold a set speed or make the cut-in reliably. The speeding car in test 3 measured about 8.5 m/s, not its set 10 m/s.",
              "- Left turns: the recordings had none across moving traffic, so a fixed rule taught them, on runs where the rule or the model was driving. The rule isn't used when driving. The model is two networks: the general driver, frozen, plus a correction learned from those lessons that only acts on a left route.",
              "- On a left turn the model is given the time-to-collision with the first car due to reach its route. That assumes each car keeps its speed and heading, so it wouldn't foresee a car turning across the route.",
              "- Tests 2 and 5 were also repeated on this model with varied traffic: 20 of 20 each, with tests 1, 3 and 4 at 5 of 5 each, and 40 of 40 left turns in harder traffic near the safe limit. That doesn't prove it passes every time: 60 of 60 still allows a failure rate of up to about 1 in 20 (95% confidence). All of this is one approach to one junction, with oncoming cars scripted at 12 m/s.",
              "- The cut-in in test 4 is at the model's normal speed of about 20 km/h, so it isn't high-speed.",
              f"- {smooth_note}", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


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
    """Tagged, coloured lines on stdout. --quiet leaves out the per-tick lines."""

    def __init__(self, quiet=False):
        self.quiet = quiet
        os.system("")  # turns on ANSI colours in a Windows console

    def line(self, tag, text):
        print(f"\033[{TAGS.get(tag, '0')}m[{tag}]\033[0m {text}", flush=True)

    def tick(self, tag, text):
        if not self.quiet:
            self.line(tag, text)


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

    def send(self, inputs, output, applied, model_driving):
        self.last = {"target": float(output[1]), "applied": applied, "model_driving": bool(model_driving)}
        super().send(inputs, output, applied, model_driving)


# ---------------------------------------------------------------------------
# CARLA
# ---------------------------------------------------------------------------
class Tick:
    """What one tick of a test sees."""

    def __init__(self, t, ego, speed, others, stop_gap, gate_on):
        self.t, self.ego, self.speed, self.others, self.stop_gap, self.gate_on = t, ego, speed, others, stop_gap, gate_on


class Suite:
    def __init__(self, carla, client, policy, console, seed=0):
        self.seed = seed  # picks the oncoming traffic in test 2
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
        self.hud = HudSender()
        self.world.wait_for_tick(10.0)
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
    def clear(self):
        """Removes every car and their sensors, and releases the brake, so no test inherits anything."""
        for sensor in self.sensors:
            try:
                sensor.stop()
            except RuntimeError:
                pass
        self.client.apply_batch_sync([self.carla.command.DestroyActor(a) for a in self.sensors])
        self.sensors = []
        cars = list(self.world.get_actors().filter("vehicle.*"))
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

    def straight_path(self, waypoint, metres=120, every=5):
        """Locations straight through the junction, for the Traffic Manager to follow."""
        chooser, current, points = TurnChooser(STRAIGHT), waypoint, []
        for step in range(metres):
            following = current.next(1.0)
            if not following:
                break
            current = chooser.choice(following) if len(following) > 1 else following[0]
            chooser.turn = STRAIGHT
            if step % every == 0:
                points.append(current.transform.location)
        return points

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
        driver = ModelDriver(self.policy, LaneDriver(start, TurnChooser(turn)), self.find_waypoint, rng, telemetry)
        hits = []
        sensor = self.world.spawn_actor(self.world.get_blueprint_library().find("sensor.other.collision"), carla.Transform(), attach_to=ml)
        sensor.listen(lambda event: hits.append((event.other_actor.id, event.other_actor.type_id)))
        self.sensors.append(sensor)
        if build is not None:
            build(ml)
        for _ in range(SETTLE_TICKS):
            self.world.wait_for_tick(TICK_TIMEOUT)

        contacts = {}
        driven = ([], [], [], [])
        lights = self.lights
        lights.runs, lights.suspects, lights.seen = [], {}, {}
        started = now = None
        last_position = None
        was_braking = False
        held_since = None
        polls = []
        min_ttc = math.inf
        self.mqtt.clear()
        done = False
        try:
            while not done:
                snapshot = self.world.wait_for_tick(TICK_TIMEOUT)
                now = snapshot.timestamp.elapsed_seconds
                started = now if started is None else started
                t = now - started
                ego = actor_state(ml)
                speed = math.hypot(ego[2], ego[3])
                others = {a.id: actor_state(a) for a in self.world.get_actors().filter("vehicle.*") if a.id != ml.id}
                walkers = [(w.get_location().x, w.get_location().y, w.get_velocity().x, w.get_velocity().y)
                           for w in self.world.get_actors().filter("walker.pedestrian.*")]
                not_green = lights.not_green()

                asked = time.perf_counter()
                try:
                    gate = self.session.get(GATE_URL, timeout=0.5).json()
                    polls.append((time.perf_counter() - asked) * 1000.0)
                except (requests.RequestException, ValueError):
                    gate = {}
                released = brake_is_off(gate)
                steer, throttle, brake = driver.controls(ego, not_green, list(others.values()), walkers, released)
                ml.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
                for series, value in zip(driven, (steer, throttle, brake, speed)):
                    series.append(value)

                safety = self.safety.poll() or {}
                ttc = safety.get("ttc")
                if ttc is not None:
                    min_ttc = min(min_ttc, ttc)
                gap = min((rectangles_gap(outline(ego[0], ego[1], ego[4]), outline(o[0], o[1], o[4]))
                           for o in others.values() if math.hypot(o[0] - ego[0], o[1] - ego[1]) < 15.0), default=None)
                if gap is not None:
                    result.min_gap = gap if result.min_gap is None else min(result.min_gap, gap)
                result.worst_off_lane = max(result.worst_off_lane, driver.off_lane)
                if last_position is not None:
                    result.distance += math.hypot(ego[0] - last_position[0], ego[1] - last_position[1])
                last_position = ego[:2]
                lights.update({ml.id: ego}, t, ml.id, "SCENARIO")
                stop_gap = red_light_gap(driver.lane.path_ahead(RED_LOOK), not_green)

                while hits:
                    other_id, kind = hits.pop()
                    if is_new_contact(contacts, other_id, t):
                        result.collisions.append({"t": round(t, 1), "with": kind, "speed": round(speed, 1)})
                        self.console.line("ALERT", f"COLLISION with {kind} at {speed:.1f} m/s, t={t:.1f} s")
                        self.hud.event(f"COLLISION with {kind.split('.')[-1]} at {speed:.1f} m/s")

                engaged = gate.get("emergency_brake") is True
                if engaged and not was_braking:
                    result.brake_events += 1
                    reason = gate.get("reason") or ""
                    if result.brake_at is None:
                        result.brake_at, result.brake_ttc = round(t, 2), gate.get("ttc")
                    banner = alert_banner(name, gate.get("ttc"), speed)
                    self.console.line("ALERT", banner.replace("[ALERT - SAFETY GATE TRIGGERED] ", "SAFETY GATE TRIGGERED: ") + f" | {reason}")
                    print(f"\033[1;91m{banner}\033[0m", flush=True)
                    self.hud.event(f"SAFETY GATE: {reason}")
                elif was_braking and not engaged:
                    self.hud.event("Safety brake released")
                was_braking = engaged
                held_since = (held_since if held_since is not None else t) if engaged else None

                state = telemetry.last or {"target": math.nan, "model_driving": False}
                error = state["target"] - speed
                self.console.tick("MODEL", f"Target {state['target'] * 3.6:5.1f} km/h | Actual {speed * 3.6:5.1f} km/h | "
                                           f"Throttle {throttle * 100:3.0f}% | Brake {brake * 100:3.0f}% | "
                                           f"Steering {steer:+.3f} | Dv {error:+.2f} m/s")
                self.console.tick("SAFETY GATE", f"Min TTC {shown(ttc)} s | Path TTC {shown(safety.get('path_ttc'))} s | PET {shown(safety.get('pet'))} s | "
                                                 f"Trajectory Intersects {safety.get('intersects', False)} | "
                                                 f"Heartbeat {shown(polls[-1] if polls else None, 0)} ms | Brake {'ON' if engaged else 'off'}")
                if int(t * 10) % 10 == 0:
                    self.console.tick("MQTT / DITTO", f"Broker {'connected' if self.mqtt.connected else 'NOT CONNECTED'} | "
                                                      f"Topic {COMMAND_TOPIC} | Last brake latency {shown(self.mqtt.latency, 0)} ms")
                self.csv.writerow((f"{time.time():.3f}", test, f"{t:.2f}", f"{ego[0]:.2f}", f"{ego[1]:.2f}", f"{ego[4]:.1f}", f"{speed:.3f}",
                                   f"{state['target']:.3f}", f"{steer:.4f}", f"{throttle:.3f}", f"{brake:.3f}", f"{error:.3f}",
                                   int(state["model_driving"]), int(engaged), shown(ttc, 3), shown(safety.get("path_ttc"), 3),
                                   shown(safety.get("pet"), 3), f"{min(stop_gap, STOP_LINE_RANGE):.1f}", shown(gap, 2), f"{driver.off_lane:.2f}"))
                self.hud.send()

                tick = Tick(t, ego, speed, others, stop_gap, engaged)
                tick.held_for = 0.0 if held_since is None else t - held_since
                tick.turn_yaw = ego[4]
                done = on_tick(tick, result, driver)
                if t >= limit:
                    result.notes.append(f"ran out of time at {limit:.0f} s")
                    self.hud.event("Ran out of time")
                    done = True
        finally:
            result.duration = round(now - started, 1) if started is not None else 0.0
            result.red_light_violations = len(lights.runs)
            result.min_ttc = None if math.isinf(min_ttc) else round(min_ttc, 3)
            result.mqtt_latency_ms = self.mqtt.latency
            result.gate_poll_ms = float(np.median(polls)) if polls else None
            result.smoothness = {k: round(v, 3) for k, v in smoothness(*driven).items()}
            self.csv_file.flush()
        return result

    # --- the tests ---
    def test_lights(self):
        stop = self.lane_with_turn(STRAIGHT)
        yaw = stop.transform.rotation.yaw
        red, green = self.carla.TrafficLightState.Red, self.carla.TrafficLightState.Green
        self.set_lights(yaw, red, red, red)
        line = stop.transform.location
        state = {"stopped_at": None, "green_at": None, "crossed": False, "up_to_speed": False, "waited": False}

        def past_line(ego):
            heading = math.radians(yaw)
            return (ego[0] - line.x) * math.cos(heading) + (ego[1] - line.y) * math.sin(heading)

        def on_tick(tick, result, driver):
            if state["stopped_at"] is None and tick.speed < 0.1 and tick.t > 3.0 and past_line(tick.ego) > -STOP_LINE_RANGE:
                state["stopped_at"] = tick.t
                result.stopping_margin = round(-past_line(tick.ego), 2)
                self.console.line("SCENARIO", f"Stopped {result.stopping_margin:.1f} m before the line, t={tick.t:.1f} s")
                self.hud.event(f"Stopped {result.stopping_margin:.1f} m before the red line")
            if state["stopped_at"] is not None and state["green_at"] is None and tick.t - state["stopped_at"] >= T1_WAIT:
                state["waited"] = True
                state["green_at"] = tick.t
                self.set_lights(yaw, green, red, red)
                self.console.line("SCENARIO", f"Light turns GREEN, t={tick.t:.1f} s")
                self.hud.event("Light turns green")
            if state["green_at"] is not None and past_line(tick.ego) > 0.0:
                state["crossed"] = True
            if state["crossed"] and tick.speed >= T1_CRUISE:
                state["up_to_speed"] = True
            if state["up_to_speed"] and tick.t - state["green_at"] > 8.0:
                return True
            return state["green_at"] is not None and tick.t - state["green_at"] > T1_RECOVER

        result = self.run_stage(1, "Traffic light and speed compliance", "[TEST 1/5] TRAFFIC LIGHT & SPEED COMPLIANCE", stop, STRAIGHT, T1_LIMIT, on_tick)
        return grade_lights(result, state["stopped_at"] is not None, state["waited"], state["crossed"], state["up_to_speed"])

    def left_turn_lanes(self):
        """(stop line with a left turn, the oncoming lanes), for the approach with a long straight view of them."""
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
                return stop, oncoming
        raise RuntimeError("No left turn at junction 189 has a long straight view of oncoming traffic.")

    def test_light_change(self):
        return self.test_left_turn(light_change=True)

    def test_left_turn(self, light_change=False):
        stop, oncoming = self.left_turn_lanes()
        yaw = stop.transform.rotation.yaw
        red, green, yellow = self.carla.TrafficLightState.Red, self.carla.TrafficLightState.Green, self.carla.TrafficLightState.Yellow
        self.set_lights(yaw, green, green, red)
        rng = random.Random(self.seed)
        schedule = deque(traffic_schedule(rng))
        starts = [o.previous(T2_FEED_FROM)[0] for o in oncoming]
        cars = []
        heading = math.radians(yaw)
        line = stop.transform.location
        state = {"turned": False, "turned_at": 0.0, "start_yaw": None, "closest": math.inf, "waited": 0.0, "released": 0,
                 "stopped": False, "go_ttc": None, "changed_at": None, "red": False, "stopping": set()}
        self.console.line("SCENARIO", f"Oncoming traffic seed {self.seed}, {T2_ONCOMING:.0f} m/s")

        def launch(start=None):
            start = start or rng.choice(starts)
            bp = self.world.get_blueprint_library().find("vehicle.audi.a2")
            bp.set_attribute("role_name", "autopilot")
            tf = start.transform
            car = self.world.try_spawn_actor(bp, self.carla.Transform(self.carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z + 0.5), tf.rotation))
            if car is None:
                return False
            car.enable_constant_velocity(self.carla.Vector3D(x=T2_ONCOMING))
            cars.append(car)
            self.actors.append(car)
            return True

        def build(ml):
            # The road already has traffic on it when the ML car sets off
            metres = T2_FEED_FROM
            while metres > T2_PREFILL_TO:
                launch(rng.choice(oncoming).previous(metres)[0])
                metres -= T2_ONCOMING * rng.uniform(*T2_HEADWAY)

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
                self.console.line("SCENARIO", f"Set off with the next car {shown(ttc, 1)} s from the crossing, t={tick.t:.1f} s")
                self.hud.event(f"Set off: next car {shown(ttc, 1)} s away")
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
        result.notes.append(f"seed {self.seed}; waited {state['waited']:.1f} s with a car due; set off with {'no car due' if state['go_ttc'] is None or not math.isfinite(state['go_ttc']) else 'the next car ' + shown(state['go_ttc'], 1) + ' s away'}; "
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

    def test_blind_corner(self):
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
                if launch_now(my_index - nearest, max(tick.speed, 4.0), their_index, T3_SPEED, T3_LEAD):
                    state["launched_at"] = tick.t
                    # Autopilot won't hold a set speed through a junction, so this speed is enforced
                    runner.enable_constant_velocity(self.carla.Vector3D(x=T3_SPEED))
                    self.console.line("SCENARIO", f"Speeding car launched through its red light, ML car {my_index - nearest} m from the crossing, t={tick.t:.1f} s")
                    self.hud.event("Speeding car launched through its red light")
            elif runner.id in tick.others:
                other = tick.others[runner.id]
                if int(tick.t * 10) % 10 == 0:
                    self.console.line("SCENARIO", f"Speeding car at {math.hypot(other[2], other[3]):.1f} m/s, "
                                                  f"{math.hypot(other[0] - my_path[my_index][0], other[1] - my_path[my_index][1]):.0f} m from the crossing; "
                                                  f"ML car {my_index - nearest} m from it at {tick.speed:.1f} m/s, t={tick.t:.1f} s")
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
        return grade_blind_corner(result, state["heard"], state["stopped_at"] is not None)

    def test_cut_in(self):
        pair = self._two_lane_start()
        start, adjacent, _ = pair
        stop = start.next(APPROACH)[0]
        yaw = stop.transform.rotation.yaw
        green = self.carla.TrafficLightState.Green
        self.set_lights(yaw, green, green, green)
        own_lane = [(w.transform.location.x, w.transform.location.y) for w in walk_lane(adjacent, T4_STRAIGHT + 40.0)]
        state = {"cutter": None, "cut_at": None, "in_my_lane": False, "path": own_lane, "speed": T4_SPEED}

        def build(ml):
            state["cutter"] = self.spawn_car("vehicle.audi.tt", adjacent)

        def on_tick(tick, result, driver):
            cutter = state["cutter"]
            if cutter.id not in tick.others:
                return False
            other = tick.others[cutter.id]
            steer, throttle, brake = follow_path(other, state["path"], state["speed"])
            cutter.apply_control(self.carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
            ahead, side_on = along_and_across(tick.ego, other)
            if state["cut_at"] is None and ahead >= T4_CUT_AHEAD and tick.speed > 3.0:
                state["cut_at"] = tick.t
                route = driver.lane.path_ahead(T4_STRAIGHT)
                state["path"] = route[max(0, int(ahead + T4_MERGE)):]
                self.console.line("SCENARIO", f"Car starts cutting in {ahead:.1f} m ahead, t={tick.t:.1f} s")
                self.hud.event(f"Car cuts in {ahead:.1f} m ahead")
            if state["cut_at"] is not None and 0.0 < ahead < T4_REAL_CUT_AHEAD and abs(side_on) < T4_IN_LANE and not state["in_my_lane"]:
                state["in_my_lane"] = True
                state["speed"] = T4_THEN
                result.min_gap = None  # from here on: being side by side at the start isn't a close call
                self.console.line("SCENARIO", f"Car is in the ML car's lane, {ahead:.1f} m ahead, and slowing, t={tick.t:.1f} s")
                self.hud.event(f"Car in the lane, {ahead:.1f} m ahead, slowing")
            return state["cut_at"] is not None and tick.t - state["cut_at"] > T4_RUN

        result = self.run_stage(4, "Cut-in and recovery", "[TEST 4/5] CUT-IN & RECOVERY", stop, STRAIGHT, T4_LIMIT, on_tick, build, start=start)
        return grade_cut_in(result, state["in_my_lane"])

    def _two_lane_start(self):
        """(ML car's start, the lane beside it, does a car there move right to reach the ML car?), on a straight
        stretch with two lanes the same way, as near junction 189 as possible."""
        options = []
        for wp in self.map.generate_waypoints(5.0):
            if wp.is_junction or wp.lane_type != self.carla.LaneType.Driving:
                continue
            for adjacent, moves_right in ((wp.get_left_lane(), True), (wp.get_right_lane(), False)):
                if adjacent is None or adjacent.lane_type != self.carla.LaneType.Driving or adjacent.lane_id * wp.lane_id <= 0:
                    continue
                ahead = wp.next(T4_STRAIGHT)
                if not ahead or ahead[0].is_junction or abs(yaw_difference(wp.transform.rotation.yaw, ahead[0].transform.rotation.yaw)) > 3.0:
                    continue
                if any(step.is_junction for step in wp.next_until_lane_end(5.0)[:int(T4_STRAIGHT / 5.0)]):
                    continue
                location = wp.transform.location
                options.append((math.hypot(location.x - JUNCTION[0], location.y - JUNCTION[1]), id(wp), wp, adjacent, moves_right))
        if not options:
            raise RuntimeError("No straight road with two lanes the same way was found.")
        _, _, wp, adjacent, moves_right = min(options, key=lambda o: o[:2])
        return wp, adjacent, moves_right

    def close(self):
        self.clear()
        self.release_lights()
        self.mqtt.stop()
        self.safety.close()
        self.csv_file.close()


# ---------------------------------------------------------------------------
# Running the suite
# ---------------------------------------------------------------------------
def simulation_running():
    probe = ("import carla; c = carla.Client('localhost', 2000); c.set_timeout(8.0); "
             "c.get_world().wait_for_tick(TICK_TIMEOUT)")
    try:
        return subprocess.run([sys.executable, "-c", probe], capture_output=True, timeout=20, check=False).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def load_rounds():
    path = OUTPUT_DIR / "driving_rounds.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_tests(carla, client, chosen, console, seed=0, repeat=1):
    suite = Suite(carla, client, Policy(), console, seed)
    tests = {1: suite.test_lights, 2: suite.test_left_turn, 3: suite.test_blind_corner, 4: suite.test_cut_in, 5: suite.test_light_change}
    results = []
    try:
        for number in [n for _ in range(repeat) for n in chosen]:
            suite.seed += 1
            try:
                result = tests[number]()
            except RuntimeError as error:
                result = StageResult(number, tests[number].__name__.replace("test_", "").replace("_", " "))
                result.notes.append(f"couldn't run: {error}")
                console.line("ALERT", f"Test {number} couldn't run: {error}")
            results.append(asdict(result))
            console.line("SCENARIO", f"TEST {number}: {'PASS' if result.passed else 'FAIL'} {result.checks} {result.notes}")
            suite.hud.event(f"TEST {number}: {'PASS' if result.passed else 'FAIL'}")
            for _ in range(20):
                suite.world.wait_for_tick(TICK_TIMEOUT)
    finally:
        suite.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", type=int, nargs="*", choices=(1, 2, 3, 4, 5), help="which tests to run; all five if left out")
    parser.add_argument("--quiet", action="store_true", help="leave out the per-tick log lines")
    parser.add_argument("--seed", type=int, help="picks test 2's traffic; different each time if left out")
    parser.add_argument("--repeat", type=int, default=1, help="run the tests this many times, saving only to output/repeat_results.json")
    parser.add_argument("--attached", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _raise_keyboard_interrupt)

    if not args.attached and not simulation_running():
        print("[SCENARIO] The simulation isn't running, so starting everything through run_simulation.py.", flush=True)
        return subprocess.run([sys.executable, str(ROOT / "run_simulation.py"), *(["--quiet"] if args.quiet else [])], cwd=ROOT, check=False).returncode

    import carla

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(30.0)
    console = Console(args.quiet)
    chosen = args.only or [1, 2, 3, 4, 5]
    results = run_tests(carla, client, chosen, console, int(time.time()) if args.seed is None else args.seed, args.repeat)

    OUTPUT_DIR.mkdir(exist_ok=True)
    if args.repeat > 1:
        (OUTPUT_DIR / "repeat_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        for number in chosen:
            mine = [r for r in results if r["test"] == number]
            console.line("SCENARIO", f"Test {number}: passed {sum(r['passed'] for r in mine)} of {len(mine)}")
        return 0 if all(r["passed"] for r in results) else 2
    (OUTPUT_DIR / "PROFESSOR_TEST_SUITE_RESULTS.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(), "control_mode": CONTROL_MODE, "tests": results}, indent=2), encoding="utf-8")
    smooth_path = OUTPUT_DIR / "smoothness.json"
    write_summary(OUTPUT_DIR / "PROFESSOR_DEMO_SUMMARY.md", results, load_rounds(),
                  "Stops end with a short snap in the last moment, from CARLA's low-speed wheel physics rather than the model.",
                  json.loads(smooth_path.read_text(encoding="utf-8")) if smooth_path.exists() else None)
    passed = sum(r["passed"] for r in results)
    console.line("SCENARIO", f"Finished: {passed} of {len(results)} tests passed. See {OUTPUT_DIR / 'PROFESSOR_DEMO_SUMMARY.md'}")
    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
