"""
Drives vehicle_ml with the trained driving model: real physics, with the steering and the speed chosen
by models/driving_policy.onnx. A small fixed controller works the pedals to reach that speed. There is
no stand-in for the model, so without the file this refuses to run.

The model is told where to go by a route that LaneDriver picks at random where roads branch, like a
sat-nav. It obeys the SafetyGate: unless the gate explicitly says the brake is off, the car brakes fully.
The brake latches and nothing here releases it. --ignore-brake is for measuring the model's own driving
in CARLA: it still reads and counts brake requests, but drives through them.

Follows the processor's clock, so it needs central_gpu_processor.py running.
    python scripts/ml_driver.py --duration 300 --seed 1 --ignore-brake
"""
import argparse
import json
import math
import random
import signal
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from central_gpu_processor import ML_PAINT
from driving_features import CROSSING_LOOK, SpeedController, encode_inputs
from roadside_link import MODES, make_view
from traffic_monitor import (
    GATE_URL,
    LOOK_AHEAD,
    OUTPUT_DIR,
    RED_LOOK,
    STOP_LINE_WIDTH,
    LaneDriver,
    RedLightWatch,
    clear_of_stop_lines,
    on_lane,
    red_light_gap,
    release_gate,
)
from waymo_predictor_node import LEFT, RIGHT

# --- Configuration ---
CARLA_HOST = "localhost"
CARLA_PORT = 2000
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "driving_policy.onnx"
LANE_LOST = 2.5       # m from the route that means the car isn't following it
PLAN_LOOK = 40         # m of the route ahead that the planner rolls the car out along
CROSSING_WIDTH = 2.0  # m between a car's line and the route that counts as reaching it
CROSSING_MOVING = 1.0  # m/s; slower cars are left to the car slots
SAME_WAY = 0.5        # cosine of the angle under which a car is going along the route, not across it
CAN_STOP = 3.5        # m/s^2; a car needing harder braking than this for its yellow light is taken to be coming through
SPAWN_CLEARANCE = 10.0  # m from any other car
STOPPED_SPEED = 0.1   # m/s
STUCK_SECONDS = 60.0  # s stopped with no red light ahead; autopilot never stood this long in an hour of recording
CONTACT_COOLDOWN = 5.0  # s before touching the same thing counts again, since the sensor fires all through a contact
HARSH_DECEL = 6.0     # m/s^2
LOG_INTERVAL = 15.0   # s between progress lines


# ---------------------------------------------------------------------------
# Pure logic (no CARLA needed)
# ---------------------------------------------------------------------------
def crossing_ttc(path, cars, not_green):
    """Seconds until the first car reaches this route ((x, y) points ahead of here), if each keeps its speed and
    heading, or inf. A car going along the route doesn't count, nor does one with a light that isn't green
    before it gets there and room to stop for it: it's meant to stop, and running a red is the safety gate's
    job. One too close to stop still counts, so a car finishing its turn as the light changes isn't caught out."""
    points = np.asarray(path, dtype=float)
    if len(points) < 2:
        return math.inf
    tangents = np.gradient(points, axis=0)
    tangents /= np.maximum(np.hypot(tangents[:, 0], tangents[:, 1]), 1e-6)[:, None]
    lines = np.asarray(not_green, dtype=float).reshape(-1, 2)
    best = math.inf
    for car in cars:
        speed = math.hypot(car[2], car[3])
        if speed < CROSSING_MOVING:
            continue
        along = np.array((car[2], car[3])) / speed
        offset = points - np.array((car[0], car[1]))
        ahead = offset @ along
        side = np.abs(offset[:, 0] * along[1] - offset[:, 1] * along[0])
        hits = np.flatnonzero((side < CROSSING_WIDTH) & (ahead >= 0.0) & (ahead / speed <= CROSSING_LOOK) & (tangents @ along < SAME_WAY))
        if not len(hits):
            continue
        reach = float(ahead[hits].min())
        to_lines = lines - np.array((car[0], car[1]))
        line_ahead = to_lines @ along
        room_to_stop = speed ** 2 / (2.0 * CAN_STOP)
        if np.any((line_ahead >= room_to_stop) & (line_ahead <= reach) & (np.abs(to_lines[:, 0] * along[1] - to_lines[:, 1] * along[0]) <= STOP_LINE_WIDTH)):
            continue
        best = min(best, reach / speed)
    return best


def inputs_for(lane, ego, not_green, cars, walkers):
    """The model's input for a car at ego on this route. The recorder calls this too, so they can't differ."""
    path = lane.path_ahead(RED_LOOK)
    gap = lane.turn_gap(LEFT)
    ttc = crossing_ttc(lane.path_ahead(LOOK_AHEAD), cars, not_green) if gap is not None else math.inf
    return encode_inputs(ego, path, lane.signal(), red_light_gap(path, not_green), cars, walkers, ttc, gap or 0.0)


def brake_is_off(gate):
    """Only an explicit False releases the car, so an unreadable or odd gate means brake."""
    return gate.get("emergency_brake") is False


class DrivingPolicy:
    """The exported model: one input vector in, (steer, target speed) out."""

    def __init__(self, path=MODEL_PATH):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing, and vehicle_ml has no other driver. Train it with training/train_driving.py.")
        import onnxruntime

        self.session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def __call__(self, inputs):
        return self.session.run(None, {"inputs": inputs[np.newaxis]})[0][0]


class ModelDriver:
    """Turns the world around one car into its controls, and keeps its route up to date."""

    def __init__(self, policy, lane, find_waypoint, rng, telemetry=None):
        self.policy = policy
        self.telemetry = telemetry  # gets the model's real input and output each tick, for the display
        self.controller = SpeedController()
        self.lane = lane
        self.find_waypoint = find_waypoint  # (x, y) -> the nearest driving lane's waypoint
        self.rng = rng
        self.off_lane = 0.0
        self.lost = False
        self.times_lost = 0
        self.planner = None    # a planner.Planner, which may lower the model's speed or have it clear the junction
        self.last_plan = None

    def _follow_route(self, x, y, yaw):
        self.off_lane = self.lane.advance_to(x, y)
        if self.off_lane <= LANE_LOST:
            self.lost = False
            return
        if not self.lost:
            self.lost = True
            self.times_lost += 1
        # Lanes overlap inside a junction, so stay on the old route until there's one that's certainly right
        waypoint = self.find_waypoint(x, y)
        if on_lane(waypoint, yaw):
            self.lane = LaneDriver(waypoint, self.rng)
            self.off_lane = self.lane.advance_to(x, y)

    def controls(self, ego, not_green, cars, walkers, brake_off):
        """(steer, throttle, brake). The model runs either way, so its route stays current under the brake."""
        self._follow_route(ego[0], ego[1], ego[4])
        inputs = inputs_for(self.lane, ego, not_green, cars, walkers)
        output = self.policy(inputs)
        drive = output
        self.last_plan = None
        if self.planner is not None:
            self.last_plan = self.planner.plan(ego, self.lane.path_ahead(PLAN_LOOK), cars, walkers, float(output[1]))
            if self.last_plan.changed:
                drive = np.array((output[0], self.last_plan.speed), dtype=np.float32)  # the model's steering, the planner's speed
        applied = self.controller.step(drive, math.hypot(ego[2], ego[3])) if brake_off else self.controller.hold()
        if self.telemetry is not None:
            self.telemetry.send(inputs, output, applied, brake_off, plan=self.last_plan)  # output stays the model's own
        return applied


def is_new_contact(last_seen, other_id, now):
    """Is this a new collision, and not more of the one already happening? Updates last_seen."""
    new = now - last_seen.get(other_id, -math.inf) >= CONTACT_COOLDOWN
    last_seen[other_id] = now
    return new


def smoothness(steers, throttles, brakes, speeds, tick=0.1):
    """How smooth some driving was, from each tick's controls and speed: the numbers Phase 1 tunes against."""
    steers, throttles, brakes, speeds = (np.asarray(v, dtype=float) for v in (steers, throttles, brakes, speeds))
    if len(speeds) < 3:
        return {}
    minutes = max(len(speeds) * tick / 60.0, 1e-6)
    pedal = np.sign(throttles - brakes)
    pressed = pedal[pedal != 0.0]
    acceleration = np.diff(speeds) / tick
    jerk = np.diff(acceleration) / tick
    return {
        "steer_change_rms": float(np.sqrt(np.mean(np.diff(steers) ** 2))),
        "steer_reversals_per_min": float(np.sum(np.diff(np.sign(np.diff(steers))) != 0) / minutes),
        "pedal_switches_per_min": float(np.sum(np.diff(pressed) != 0) / minutes) if len(pressed) > 1 else 0.0,
        "speed_change_p95": float(np.percentile(np.abs(acceleration), 95)),
        "jerk_p95": float(np.percentile(np.abs(jerk), 95)),
        "hardest_brake": float(-acceleration.min()),
        "harsh_stops_per_min": float(np.sum((acceleration[1:] < -HARSH_DECEL) & (acceleration[:-1] >= -HARSH_DECEL)) / minutes),
    }


class StopWatch:
    """How long the car has stood still with no red light to explain it."""

    def __init__(self):
        self.since = None
        self.longest = 0.0

    def update(self, now, speed, at_red):
        if speed >= STOPPED_SPEED or at_red:
            self.since = None
            return 0.0
        self.since = now if self.since is None else self.since
        self.longest = max(self.longest, now - self.since)
        return now - self.since


@dataclass
class DriveReport:
    seed: int = 0
    ignored_brake: bool = False
    duration: float = 0.0
    distance: float = 0.0
    missed_ticks: int = 0
    off_lane_mean: float = 0.0
    off_lane_p95: float = 0.0
    off_lane_max: float = 0.0
    times_lost: int = 0
    turns_planned: dict = field(default_factory=lambda: {"left": 0, "right": 0})
    collisions: list = field(default_factory=list)
    red_light_runs: list = field(default_factory=list)
    brake_requests: int = 0
    longest_stop: float = 0.0
    got_stuck: bool = False
    smoothness: dict = field(default_factory=dict)


def print_report(r):
    km = max(r.distance / 1000.0, 1e-6)
    print("\n" + "=" * 60)
    print("ML DRIVER REPORT (the model drove)")
    print("=" * 60)
    print(f"Drove               {r.distance:.0f} m in {r.duration:.0f} s, seed {r.seed}, {r.missed_ticks} ticks missed")
    print(f"Off its lane        mean {r.off_lane_mean:.2f} m, p95 {r.off_lane_p95:.2f} m, worst {r.off_lane_max:.2f} m; lost it {r.times_lost} times")
    print(f"Turns on its route  {r.turns_planned['left']} left, {r.turns_planned['right']} right")
    print(f"Collisions          {len(r.collisions)} ({len(r.collisions) / km:.1f} per km)")
    print(f"Red lights run      {len(r.red_light_runs)} ({len(r.red_light_runs) / km:.1f} per km)")
    print(f"Brake requests      {r.brake_requests}{' (ignored)' if r.ignored_brake else ''}")
    print(f"Smoothness          {r.smoothness}")
    print(f"Longest unexplained stop {r.longest_stop:.0f} s{'; GOT STUCK, run ended early' if r.got_stuck else ''}")


# ---------------------------------------------------------------------------
# CARLA
# ---------------------------------------------------------------------------
def spawn_ml_car(world, rng, stop_lines=()):
    """vehicle_ml at a random free spawn point anywhere on the map, with physics on, and back from every
    stop line, since a start right at a red light says nothing about how it drives."""
    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    bp.set_attribute("role_name", "vehicle_ml")
    bp.set_attribute("color", ML_PAINT)
    cars = list(world.get_actors().filter("vehicle.*"))
    points = clear_of_stop_lines(world.get_map().get_spawn_points(), stop_lines)
    rng.shuffle(points)
    for point in points:
        if any(car.get_location().distance(point.location) < SPAWN_CLEARANCE for car in cars):
            continue
        car = world.try_spawn_actor(bp, point)
        if car is not None:
            return car, point
    raise RuntimeError("No free spawn point for vehicle_ml.")


def actor_state(actor):
    tf, v = actor.get_transform(), actor.get_velocity()
    return tf.location.x, tf.location.y, v.x, v.y, tf.rotation.yaw


def drive(carla, world, policy, duration, seed, ignore_brake, release_hold=None, perception="truth"):
    """release_hold is seconds after which a latched brake is released, standing in for an operator. Left
    as None, nothing here releases it."""
    rng = random.Random(seed)
    report = DriveReport(seed=seed, ignored_brake=ignore_brake)
    world.wait_for_tick(10.0)
    lights = RedLightWatch(carla, world)
    world_map = world.get_map()
    car, point = spawn_ml_car(world, rng, lights.stop_line_positions())
    print(f"[ML DRIVER] vehicle_ml is driven by {MODEL_PATH.name}, starting at x={point.location.x:.0f} y={point.location.y:.0f}.", flush=True)

    def find_waypoint(x, y):
        return world_map.get_waypoint(carla.Location(x=float(x), y=float(y), z=0.0), project_to_road=True,
                                      lane_type=carla.LaneType.Driving)

    driver = ModelDriver(policy, LaneDriver(world_map.get_waypoint(point.location), rng), find_waypoint, rng)
    sensor = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"), carla.Transform(), attach_to=car)
    hits = []
    sensor.listen(lambda event: hits.append((event.other_actor.id, event.other_actor.type_id)))
    contacts = {}

    session = requests.Session()
    blinkers = carla.VehicleLightState
    shown_signal = None
    stops = StopWatch()
    off_lane = []
    driven = ([], [], [], [])
    last_frame = last_position = started = None
    braked_before = False
    held_since = None
    last_log = now = 0.0
    seen_turns = set()
    view = make_view(perception, carla, world, car, 0.1)
    print(f"[ML DRIVER] It's told about other actors by: {view.name}", flush=True)
    try:
        while duration <= 0 or now < duration:
            snapshot = world.wait_for_tick(10.0)
            if last_frame is not None:
                report.missed_ticks += snapshot.frame - last_frame - 1
            last_frame = snapshot.frame
            started = snapshot.timestamp.elapsed_seconds if started is None else started
            now = snapshot.timestamp.elapsed_seconds - started

            ego = actor_state(car)
            speed = math.hypot(ego[2], ego[3])
            cars = {a.id: actor_state(a) for a in world.get_actors().filter("vehicle.*") if a.id != car.id}
            walkers = {w.id: (w.get_location().x, w.get_location().y, w.get_velocity().x, w.get_velocity().y)
                       for w in world.get_actors().filter("walker.pedestrian.*")}
            not_green = lights.not_green()
            seen_cars, seen_walkers, fresh = view.see(ego, cars, walkers, snapshot.timestamp.elapsed_seconds)

            try:
                gate = session.get(GATE_URL, timeout=0.5).json()
            except (requests.RequestException, ValueError):
                gate = {}
            released = brake_is_off(gate)
            if not released and not braked_before:
                report.brake_requests += 1
                print(f"[ML DRIVER] t={now:.1f} s: brake requested: {gate.get('reason') or 'gate unreadable'}", flush=True)
            braked_before = not released
            if gate.get("emergency_brake") is True:
                held_since = now if held_since is None else held_since
                if release_hold is not None and now - held_since >= release_hold:
                    try:
                        release_gate(session)
                        held_since = None
                    except requests.RequestException:
                        pass  # still braked, so try again next tick
            elif gate:
                held_since = None

            steer, throttle, brake = driver.controls(ego, not_green, seen_cars, seen_walkers, fresh and (released or ignore_brake))
            car.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
            for series, value in zip(driven, (steer, throttle, brake, speed)):
                series.append(value)

            wanted = driver.lane.signal()
            if wanted != shown_signal:
                shown_signal = wanted
                car.set_light_state({LEFT: blinkers.LeftBlinker, RIGHT: blinkers.RightBlinker}.get(wanted, blinkers.NONE))
            for index, turn in driver.lane.turns.items():
                if (id(driver.lane), index) not in seen_turns:
                    seen_turns.add((id(driver.lane), index))
                    report.turns_planned["left" if turn == LEFT else "right"] += 1

            off_lane.append(driver.off_lane)
            if last_position is not None:
                report.distance += math.hypot(ego[0] - last_position[0], ego[1] - last_position[1])
            last_position = ego[:2]
            lights.update({car.id: ego}, now, car.id, "ML DRIVER")
            while hits:
                other_id, other = hits.pop()
                if not is_new_contact(contacts, other_id, now):
                    continue
                report.collisions.append({"t": round(now, 1), "with": other, "speed": round(speed, 1)})
                print(f"[ML DRIVER] t={now:.1f} s: COLLISION with {other} at {speed:.1f} m/s", flush=True)

            at_red = red_light_gap(driver.lane.path_ahead(RED_LOOK), not_green) < RED_LOOK
            if not released and not ignore_brake:
                at_red = True  # held by the brake, so the stop isn't the model's
            if stops.update(now, speed, at_red) >= STUCK_SECONDS:
                report.got_stuck = True
                print(f"[ML DRIVER] t={now:.1f} s: stuck for {STUCK_SECONDS:.0f} s at x={ego[0]:.0f} y={ego[1]:.0f}; ending the run.", flush=True)
                break
            if now - last_log >= LOG_INTERVAL:
                last_log = now
                print(f"[ML DRIVER] t={now:5.0f} s: {report.distance:.0f} m, {speed:.1f} m/s, off lane {driver.off_lane:.2f} m, "
                      f"lost {driver.times_lost}, collisions {len(report.collisions)}, red runs {len(lights.runs)}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        view.close()
        sensor.stop()
        sensor.destroy()
        car.destroy()

    report.duration = now
    report.times_lost = driver.times_lost
    report.red_light_runs = lights.runs
    report.longest_stop = round(stops.longest, 1)
    report.smoothness = {k: round(v, 3) for k, v in smoothness(*driven).items()}
    if off_lane:
        report.off_lane_mean = round(float(np.mean(off_lane)), 3)
        report.off_lane_p95 = round(float(np.percentile(off_lane, 95)), 3)
        report.off_lane_max = round(float(np.max(off_lane)), 3)
    return report


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=300.0, help="simulated seconds; 0 runs until Ctrl+C")
    parser.add_argument("--seed", type=int, default=0, help="picks the start and the route")
    parser.add_argument("--ignore-brake", action="store_true", help="drive through brake requests, to measure the model alone")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--perception", choices=MODES, default="truth", help="what it's told about other actors; rsu needs the processor running with --detector onnx")
    args = parser.parse_args()

    for name in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _raise_keyboard_interrupt)

    try:
        policy = DrivingPolicy(args.model)
    except FileNotFoundError as error:
        print(f"[ML DRIVER] Refusing to run: {error}")
        return 1

    import carla

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(30.0)
    try:
        report = drive(carla, client.get_world(), policy, args.duration, args.seed, args.ignore_brake, perception=args.perception)
    except RuntimeError as error:
        print(f"[ML DRIVER] Stopped: {error}. Is central_gpu_processor.py running?")
        return 1
    print_report(report)
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"ml_drive_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(asdict(report), indent=2))
    print(f"\nReport: {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
