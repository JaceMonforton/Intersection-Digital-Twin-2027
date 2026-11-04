"""
Records the ML car being corrected, to retrain the driving model where it goes wrong.

The model drives until it gets into trouble: closing on a car or a pedestrian too fast, reaching a red
line too fast, straying from its lane, or standing still for no reason. Then CARLA's autopilot takes the car for a few seconds
and hands it back. Only what autopilot did during those takeovers becomes training data, because that's
the one thing the first recording lacks: how autopilot gets out of places the model gets itself into.

This is simulation-only training, so it doesn't read the SafetyGate. Needs the processor and
test_traffic.py running, and the main Python environment, which has carla.
    python training/record_corrections.py --episodes 6 --minutes 5
"""
import argparse
import math
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from central_gpu_processor import TRAFFIC_MANAGER_PORT
from ml_driver import DrivingPolicy, ModelDriver, StopWatch, actor_state, spawn_ml_car
from record_driving import (
    CARLA_HOST,
    CARLA_PORT,
    RAW_DIR,
    capture_tick,
    convert_all,
    save_chunk,
)
from traffic_monitor import (
    RED_LOOK,
    LaneDriver,
    RedLightWatch,
    closing_speed,
    red_light_gap,
    relative_position,
)

# --- Configuration ---
TAKEOVER_SECONDS = 5.0  # s that autopilot keeps the car after the last reason to
SETTLE_TICKS = 2        # ticks before autopilot's controls are really what the car is doing
COMFORT_DECEL = 4.0     # m/s^2; needing more than this to stop in time is trouble
IN_PATH = 1.6           # m to the side that a car counts as in the way
WALKER_IN_PATH = 3.0    # m to the side, since a pedestrian may be walking into the lane
CAR_LENGTH = 5.0        # m between centres of cars that are touching
WALKER_CLEARANCE = 3.0  # m from the car's centre
STOP_MARGIN = 1.0       # m before a line
MIN_SPEED = 1.0         # m/s; slower than this nothing is urgent
OFF_LANE = 1.0          # m from the route; normal driving stays within about 0.6 m
STUCK_SECONDS = 5.0     # s stopped with no red light ahead
EPISODE_STUCK = 60.0    # s with even autopilot not moving, which ends the episode


# ---------------------------------------------------------------------------
# When autopilot takes over (no CARLA needed)
# ---------------------------------------------------------------------------
def closing_too_fast(ego, other, clearance, side_limit):
    """Would stopping for this actor take harder braking than is comfortable? Actors are (x, y, vx, vy, ...)."""
    ahead, left = relative_position(ego, (*other[:4], 0.0))
    if ahead <= 0.0 or abs(left) > side_limit:
        return False
    closing = closing_speed(ego, other)
    if closing < MIN_SPEED:
        return False
    return closing ** 2 / (2.0 * COMFORT_DECEL) >= ahead - clearance


def takeover_reason(ego, stop_gap, cars, walkers, stopped_for, off_lane=0.0):
    """Why autopilot should take the car now, or None."""
    speed = math.hypot(ego[2], ego[3])
    if speed >= MIN_SPEED and speed ** 2 / (2.0 * COMFORT_DECEL) >= stop_gap - STOP_MARGIN:
        return "red line"
    if any(closing_too_fast(ego, car, CAR_LENGTH, IN_PATH) for car in cars):
        return "car ahead"
    if any(closing_too_fast(ego, walker, WALKER_CLEARANCE, WALKER_IN_PATH) for walker in walkers):
        return "pedestrian"
    if off_lane >= OFF_LANE:
        return "off lane"
    return "stopped" if stopped_for >= STUCK_SECONDS else None


# ---------------------------------------------------------------------------
# Recording (needs carla)
# ---------------------------------------------------------------------------
def run_episode(carla, world, traffic_manager, tm_port, policy, seconds, seed, path):
    rng = random.Random(seed)
    world.wait_for_tick(10.0)
    lights = RedLightWatch(carla, world)
    lines = np.array([(x, y) for _, x, y, _ in lights.lines], np.float64)
    light_list = [light for light, _, _, _ in lights.lines]
    green = carla.TrafficLightState.Green
    world_map = world.get_map()
    car, point = spawn_ml_car(world, rng)

    def find_waypoint(x, y):
        return world_map.get_waypoint(carla.Location(x=float(x), y=float(y), z=0.0), project_to_road=True,
                                      lane_type=carla.LaneType.Driving)

    driver = ModelDriver(policy, LaneDriver(world_map.get_waypoint(point.location), rng), find_waypoint, rng)
    traffic_manager.auto_lane_change(car, False)
    stops = StopWatch()
    car_rows, walker_rows, not_green, samples_from = [], [], [], []
    first_tick, last_frame = None, None
    autopilot_until, settled_at = None, None
    reasons = {}
    started = now = None
    try:
        while started is None or now - started < seconds:
            snapshot = world.wait_for_tick(10.0)
            frame = snapshot.frame
            now = snapshot.timestamp.elapsed_seconds
            started = now if started is None else started
            if last_frame is not None and frame != last_frame + 1:
                print(f"[CORRECT] Missed {frame - last_frame - 1} ticks; stopping this episode here.", flush=True)
                break
            last_frame = frame
            first_tick = frame if first_tick is None else first_tick
            capture_tick(world, frame, light_list, green, car_rows, walker_rows, not_green)

            ego = actor_state(car)
            cars = [actor_state(a) for a in world.get_actors().filter("vehicle.*") if a.id != car.id]
            walkers = [(w.get_location().x, w.get_location().y, w.get_velocity().x, w.get_velocity().y)
                       for w in world.get_actors().filter("walker.pedestrian.*")]
            lit = lights.not_green()
            controls = driver.controls(ego, lit, cars, walkers, True)  # keeps the route up to date in a takeover
            stop_gap = red_light_gap(driver.lane.path_ahead(RED_LOOK), lit)
            stopped_for = stops.update(now, math.hypot(ego[2], ego[3]), stop_gap < RED_LOOK)
            if stopped_for >= EPISODE_STUCK:
                print(f"[CORRECT] Even autopilot is stuck at x={ego[0]:.0f} y={ego[1]:.0f}; ending this episode.", flush=True)
                break

            reason = takeover_reason(ego, stop_gap, cars, walkers, stopped_for, driver.off_lane)
            if reason is not None:
                if autopilot_until is None:
                    reasons[reason] = reasons.get(reason, 0) + 1
                    settled_at = frame + SETTLE_TICKS
                    car.set_autopilot(True, tm_port)
                    print(f"[CORRECT] t={now - started:.1f} s: autopilot takes over: {reason}", flush=True)
                autopilot_until = now + TAKEOVER_SECONDS
            if autopilot_until is not None and now >= autopilot_until:
                autopilot_until = None
                car.set_autopilot(False, tm_port)
            if autopilot_until is None:
                steer, throttle, brake = controls
                car.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
            elif frame >= settled_at:
                samples_from.append((frame, car.id))
    finally:
        car.set_autopilot(False, tm_port)
        car.destroy()
    save_chunk(path, car_rows, walker_rows, not_green, lines, first_tick, world_map.to_opendrive(),
               samples_from=np.array(samples_from, np.int64).reshape(-1, 2))
    print(f"[CORRECT] Episode {seed}: takeovers {reasons}, {len(samples_from)} samples.", flush=True)
    return reasons, len(samples_from)


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--minutes", type=float, default=5.0, help="simulated minutes per episode")
    parser.add_argument("--seed", type=int, default=100, help="seed of the first episode")
    parser.add_argument("--tm-port", type=int, default=TRAFFIC_MANAGER_PORT)
    args = parser.parse_args()
    for name in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _raise_keyboard_interrupt)

    policy = DrivingPolicy()
    import carla

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(30.0)
    world = client.get_world()
    traffic_manager = client.get_trafficmanager(args.tm_port)
    run = time.strftime("%Y%m%d_%H%M%S")
    print(f"[CORRECT] The model drives; autopilot corrects it. {args.episodes} episodes of {args.minutes:g} simulated minutes.", flush=True)
    totals, samples = {}, 0
    try:
        for i in range(args.episodes):
            seed = args.seed + i
            reasons, count = run_episode(carla, world, traffic_manager, args.tm_port, policy, args.minutes * 60.0, seed,
                                         RAW_DIR / f"corrections_{run}_{i:03d}.npz")
            samples += count
            for reason, n in reasons.items():
                totals[reason] = totals.get(reason, 0) + n
    except KeyboardInterrupt:
        print("[CORRECT] Stopping early.", flush=True)
    print(f"[CORRECT] Done: takeovers {totals}, {samples} samples.", flush=True)
    return convert_all()


if __name__ == "__main__":
    sys.exit(main())
