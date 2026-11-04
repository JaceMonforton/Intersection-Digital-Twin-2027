"""
Records autopilot cars to train the ML car's driving model, and converts the recordings into the
inputs and targets that scripts/driving_features.py defines.

Recording only reads the world and follows the processor's clock. It turns off the cars' automatic
lane changes, because the route given to the model never changes lane. The raw files keep the plain
car states, so changing the model's inputs doesn't mean recording again.

Each car's route comes from hindsight: LaneDriver walks the lanes ahead of it, as it will for the ML
car, but takes whichever branch the car really went down.

Needs the processor and test_traffic.py running, and the main Python environment, which has carla.
    python training/record_driving.py --minutes 60
    python training/record_driving.py --convert     # convert what's recorded, with no CARLA running
"""
import argparse
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from central_gpu_processor import MAX_PLAUSIBLE_SPEED, TRAFFIC_MANAGER_PORT
from driving_features import (
    INPUT_SIZE,
    OUTPUT_SIZE,
    STOP_GAP,
    STOP_PRESENT,
    TARGET_HORIZON,
    output_from_driving,
)
from ml_driver import LANE_LOST, inputs_for
from traffic_monitor import (
    BRANCH_LOOK,
    LOOK_AHEAD,
    RED_RUN_CONFIRM,
    WAYPOINT_STEP,
    LaneDriver,
    RedLightWatch,
    on_lane,
)
from waymo_predictor_node import LEFT, RIGHT, STEP_SECONDS, STRAIGHT

# --- Configuration ---
CARLA_HOST = "localhost"
CARLA_PORT = 2000
DRIVING_DATA = Path(__file__).resolve().parent.parent / "data" / "driving"
RAW_DIR = DRIVING_DATA / "raw"
CONVERTED_DIR = DRIVING_DATA / "converted"

CHUNK_MINUTES = 5.0    # simulated minutes per file, so that a crash loses little
# A route can only be known once the car has driven it, so the end of each track has no label
ROUTE_DISTANCE = LOOK_AHEAD + BRANCH_LOOK * WAYPOINT_STEP
FUTURE_WINDOW = 2 * ROUTE_DISTANCE  # m of the car's track that branches are compared against
BRANCH_COMPARE = 15    # waypoints walked down a branch before comparing it with the track
STOPPED_SPEED = 0.5    # m/s; under this a car has stopped for a line

CAR_COLUMNS = ("tick", "id", "x", "y", "vx", "vy", "yaw", "steer", "throttle", "brake")
TICK, ID, X, Y, VX, VY, YAW, STEER, THROTTLE, BRAKE = range(len(CAR_COLUMNS))
HORIZON_TICKS = round(TARGET_HORIZON / STEP_SECONDS)


# ---------------------------------------------------------------------------
# Hindsight labels (no CARLA needed)
# ---------------------------------------------------------------------------
class FollowTrack:
    """Stands in for LaneDriver's random choice: picks the branch the car really took."""

    def __init__(self, track):
        self.track = track  # N x (x, y)
        self.distance = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(track, axis=0).T))))
        self.future = track

    def look_from(self, index):
        end = np.searchsorted(self.distance, self.distance[index] + FUTURE_WINDOW)
        self.future = self.track[index:end + 1]

    def remaining(self, index):
        return self.distance[-1] - self.distance[index]

    def choice(self, branches):
        return min(branches, key=self._miss)

    def _miss(self, waypoint):
        end = waypoint
        for _ in range(BRANCH_COMPARE):
            following = end.next(WAYPOINT_STEP)
            if not following:
                break
            end = following[0]
        location = end.transform.location
        return np.min(np.hypot(self.future[:, 0] - location.x, self.future[:, 1] - location.y))


def split_tracks(rows):
    """One car's rows, sorted by tick, cut wherever a tick was missed or the car was moved."""
    rows = rows[np.argsort(rows[:, TICK])]
    if len(rows) < 2:
        return []
    steps = np.hypot(np.diff(rows[:, X]), np.diff(rows[:, Y]))
    cuts = np.flatnonzero((np.diff(rows[:, TICK]) != 1) | (steps > MAX_PLAUSIBLE_SPEED * STEP_SECONDS)) + 1
    return [track for track in np.split(rows, cuts) if len(track) >= 2]


def drives_through(track, distance, index, gap):
    """Does this car, gap metres from a line whose light isn't green, carry on past it without stopping?
    Autopilot clears amber and sometimes stops right at the line, hence the margin past it."""
    end = np.searchsorted(distance, distance[index] + gap + RED_RUN_CONFIRM)
    return np.hypot(track[index:end + 1, VX], track[index:end + 1, VY]).min() > STOPPED_SPEED


def label_track(track, find_waypoint, not_green_at, cars_at, walkers_at, counts, samples_from=None):
    """(inputs, target, route) for each tick of a car's track where its route is known."""
    follow = FollowTrack(track[:, [X, Y]])
    driver = None
    samples = []
    for index, row in enumerate(track):
        if follow.remaining(index) < ROUTE_DISTANCE:
            counts["no_future"] += len(track) - index
            break
        x, y, yaw = row[X], row[Y], row[YAW]
        follow.look_from(index)
        if driver is None:
            # Lanes overlap inside a junction, so the nearest one could be the wrong one
            waypoint = find_waypoint(x, y)
            if not on_lane(waypoint, yaw):
                counts["no_lane"] += 1
                continue
            driver = LaneDriver(waypoint, follow)
        if driver.advance_to(x, y) > LANE_LOST:
            driver = None
            counts["lost_lane"] += 1
            continue
        tick = int(row[TICK])
        if samples_from is not None and (tick, int(row[ID])) not in samples_from:
            counts["not_wanted"] += 1
            continue
        others = [car for car in cars_at(tick) if car[ID] != row[ID]]
        inputs = inputs_for(driver, (x, y, row[VX], row[VY], yaw), not_green_at(tick),
                            [car[X:YAW + 1] for car in others], walkers_at(tick))
        # The same input can't mean both stop and carry on, and the model has to learn to stop
        if inputs[STOP_PRESENT] == 1.0 and drives_through(track, follow.distance, index, float(inputs[STOP_GAP])):
            counts["drove_through"] += 1
            continue
        if index + HORIZON_TICKS >= len(track):
            counts["no_future"] += 1
            continue
        later = track[index + HORIZON_TICKS]
        samples.append((inputs, output_from_driving(row[STEER], np.hypot(later[VX], later[VY])), driver.signal()))
        counts["kept"] += 1
    return samples


def _rows_by_tick(rows, first_tick, ticks):
    """A lookup of the rows belonging to each tick. Column 0 of rows must be the tick."""
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    edges = np.searchsorted(rows[:, 0], first_tick + np.arange(ticks + 1))
    return lambda tick: rows[edges[tick - first_tick]:edges[tick - first_tick + 1]]


def convert_chunk(cars, walkers, not_green, lines, first_tick, find_waypoint, samples_from=None):
    """Inputs, targets, routes and counts from one raw recording. samples_from limits the samples to
    these (tick, car id) pairs; the other rows still give the routes and the traffic around them."""
    ticks = len(not_green)
    cars_at = _rows_by_tick(cars, first_tick, ticks)
    walkers_by_tick = _rows_by_tick(walkers, first_tick, ticks)
    red_lines = [[tuple(line) for line in lines[flags]] for flags in not_green]
    counts = {"kept": 0, "no_future": 0, "no_lane": 0, "lost_lane": 0, "drove_through": 0, "not_wanted": 0}
    samples = []
    for car_id in np.unique(cars[:, ID]):
        for track in split_tracks(cars[cars[:, ID] == car_id]):
            samples += label_track(track, find_waypoint, lambda tick: red_lines[tick - first_tick], cars_at,
                                   lambda tick: walkers_by_tick(tick)[:, 1:], counts, samples_from)
    if not samples:
        return (np.zeros((0, INPUT_SIZE), np.float32), np.zeros((0, OUTPUT_SIZE), np.float32),
                np.zeros(0, np.int8), counts)
    inputs, targets, routes = zip(*samples)
    return np.stack(inputs), np.stack(targets), np.array(routes, np.int8), counts


# ---------------------------------------------------------------------------
# Recording and converting (needs carla)
# ---------------------------------------------------------------------------
def capture_tick(world, frame, lights, green, car_rows, walker_rows, not_green, prepare_car=None, role=None):
    """Appends this tick's cars (only this role, if given), walkers and light states."""
    for car in world.get_actors().filter("vehicle.*"):
        if role is not None and car.attributes.get("role_name") != role:
            continue
        if prepare_car is not None:
            prepare_car(car)
        tf, v, c = car.get_transform(), car.get_velocity(), car.get_control()
        car_rows.append((frame, car.id, tf.location.x, tf.location.y, v.x, v.y, tf.rotation.yaw,
                         c.steer, c.throttle, c.brake))
    for walker in world.get_actors().filter("walker.pedestrian.*"):
        location, v = walker.get_location(), walker.get_velocity()
        walker_rows.append((frame, location.x, location.y, v.x, v.y))
    not_green.append([light.get_state() != green for light in lights])


def save_chunk(path, car_rows, walker_rows, not_green, lines, first_tick, xodr, **extra):
    if not car_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, cars=np.array(car_rows, np.float64),
                        walkers=np.array(walker_rows, np.float64).reshape(-1, 5),
                        not_green=np.array(not_green, bool), lines=lines, first_tick=first_tick, xodr=xodr, **extra)
    print(f"[RECORD] Saved {len(car_rows)} car samples to {path}", flush=True)


def record(minutes, tm_port):
    import carla

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(20.0)
    world = client.get_world()
    try:
        world.wait_for_tick(10.0)
    except RuntimeError:
        print("[RECORD] No ticks arriving. Is central_gpu_processor.py running?")
        return 1
    watch = RedLightWatch(carla, world)
    lines = np.array([(x, y) for _, x, y, _ in watch.lines], np.float64)
    lights = [light for light, _, _, _ in watch.lines]
    green = carla.TrafficLightState.Green
    xodr = world.get_map().to_opendrive()
    traffic_manager = client.get_trafficmanager(tm_port)
    prepared = set()

    def prepare(car):
        if car.id not in prepared:
            traffic_manager.auto_lane_change(car, False)
            prepared.add(car.id)

    run = time.strftime("%Y%m%d_%H%M%S")
    chunk_ticks = round(CHUNK_MINUTES * 60.0 / STEP_SECONDS)
    total_ticks = round(minutes * 60.0 / STEP_SECONDS)
    car_rows, walker_rows, not_green = [], [], []
    first_tick, last_frame, chunk, recorded, missed = None, None, 0, 0, 0
    print(f"[RECORD] Recording autopilot cars for {minutes:g} simulated minutes. Ctrl+C saves what there is.", flush=True)
    try:
        while recorded < total_ticks:
            frame = world.wait_for_tick(10.0).frame
            if last_frame is not None and frame != last_frame + 1:
                # Cut the chunk, so each file's light states stay in step with its ticks
                missed += frame - last_frame - 1
                save_chunk(RAW_DIR / f"{run}_{chunk:03d}.npz", car_rows, walker_rows, not_green, lines, first_tick, xodr)
                car_rows, walker_rows, not_green, first_tick = [], [], [], None
                chunk += 1
            last_frame = frame
            first_tick = frame if first_tick is None else first_tick

            capture_tick(world, frame, lights, green, car_rows, walker_rows, not_green, prepare, "autopilot")

            recorded += 1
            if len(not_green) >= chunk_ticks:
                save_chunk(RAW_DIR / f"{run}_{chunk:03d}.npz", car_rows, walker_rows, not_green, lines, first_tick, xodr)
                car_rows, walker_rows, not_green, first_tick = [], [], [], None
                chunk += 1
    except KeyboardInterrupt:
        print("[RECORD] Stopping early.", flush=True)
    save_chunk(RAW_DIR / f"{run}_{chunk:03d}.npz", car_rows, walker_rows, not_green, lines, first_tick, xodr)
    print(f"[RECORD] Done: {recorded} ticks, {missed} missed.", flush=True)
    return 0


def convert_all():
    """Converts every raw file that hasn't been converted yet. Needs the carla package, not a server."""
    import carla

    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
    maps = {}
    for path in sorted(RAW_DIR.glob("*.npz")):
        out = CONVERTED_DIR / path.name
        if out.exists():
            continue
        started = time.perf_counter()
        raw = np.load(path)
        xodr = str(raw["xodr"])
        world_map = maps.setdefault(xodr, carla.Map("recorded", xodr))

        def find_waypoint(x, y, world_map=world_map):
            return world_map.get_waypoint(carla.Location(x=float(x), y=float(y), z=0.0), project_to_road=True,
                                          lane_type=carla.LaneType.Driving)

        samples_from = {(int(tick), int(car)) for tick, car in raw["samples_from"]} if "samples_from" in raw else None
        inputs, targets, routes, counts = convert_chunk(raw["cars"], raw["walkers"], raw["not_green"], raw["lines"],
                                                        int(raw["first_tick"]), find_waypoint, samples_from)
        np.savez_compressed(out, inputs=inputs, targets=targets, routes=routes)
        turns = {name: int((routes == value).sum()) for name, value in (("left", LEFT), ("straight", STRAIGHT), ("right", RIGHT))}
        print(f"[CONVERT] {path.name}: {counts}, routes {turns}, {math.ceil(time.perf_counter() - started)} s", flush=True)
    return 0


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=60.0, help="simulated minutes to record")
    parser.add_argument("--tm-port", type=int, default=TRAFFIC_MANAGER_PORT)
    parser.add_argument("--convert", action="store_true", help="convert recordings and don't record")
    args = parser.parse_args()
    for name in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _raise_keyboard_interrupt)
    code = 0 if args.convert else record(args.minutes, args.tm_port)
    return code or convert_all()


if __name__ == "__main__":
    sys.exit(main())
