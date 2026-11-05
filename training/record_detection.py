"""
Records merged sweeps from the corner LiDARs with the true box of every actor, to train the roadside detector.

For each seed it starts test_traffic.py at one of a few traffic levels, then adds its own pedestrians around
the junction. test_traffic.py scatters its pedestrians over the whole map and doesn't let them cross roads, so
it would give almost none here. They keep to the pavements and sometimes use a crossing. They're steered by this script and not by CARLA's AI walkers, which the LiDAR
passes straight through and which report no speed. Each file holds one sweep, and the raw points are kept, so changing the grid
doesn't mean recording again. A sweep is only saved if it's from the same frame as the labels.

Needs the processor running, and the main Python environment, which has carla. Keep whole seeds aside for
testing, since neighbouring sweeps are nearly the same.
    python training/record_detection.py --seeds 1 2 3 4 5 --sweeps 600
    python training/record_detection.py --check 1    # summarise a seed and save an image with its boxes on
"""
import argparse
import itertools
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from central_gpu_processor import (
    CARLA_HOST,
    CARLA_MAP,
    CARLA_PORT,
    INTERSECTION_JUNCTION,
    LIDAR_MOUNTS,
    LIDAR_POINTS_PER_SECOND,
    TRAFFIC_MANAGER_PORT,
    find_junction,
    merged_sweep,
    spawn_lidars,
)
from detection_features import (
    GRID_HALF,
    MIN_POINTS,
    PEDESTRIAN,
    POINT_SCALE,
    VEHICLE,
    count_points,
    crop_to_grid,
    label_row,
    load_sweep,
)

RAW_DIR = ROOT / "data" / "detection" / "raw"
CHECK_DIR = ROOT / "data" / "lidar_check"
TICKS_PER_SWEEP = 10    # 1 s apart, since neighbouring ticks are nearly the same scene
WARM_UP_TICKS = 300     # so that the traffic has spread out and reached the junction
# (vehicles, pedestrians at the junction), by seed. --near-view has 53 spawn points far enough from the stop lines
TRAFFIC_LEVELS = ((25, 15), (35, 30), (50, 45))
WALKER_RADIUS = 45.0    # m around the junction centre where the added pedestrians spawn and walk to
WALKER_STEER_TICKS = 10
WALKER_ARRIVED = 1.5    # m from its target where a pedestrian picks a new one
WALKER_MAX_TICKS = 150  # and when it picks one anyway, in case something's in its way
WALKER_HOP = 8.0        # m to the next spot on a pavement: too short to reach the far side of a road
WALKER_SPOTS = 300      # spots on the pavements around the junction
CROSSING_REACH = 15.0   # m from a crossing for a pedestrian to choose to use it
CROSSING_CHANCE = 0.1   # of using a crossing in reach, each time it picks somewhere to go
WALKER_STANDING = 0.15  # the share of pedestrians that stand still, as at a crossing
LOCATION_TRIES = 5000
TRAFFIC_TIMEOUT = 120.0


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def traffic_level(seed):
    return TRAFFIC_LEVELS[seed % len(TRAFFIC_LEVELS)]


def save_sweep(path, points, labels):
    np.savez_compressed(path, points=np.round(crop_to_grid(points)[:, :3] * POINT_SCALE).astype(np.int16),
                        labels=np.array(labels, np.float32).reshape(-1, 11))


def labels_for(actors, snapshot, center, points):
    """A label for each actor in the grid, read from the snapshot so that it's the sweep's own frame."""
    rows = []
    for actor in actors:
        state = snapshot.find(actor.id)
        if state is None:
            continue
        tf, box = state.get_transform(), actor.bounding_box
        c = tf.transform(box.location)
        offset = (c.x - center.x, c.y - center.y, c.z - center.z)
        if max(abs(offset[0]), abs(offset[1])) >= GRID_HALF:
            continue
        v = state.get_velocity()
        kind = PEDESTRIAN if actor.type_id.startswith("walker.") else VEHICLE
        row = label_row(offset, (box.extent.x, box.extent.y, box.extent.z), tf.rotation.yaw, (v.x, v.y), kind)
        row[8] = count_points(row, points)
        rows.append(row)
    return rows


@dataclass
class Walker:
    actor: object
    speed: float
    route: list = field(default_factory=list)
    ticks_left: int = 0


def steer(position, target, arrived=WALKER_ARRIVED):
    """The unit (x, y) from a position towards its target, or None once it's there."""
    dx, dy = target[0] - position[0], target[1] - position[1]
    distance = math.hypot(dx, dy)
    return None if distance <= arrived else (dx / distance, dy / distance)


def crossing_ends(corners):
    """The two kerbs of a crossing, from its four corners: the middles of its two short sides."""
    a, b, c, d = [np.array(corner[:2], dtype=float) for corner in corners[:4]]
    if np.linalg.norm(a - b) < np.linalg.norm(b - c):
        return tuple((a + b) / 2), tuple((c + d) / 2)
    return tuple((b + c) / 2), tuple((d + a) / 2)


def next_route(position, spots, crossings, rng):
    """Where a pedestrian goes next: over a crossing now and then, and otherwise to a nearby spot on the
    pavement, so that they stay off the road except where people really cross it."""
    near_crossing = [ends for ends in crossings if min(math.dist(position, end) for end in ends) <= CROSSING_REACH]
    if near_crossing and rng.random() < CROSSING_CHANCE:
        ends = rng.choice(near_crossing)
        return sorted(ends, key=lambda end: math.dist(position, end))
    close = [spot for spot in spots if math.dist(position, spot) <= WALKER_HOP]
    return [rng.choice(close or sorted(spots, key=lambda spot: math.dist(position, spot))[:3])]


def crossings_near(world, center):
    """The ends of each crossing around the junction. The map lists each one as five corners, closing the loop."""
    points = [(p.x, p.y) for p in world.get_map().get_crosswalks()]
    crossings = [crossing_ends(points[i:i + 4]) for i in range(0, len(points) - 4, 5)]
    return [ends for ends in crossings if min(math.hypot(e[0] - center.x, e[1] - center.y) for e in ends) <= WALKER_RADIUS]


def spawn_walkers(carla, client, world, center, count, seed):
    """Pedestrians on the pavements around the junction, with the spots and crossings they walk between."""
    spots = []
    for _ in range(LOCATION_TRIES):
        location = world.get_random_location_from_navigation()
        if (location is not None and location.distance(center) <= WALKER_RADIUS
                and world.get_map().get_waypoint(location, project_to_road=False, lane_type=carla.LaneType.Driving) is None):
            spots.append((location.x, location.y))
            if len(spots) >= WALKER_SPOTS:
                break
    if not spots:
        raise RuntimeError("Found nowhere around the junction for pedestrians to walk.")
    crossings = crossings_near(world, center)
    rng = random.Random(seed)
    batch = []
    for _ in range(count):
        bp = rng.choice(world.get_blueprint_library().filter("walker.pedestrian.*"))
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        x, y = rng.choice(spots)
        batch.append(carla.command.SpawnActor(bp, carla.Transform(carla.Location(x=x, y=y, z=center.z + 1.2))))
    ids = [r.actor_id for r in client.apply_batch_sync(batch) if not r.error]
    world.wait_for_tick(10.0)
    walkers = [Walker(actor, 0.0 if rng.random() < WALKER_STANDING else rng.uniform(1.0, 2.0)) for actor in world.get_actors(ids)]
    print(f"[RECORD] {len(walkers)} pedestrians on {len(spots)} pavement spots, with {len(crossings)} crossings.", flush=True)
    return walkers, spots, crossings, rng


def steer_walkers(carla, walkers, snapshot, spots, crossings, rng):
    for walker in walkers:
        state = snapshot.find(walker.actor.id)
        if state is None:
            continue
        location = state.get_transform().location
        position = (location.x, location.y)
        walker.ticks_left -= WALKER_STEER_TICKS
        if walker.ticks_left <= 0:
            walker.route = []
        while walker.route and steer(position, walker.route[0]) is None:
            walker.route.pop(0)
        if not walker.route:
            walker.route, walker.ticks_left = next_route(position, spots, crossings, rng), WALKER_MAX_TICKS
        direction = steer(position, walker.route[0], arrived=0.0) or (1.0, 0.0)
        walker.actor.apply_control(carla.WalkerControl(direction=carla.Vector3D(x=direction[0], y=direction[1]), speed=walker.speed))


def start_traffic(seed, vehicles, log):
    # Near the view, so that the traffic starts around the junction. Bikes are left in, so the model sees them.
    return subprocess.Popen(
        [sys.executable, "-u", "scripts/test_traffic.py", "--asynch", "--near-view", "-n", str(vehicles), "-w", "0",
         "--tm-port", str(TRAFFIC_MANAGER_PORT), "--seed", str(seed)],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)


def stop_traffic(process):
    if process.poll() is None:
        process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


def record_seed(carla, client, world, center, seed, sweeps):
    out = RAW_DIR / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    vehicles, pedestrians = traffic_level(seed)
    (out / "meta.json").write_text(json.dumps({
        "seed": seed, "vehicles": vehicles, "pedestrians": pedestrians, "mounts": LIDAR_MOUNTS,
        "points_per_second": LIDAR_POINTS_PER_SECOND, "seconds_apart": TICKS_PER_SWEEP * 0.1}))
    print(f"[RECORD] Seed {seed}: {vehicles} vehicles, {pedestrians} pedestrians, {sweeps} sweeps.", flush=True)

    walkers, lidars = [], []
    with open(out / "traffic.log", "w", encoding="utf-8", errors="replace") as log:
        traffic = start_traffic(seed, vehicles, log)
        try:
            deadline = time.time() + TRAFFIC_TIMEOUT
            while len(world.get_actors().filter("vehicle.*")) < 0.8 * vehicles:
                if traffic.poll() is not None or time.time() > deadline:
                    raise RuntimeError(f"The traffic didn't start. See {out / 'traffic.log'}.")
                world.wait_for_tick(10.0)
            walkers, spots, crossings, rng = spawn_walkers(carla, client, world, center, pedestrians, seed)
            for tick in range(WARM_UP_TICKS):
                snapshot = world.wait_for_tick(10.0)
                if tick % WALKER_STEER_TICKS == 0:
                    steer_walkers(carla, walkers, snapshot, spots, crossings, rng)
            lidars = spawn_lidars(carla, world, center)

            saved, late, ticks = 0, 0, 0
            while saved < sweeps:
                snapshot = world.wait_for_tick(10.0)
                ticks += 1
                if ticks % WALKER_STEER_TICKS == 0:
                    steer_walkers(carla, walkers, snapshot, spots, crossings, rng)
                if ticks % TICKS_PER_SWEEP:
                    continue
                points = merged_sweep(lidars, snapshot.frame, exact=True)
                if points is None:
                    late += 1
                    continue
                actors = [*world.get_actors().filter("vehicle.*"), *world.get_actors().filter("walker.pedestrian.*")]
                save_sweep(out / f"{snapshot.frame}.npz", points, labels_for(actors, snapshot, center, points))
                saved += 1
                if saved % 60 == 0:
                    print(f"[RECORD] Seed {seed}: {saved} of {sweeps} sweeps, {late} dropped for arriving late.", flush=True)
        finally:
            for lidar in lidars:
                lidar.sensor.stop()
                lidar.sensor.destroy()
            client.apply_batch_sync([carla.command.DestroyActor(w.actor.id) for w in walkers])
            stop_traffic(traffic)


def summarise(rows):
    """Totals from a seed's label rows, for telling whether the data covers what the model needs."""
    rows = np.asarray(rows, np.float32).reshape(-1, 11)
    out = {}
    for name, kind in (("vehicles", VEHICLE), ("pedestrians", PEDESTRIAN)):
        mine = rows[rows[:, 7] == kind]
        seen = mine[mine[:, 8] >= MIN_POINTS]
        moving = np.hypot(seen[:, 9], seen[:, 10]) > 0.5
        out[name] = {"in_grid": len(mine), "seen": len(seen), "moving_share": float(moving.mean()) if len(seen) else 0.0,
                     "median_points": float(np.median(seen[:, 8])) if len(seen) else 0.0}
    return out


def boxes_image(points, labels):
    """The top-down view with each seen actor's outline on, in the orientation of check_lidar_coverage.py."""
    from check_lidar_coverage import bev_image

    image = bev_image(points, half_size=GRID_HALF)
    size, cell = image.shape[0], 2 * GRID_HALF / image.shape[0]
    for row in labels[labels[:, 8] >= MIN_POINTS]:
        along, across = np.array([np.cos(row[6]), np.sin(row[6])]), np.array([-np.sin(row[6]), np.cos(row[6])])
        corners = [row[:2] + a * row[3] / 2 * along + b * row[4] / 2 * across for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1))]
        for start, end in itertools.pairwise(corners):
            for share in np.linspace(0.0, 1.0, 24):
                x, y = start + share * (end - start)
                r, c = size - 1 - int((x + GRID_HALF) / cell), int((GRID_HALF - y) / cell)
                if 0 <= r < size and 0 <= c < size:
                    image[r, c] = (60, 230, 90) if row[7] == VEHICLE else (70, 170, 255)
    return image


def check_seed(seed):
    from check_lidar_coverage import save_image

    files = sorted((RAW_DIR / f"seed_{seed}").glob("*.npz"))
    if not files:
        sys.exit(f"Nothing recorded for seed {seed}.")
    rows = [load_sweep(path)[1] for path in files]
    print(f"Seed {seed}: {len(files)} sweeps, {sum(p.stat().st_size for p in files) / 1e6:.0f} MB")
    print(json.dumps(summarise(np.vstack(rows)), indent=1))
    busiest = int(np.argmax([np.sum(r[:, 8] >= MIN_POINTS) for r in rows]))
    points, labels = load_sweep(files[busiest])
    save_image(boxes_image(points, labels), CHECK_DIR / f"labels_seed_{seed}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--sweeps", type=int, default=600, help="sweeps per seed")
    parser.add_argument("--check", type=int, help="summarise this seed's recording, with no CARLA running")
    args = parser.parse_args()
    if args.check is not None:
        check_seed(args.check)
        return

    # Without this, Ctrl+Break or a closed console skips the cleanup and leaves LiDARs slowing every tick
    for name in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _raise_keyboard_interrupt)

    import carla

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(60.0)
    world = client.get_world()
    if CARLA_MAP not in world.get_map().name or not world.get_settings().synchronous_mode:
        sys.exit("Start the processor first: it loads the map and owns the clock.")
    center = find_junction(world, INTERSECTION_JUNCTION).bounding_box.location
    for seed in args.seeds:
        record_seed(carla, client, world, center, seed, args.sweeps)
    print("[RECORD] Done.", flush=True)


if __name__ == "__main__":
    main()
