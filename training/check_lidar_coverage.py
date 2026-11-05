"""
Shows what the roadside LiDARs can see, to choose their corners and check them afterwards.

By default it casts a ray from each candidate corner to the roads leading through the junction, and compares
each pair of corners. That only reads the map, so it needs CARLA but not the processor. With --bev it
spawns the real LiDARs at LIDAR_MOUNTS and saves one merged sweep as the model's top-down view, along with
the time each tick takes without and with them. That needs the processor's clock, and it's better with
test_traffic.py running, so that there are cars in the picture. It also saves the view from each mount.

--probe is the real test. It stands one car or person on each road point in turn and counts the LiDAR
points that land on it, which needs the processor's clock and an otherwise empty junction.

    python training/check_lidar_coverage.py --margin -5
    python training/check_lidar_coverage.py --bev
    python training/check_lidar_coverage.py --probe
    python training/check_lidar_coverage.py --probe --actor walker.pedestrian.0001

Both images have x up and CARLA's y to the right, the same way up as the overhead view.
"""
import argparse
import itertools
import math
import queue
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from central_gpu_processor import (
    CARLA_HOST,
    CARLA_MAP,
    CARLA_PORT,
    INTERSECTION_JUNCTION,
    LIDAR_MOUNTS,
    find_junction,
    merged_sweep,
    spawn_lidars,
)
from detection_features import ABOVE_ROAD, MIN_POINTS, MODEL_GROUND_Z, points_in_box
from ml_car_view import bgra_to_rgb

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "lidar_check"
RADIUS = 60.0        # m around the junction centre shown in the images
APPROACH = 45.0      # m of road either side of the junction to cover: 3 s at 14 m/s, for the path check
ROAD_STEP = 2.0      # m between road points
TARGET_HEIGHT = 1.4  # m above the road: about a car's roof, the part a raised sensor sees first
NEAR_TARGET = 0.5    # m; a hit this close to the target is the target's own surface, not something in the way
COVERAGE_CELL = 1.0  # m per pixel in the coverage map, before scaling up
BEV_CELL = 0.25      # m per pixel in the top-down view
SCALE = 6            # the coverage map's screen pixels per cell
TICKS_TIMED = 30


def corner_mounts(half_x, half_y, margin, height):
    """The four corners outside a junction this size, as (x, y, z) from its centre, in CARLA's frame."""
    x, y = half_x + margin, half_y + margin
    return {"NE": (x, y, height), "SE": (-x, y, height), "SW": (-x, -y, height), "NW": (x, -y, height)}


def is_clear(hits, target, near=NEAR_TARGET):
    """Whether a ray's hits, as (x, y, z), include nothing short of the target itself."""
    return all(np.linalg.norm(np.subtract(hit, target)) <= near for hit in hits)


def rank_pairs(seen):
    """Every pair of corners as (share of road seen by either, share seen by both, names), best first.
    seen is corner name -> a bool for each road point."""
    pairs = [(float(np.mean(seen[a] | seen[b])), float(np.mean(seen[a] & seen[b])), (a, b))
             for a, b in itertools.combinations(seen, 2)]
    return sorted(pairs, key=lambda p: p[:2], reverse=True)


def cell_counts(xs, ys, half_size, cell):
    """How many of the points fall in each cell of a square around the centre. Rows run with x down the
    image from the top, and columns with CARLA's y, which is minus the model's y."""
    size = round(2 * half_size / cell)
    edges = np.linspace(-half_size, half_size, size + 1)
    counts, _, _ = np.histogram2d(xs, ys, bins=(edges, edges))
    return counts[::-1]


def bev_image(points, half_size=RADIUS, cell=BEV_CELL, ground_z=MODEL_GROUND_Z):
    """A cloud in the model's frame as an RGB image: the road in grey, anything above it in orange."""
    above = points[:, 2] > ground_z + ABOVE_ROAD
    road = cell_counts(points[~above, 0], -points[~above, 1], half_size, cell)
    objects = cell_counts(points[above, 0], -points[above, 1], half_size, cell)
    image = np.zeros((*road.shape, 3), dtype=np.uint8)
    image[road > 0] = (90, 90, 90)
    image[objects > 0] = (255, 150, 30)
    return image


def coverage_image(road, seen_a, seen_b, half_size=RADIUS, cell=COVERAGE_CELL):
    """Road points (x, y from the centre) in green if both corners see them, yellow if one, red if neither."""
    image = np.zeros((round(2 * half_size / cell),) * 2 + (3,), dtype=np.uint8)
    colours = ((220, 40, 40), (230, 200, 40), (40, 180, 70))
    for (x, y), a, b in zip(road, seen_a, seen_b):
        row = np.clip(image.shape[0] - 1 - int((half_size + x) / cell), 0, image.shape[0] - 1)
        column = np.clip(int((half_size + y) / cell), 0, image.shape[1] - 1)
        image[row, column] = colours[int(a) + int(b)]
    return image.repeat(SCALE, axis=0).repeat(SCALE, axis=1)


def save_image(image, path):
    import pygame

    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(pygame.surfarray.make_surface(image.transpose(1, 0, 2)), str(path))
    print(f"Saved {path}")


def approach_roads(carla, junction):
    """(x, y from the centre, target location) along the lanes through the junction and APPROACH metres
    of road either side, which is what the car needs watched. Not every road nearby: parallel streets
    behind the buildings are out of sight of any corner, and don't matter here.
    shortcut: driving lanes only, so pavements aren't checked; add them for pedestrian detection in Phase 2."""
    center = junction.bounding_box.location
    found = {}

    def add(waypoint):
        location = waypoint.transform.location
        location.z += TARGET_HEIGHT
        x, y = location.x - center.x, location.y - center.y
        found[(round(x / ROAD_STEP), round(y / ROAD_STEP))] = ((x, y), location)

    for entry, leaving in junction.get_waypoints(carla.LaneType.Driving):
        for waypoint in entry.next_until_lane_end(ROAD_STEP):
            add(waypoint)
        for start, walk in ((entry, "previous"), (leaving, "next")):
            frontier = [start]
            for _ in range(round(APPROACH / ROAD_STEP)):
                frontier = [following for waypoint in frontier for following in getattr(waypoint, walk)(ROAD_STEP)]
                for waypoint in frontier:
                    add(waypoint)
    return list(found.values())


def compare_corners(carla, world, junction, margin, height):
    center, extent = junction.bounding_box.location, junction.bounding_box.extent
    print(f"Junction centre x={center.x:.1f} y={center.y:.1f} z={center.z:.1f}, half-size {extent.x:.1f} x {extent.y:.1f} m")
    road = approach_roads(carla, junction)
    seen = {}
    for name, (x, y, z) in corner_mounts(extent.x, extent.y, margin, height).items():
        mount = carla.Location(x=center.x + x, y=center.y + y, z=center.z + z)
        clear, blockers = [], Counter()
        for _, target in road:
            hits = world.cast_ray(mount, target)
            ok = is_clear([(h.location.x, h.location.y, h.location.z) for h in hits], (target.x, target.y, target.z))
            clear.append(ok)
            if not ok:
                blockers[str(hits[0].label)] += 1
        seen[name] = np.array(clear)
        below = [str(h.label) for h in world.cast_ray(mount, carla.Location(x=mount.x, y=mount.y, z=center.z - 1.0))]
        print(f"{name} at ({x:+.1f}, {y:+.1f}, {z:.1f}), above {below}: sees {seen[name].mean():6.1%}. First blocked by: {blockers.most_common(3)}")

    print(f"\n{len(road)} road points. Either = seen by at least one, both = seen by the two.")
    ranked = rank_pairs(seen)
    for either, both, names in ranked:
        print(f"{names[0]} + {names[1]}: either {either:6.1%}, both {both:6.1%}, "
              f"mounts {corner_mounts(extent.x, extent.y, margin, height)[names[0]]} {corner_mounts(extent.x, extent.y, margin, height)[names[1]]}")
    best = ranked[0][2]
    out = OUT_DIR / f"coverage_margin{margin:g}_height{height:g}.png"
    save_image(coverage_image([xy for xy, _ in road], seen[best[0]], seen[best[1]]), out)


def tick_seconds(world, ticks=TICKS_TIMED):
    world.wait_for_tick(10.0)
    started = time.perf_counter()
    for _ in range(ticks):
        world.wait_for_tick(10.0)
    return (time.perf_counter() - started) / ticks


def capture_bev(carla, world, junction):
    center = junction.bounding_box.location
    before = tick_seconds(world)
    lidars = spawn_lidars(carla, world, center)
    try:
        after = tick_seconds(world)
        points = merged_sweep(lidars, world.wait_for_tick(10.0).frame)
    finally:
        for lidar in lidars:
            lidar.sensor.stop()
            lidar.sensor.destroy()
    if points is None:
        raise RuntimeError("A LiDAR sweep didn't arrive.")
    print(f"Mounts {LIDAR_MOUNTS}\n{len(points)} points in the merged sweep. Each tick took {before * 1000:.0f} ms "
          f"without the LiDARs and {after * 1000:.0f} ms with them.")
    save_image(bev_image(points), OUT_DIR / "bev.png")
    for number, mount in enumerate(LIDAR_MOUNTS, 1):
        save_image(view_from(carla, world, center, mount), OUT_DIR / f"view_from_lidar_{number}.png")


def view_from(carla, world, center, mount):
    """What a mount can see, looking at the junction centre, as an RGB image."""
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "1024")
    bp.set_attribute("image_size_y", "576")
    bp.set_attribute("fov", "110")
    place = carla.Location(x=center.x + mount[0], y=center.y + mount[1], z=center.z + mount[2])
    toward = carla.Rotation(pitch=-15.0, yaw=math.degrees(math.atan2(-mount[1], -mount[0])))
    camera = world.spawn_actor(bp, carla.Transform(place, toward))
    images = queue.Queue()
    # Copied here: CARLA frees an image's buffer once its camera is gone, and reading it then crashes
    camera.listen(lambda image: images.put(bgra_to_rgb(image.raw_data, image.width, image.height).copy()))
    try:
        for _ in range(5):  # the first frames can still be blank
            world.wait_for_tick(10.0)
        return images.get(timeout=10.0)
    finally:
        camera.stop()
        camera.destroy()


def probe_actor(carla, world, junction, blueprint):
    center = junction.bounding_box.location
    road = approach_roads(carla, junction)
    world.wait_for_tick(10.0)
    above = carla.Transform(carla.Location(x=center.x, y=center.y, z=center.z + 60.0))
    actor = world.spawn_actor(world.get_blueprint_library().find(blueprint), above)
    actor.set_simulate_physics(False)
    lidars = spawn_lidars(carla, world, center)
    counts = []
    try:
        box = actor.bounding_box
        for number, ((x, y), target) in enumerate(road, 1):
            yaw = world.get_map().get_waypoint(target).transform.rotation.yaw
            z = target.z - TARGET_HEIGHT + box.extent.z - box.location.z
            actor.set_transform(carla.Transform(carla.Location(x=target.x, y=target.y, z=z), carla.Rotation(yaw=yaw)))
            world.wait_for_tick(10.0)  # the move lands on this tick, so the next sweep is the first to see it
            frame = world.wait_for_tick(10.0).frame
            row = []
            for lidar in lidars:
                points = merged_sweep([lidar], frame)
                if points is None:
                    raise RuntimeError("A LiDAR sweep didn't arrive.")
                row.append(points_in_box(points, x, -y, -math.radians(yaw), box.extent.x, box.extent.y, 2 * box.extent.z))
            counts.append(row)
            if number % 100 == 0:
                print(f"{number} of {len(road)} road points...", flush=True)
    finally:
        for lidar in lidars:
            lidar.sensor.stop()
            lidar.sensor.destroy()
        actor.destroy()
    counts = np.array(counts)
    seen = counts >= MIN_POINTS
    total = counts.sum(axis=1)
    print(f"{blueprint} at {len(road)} road points, with mounts {LIDAR_MOUNTS}. Seen = at least {MIN_POINTS} LiDAR points.")
    print(f"Seen by either {seen.any(axis=1).mean():.1%}, by both {seen.all(axis=1).mean():.1%}, each {seen.mean(axis=0).round(3).tolist()}")
    print(f"Points on it, both LiDARs together: median {np.median(total):.0f}, 10th percentile {np.percentile(total, 10):.0f}, max {total.max()}")
    within = np.array([math.hypot(*xy) <= 30.0 for xy, _ in road])
    print(f"Within 30 m of the centre: seen by either {seen[within].any(axis=1).mean():.1%}, median {np.median(total[within]):.0f} points")
    out = OUT_DIR / f"probe_{blueprint.replace('.', '_')}.png"
    save_image(coverage_image([xy for xy, _ in road], seen[:, 0], seen[:, 1]), out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--margin", type=float, default=-5.0, help="metres outside the junction's box; it reaches past the street corners, so negative")
    parser.add_argument("--height", type=float, default=7.0, help="metres above the junction centre")
    parser.add_argument("--bev", action="store_true", help="spawn the real LiDARs and save a top-down view")
    parser.add_argument("--probe", action="store_true", help="count the LiDAR points on an actor at each road point")
    parser.add_argument("--actor", default="vehicle.tesla.model3", help="blueprint for --probe")
    args = parser.parse_args()

    import carla

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(60.0)
    world = client.get_world()
    if CARLA_MAP not in world.get_map().name:
        sys.exit(f"CARLA has {world.get_map().name} loaded, not {CARLA_MAP}. Start the processor, which loads it.")
    junction = find_junction(world, INTERSECTION_JUNCTION)
    if args.bev:
        capture_bev(carla, world, junction)
    elif args.probe:
        probe_actor(carla, world, junction, args.actor)
    else:
        compare_corners(carla, world, junction, args.margin, args.height)


if __name__ == "__main__":
    main()
