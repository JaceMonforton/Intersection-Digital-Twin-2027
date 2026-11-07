"""
Records images from a camera at each roadside mount with a label for every actor, to train the roadside camera
detector (scripts/camera_features.py). The traffic and the pedestrians around the junction are made as in
training/record_detection.py, seed for seed. Each label says how many of the image's pixels are that actor, read
from an instance segmentation camera at the same place, so an actor behind a lorry or a container counts as unseen.

Needs the processor running, and the main Python environment, which has carla. Keep whole seeds aside for testing.
    python training/record_camera.py --seeds 20 21 22 23 25 26 28 29 --frames 400
    python training/record_camera.py --check 20     # summarise a seed and save an image with its labels on
"""
import argparse
import json
import queue
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from camera_features import (
    FOV,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LABEL_SIZE,
    MIN_PIXELS,
    aim_at_centre,
    instance_ids,
    project,
)
from central_gpu_processor import (
    CARLA_HOST,
    CARLA_MAP,
    CARLA_PORT,
    INTERSECTION_JUNCTION,
    LIDAR_MOUNTS,
    find_junction,
)
from detection_features import GRID_HALF, PEDESTRIAN, label_row
from record_detection import (
    TICKS_PER_SWEEP,
    TRAFFIC_TIMEOUT,
    WALKER_STEER_TICKS,
    WARM_UP_TICKS,
    _raise_keyboard_interrupt,
    spawn_walkers,
    start_traffic,
    steer_walkers,
    stop_traffic,
    traffic_level,
)

RAW_DIR = ROOT / "data" / "camera" / "raw"
CHECK_DIR = ROOT / "data" / "camera" / "check"
IMAGE_TIMEOUT = 10.0


class Camera:
    """An RGB camera and an instance segmentation camera in the same place, queueing (frame, raw bytes)."""

    def __init__(self, carla, world, center, mount):
        pitch, yaw = aim_at_centre(mount)
        self.transform = carla.Transform(carla.Location(x=center.x + mount[0], y=center.y + mount[1], z=center.z + mount[2]),
                                         carla.Rotation(pitch=pitch, yaw=yaw))
        self.queues, self.sensors = [], []
        for kind in ("sensor.camera.rgb", "sensor.camera.instance_segmentation"):
            bp = world.get_blueprint_library().find(kind)
            for key, value in (("image_size_x", IMAGE_WIDTH), ("image_size_y", IMAGE_HEIGHT), ("fov", FOV)):
                bp.set_attribute(key, str(value))
            images = queue.Queue()
            sensor = world.spawn_actor(bp, self.transform)
            sensor.listen(lambda image, q=images: q.put((image.frame, bytes(image.raw_data))))
            self.queues.append(images)
            self.sensors.append(sensor)

    def matrices(self):
        return np.array(self.transform.get_matrix()), np.array(self.transform.get_inverse_matrix())

    def images_at(self, frame):
        """(rgb, instance) raw bytes from this frame, or None if a sensor has moved past it or doesn't answer."""
        out = []
        for images in self.queues:
            deadline = time.time() + IMAGE_TIMEOUT
            while True:
                try:
                    got, raw = images.get(timeout=max(deadline - time.time(), 0.01))
                except queue.Empty:
                    return None
                if got >= frame:
                    break
            if got != frame:
                return None
            out.append(raw)
        return out

    def destroy(self):
        for sensor in self.sensors:
            sensor.stop()
            sensor.destroy()


def camera_labels(actors, snapshot, center, world_to_camera, ids):
    """A label row (camera_features.LABEL_COLUMNS) for each actor in the LiDAR grid whose ground point is in front of
    the camera, with its pixel count from the instance ids."""
    found, counts = np.unique(ids, return_counts=True)
    pixel_counts = dict(zip(found.tolist(), counts.tolist()))
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
        pixel = project(np.array([[c.x, c.y, c.z - box.extent.z]]), world_to_camera)[0]
        if np.isnan(pixel).any():
            continue
        kind = PEDESTRIAN if actor.type_id.startswith("walker.") else 0
        row = label_row(offset, (box.extent.x, box.extent.y, box.extent.z), tf.rotation.yaw, (0.0, 0.0), kind)
        rows.append([pixel[0], pixel[1], *row[:8], pixel_counts.get(actor.id, 0)])
    return np.array(rows, np.float32).reshape(-1, LABEL_SIZE)


def rgb(raw):
    return np.frombuffer(raw, np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 4)[:, :, 2::-1].copy()


def record_seed(carla, client, world, center, seed, frames):
    out = RAW_DIR / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    vehicles, pedestrians = traffic_level(seed)
    print(f"[RECORD] Seed {seed}: {vehicles} vehicles, {pedestrians} pedestrians, {frames} frames from each camera.", flush=True)
    walkers, cameras = [], []
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
            cameras = [Camera(carla, world, center, mount) for mount in LIDAR_MOUNTS]
            (out / "meta.json").write_text(json.dumps({
                "seed": seed, "vehicles": vehicles, "pedestrians": pedestrians, "mounts": LIDAR_MOUNTS,
                "center": [center.x, center.y, center.z], "seconds_apart": TICKS_PER_SWEEP * 0.1,
                "cameras": [{"camera_to_world": m.tolist(), "world_to_camera": i.tolist()} for m, i in (c.matrices() for c in cameras)]}))
            saved, late, ticks = 0, 0, 0
            while saved < frames:
                snapshot = world.wait_for_tick(10.0)
                ticks += 1
                if ticks % WALKER_STEER_TICKS == 0:
                    steer_walkers(carla, walkers, snapshot, spots, crossings, rng)
                if ticks % TICKS_PER_SWEEP:
                    continue
                images = [camera.images_at(snapshot.frame) for camera in cameras]
                if any(i is None for i in images):
                    late += 1
                    continue
                actors = [*world.get_actors().filter("vehicle.*"), *world.get_actors().filter("walker.pedestrian.*")]
                for index, (camera, (raw, instance)) in enumerate(zip(cameras, images)):
                    labels = camera_labels(actors, snapshot, center, camera.matrices()[1], instance_ids(instance))
                    np.savez_compressed(out / f"{snapshot.frame}_{index}.npz", image=rgb(raw), labels=labels)
                saved += 1
                if saved % 50 == 0:
                    print(f"[RECORD] Seed {seed}: {saved} of {frames} frames, {late} dropped for arriving late.", flush=True)
        finally:
            for camera in cameras:
                camera.destroy()
            client.apply_batch_sync([carla.command.DestroyActor(w.actor.id) for w in walkers])
            stop_traffic(traffic)


def check_seed(seed):
    """Totals for a seed, and its busiest image with each label's ground point marked (green seen, red barely)."""
    import pygame

    files = sorted((RAW_DIR / f"seed_{seed}").glob("*.npz"))
    if not files:
        sys.exit(f"Nothing recorded for seed {seed}.")
    labels = [np.load(path)["labels"] for path in files]
    rows = np.vstack(labels)
    for name, kind in (("vehicles", 0), ("pedestrians", PEDESTRIAN)):
        mine = rows[rows[:, 9] == kind]
        print(f"{name}: {len(mine)} labels in front of a camera, {int((mine[:, 10] >= MIN_PIXELS).sum())} seen, "
              f"{int(((mine[:, 10] > 0) & (mine[:, 10] < MIN_PIXELS)).sum())} barely")
    print(f"{len(files)} images, {sum(p.stat().st_size for p in files) / 1e6:.0f} MB")
    busiest = int(np.argmax([np.sum(r[:, 10] >= MIN_PIXELS) for r in labels]))
    data = np.load(files[busiest])
    image = data["image"].copy()
    for row in data["labels"]:
        u, v = int(row[0]), int(row[1])
        if 0 <= u < IMAGE_WIDTH and 0 <= v < IMAGE_HEIGHT and row[10] > 0:
            image[max(v - 2, 0):v + 3, max(u - 2, 0):u + 3] = (60, 230, 90) if row[10] >= MIN_PIXELS else (230, 60, 60)
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECK_DIR / f"labels_seed_{seed}.png"
    pygame.image.save(pygame.surfarray.make_surface(image.swapaxes(0, 1)), str(path))
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20, 21, 22, 23, 25, 26, 28, 29])
    parser.add_argument("--frames", type=int, default=400, help="frames per seed, each an image from every camera")
    parser.add_argument("--check", type=int, help="summarise this seed's recording, with no CARLA running")
    args = parser.parse_args()
    if args.check is not None:
        check_seed(args.check)
        return
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
        record_seed(carla, client, world, center, seed, args.frames)
    print("[RECORD] Done.", flush=True)


if __name__ == "__main__":
    main()
