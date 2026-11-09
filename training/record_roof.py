"""
Records roof LiDAR sweeps with the true box of every actor, to train the ML car's own detector: what a Waymo-style
car sees from its roof, rather than which actors CARLA says its rays hit.

A few cars on autopilot each carry the roof LiDAR the ML car will (roof_features.ROOF_LIDAR) through the junction's
traffic and pedestrians, which are set up as for the roadside recording (record_detection.py). Autopilot wanders off,
so they're put back near the junction every RESPAWN_SWEEPS sweeps. Each sweep is saved in the car-centred model
frame with its labels, and only if it's from the same frame as the labels.

Seeds 40 to 45 train, 48 picks the best epoch and the thresholds, and 49 is kept for the final measurement. Needs the
processor running (it owns the clock); nothing else may be using CARLA.
    python training/record_roof.py --seeds 40 41 42 43 44 45 48 49
"""
import argparse
import math
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from central_gpu_processor import (
    CARLA_HOST,
    CARLA_MAP,
    CARLA_PORT,
    INTERSECTION_JUNCTION,
    TRAFFIC_MANAGER_PORT,
    find_junction,
)
from detection_features import count_points
from record_detection import (
    TRAFFIC_TIMEOUT,
    WALKER_STEER_TICKS,
    WARM_UP_TICKS,
    save_sweep,
    spawn_walkers,
    start_traffic,
    steer_walkers,
    stop_traffic,
    traffic_level,
)
from roof_features import (
    ROOF_HEIGHT,
    ROOF_LIDAR,
    ROOF_RAW_DIR_NAME,
    actor_label,
    sweep_to_model,
)

RAW_DIR = ROOT / "data" / "detection" / ROOF_RAW_DIR_NAME
SENSOR_CARS = 3
TICKS_PER_SWEEP = 3     # 0.3 s apart: each LiDAR sweeps every tick anyway, and the cars move enough in 0.3 s
RESPAWN_SWEEPS = 20     # sweeps before the sensor cars are put back near the junction
SPAWN_REACH = 90.0      # m from the junction centre that a sensor car may start
SWEEP_TIMEOUT = 1.0     # s to wait for a sweep from the tick's frame
TRAIN_SEEDS, VALIDATION_SEED, TEST_SEED = (40, 41, 42, 43, 44, 45), 48, 49


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


class SensorCar:
    """An autopilot car with a roof LiDAR, holding the newest sweep."""

    def __init__(self, carla, world, place, rng):
        bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        bp.set_attribute("role_name", "roof_recorder")
        self.car = world.try_spawn_actor(bp, place)
        self.lidar, self.latest = None, None
        if self.car is None:
            return
        self.car.set_autopilot(True, TRAFFIC_MANAGER_PORT)
        lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
        for key, value in ROOF_LIDAR.items():
            lidar_bp.set_attribute(key, value)
        # A random spin phase, as a real sensor holds none, so the model doesn't learn one scan pattern
        mount = carla.Transform(carla.Location(z=ROOF_HEIGHT), carla.Rotation(yaw=rng.uniform(0.0, 360.0)))
        self.lidar = world.spawn_actor(lidar_bp, mount, attach_to=self.car)
        self.lidar.listen(self._keep)

    def _keep(self, measurement):
        self.latest = measurement

    def sweep_at(self, frame, timeout=SWEEP_TIMEOUT):
        deadline = time.time() + timeout
        while time.time() < deadline:
            m = self.latest
            if m is not None and m.frame >= frame:
                return m if m.frame == frame else None
            time.sleep(0.005)
        return None

    def stop(self):
        if self.lidar is not None:
            self.lidar.stop()


def destroy_cars(carla, client, cars):
    """In one synchronous batch, as one at a time once left a set of LiDARs behind, slowing every tick."""
    for car in cars:
        car.stop()
    client.apply_batch_sync([carla.command.DestroyActor(a.id) for car in cars for a in (car.lidar, car.car) if a is not None])


def spawn_sensor_cars(carla, world, center, rng, count=SENSOR_CARS):
    places = [p for p in world.get_map().get_spawn_points() if math.hypot(p.location.x - center.x, p.location.y - center.y) < SPAWN_REACH]
    rng.shuffle(places)
    cars = []
    for place in places:
        car = SensorCar(carla, world, place, rng)
        if car.car is not None:
            cars.append(car)
        if len(cars) == count:
            break
    return cars


def actor_rows(sensor_tf, actors, snapshot, own_id, points):
    sensor = (sensor_tf.location.x, sensor_tf.location.y, sensor_tf.location.z, sensor_tf.rotation.yaw)
    rows = []
    for actor in actors:
        state = snapshot.find(actor.id)
        if state is None or actor.id == own_id:
            continue
        tf, box, v = state.get_transform(), actor.bounding_box, state.get_velocity()
        c = tf.transform(box.location)
        row = actor_label(sensor, (c.x, c.y, c.z, box.extent.x, box.extent.y, box.extent.z, tf.rotation.yaw, v.x, v.y,
                                   actor.type_id.startswith("walker.")))
        if row is not None:
            row[8] = count_points(row, points)
            rows.append(row)
    return rows


def record_seed(carla, client, world, center, seed, sweeps):
    out = RAW_DIR / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    vehicles, pedestrians = traffic_level(seed)
    print(f"[RECORD] Seed {seed}: {vehicles} vehicles, {pedestrians} pedestrians, {sweeps} sweeps from {SENSOR_CARS} roof LiDARs.", flush=True)
    rng = random.Random(seed)
    walkers, cars = [], []
    with open(out / "traffic.log", "w", encoding="utf-8", errors="replace") as log:
        traffic = start_traffic(seed, vehicles, log)
        try:
            deadline = time.time() + TRAFFIC_TIMEOUT
            while len(world.get_actors().filter("vehicle.*")) < 0.8 * vehicles:
                if traffic.poll() is not None or time.time() > deadline:
                    raise RuntimeError(f"The traffic didn't start. See {out / 'traffic.log'}.")
                world.wait_for_tick(10.0)
            # Pedestrians and their controllers before any LiDAR: starting them with LiDARs running hung the server
            walkers, spots, crossings, walk_rng = spawn_walkers(carla, client, world, center, pedestrians, seed)
            for tick in range(WARM_UP_TICKS):
                snapshot = world.wait_for_tick(10.0)
                if tick % WALKER_STEER_TICKS == 0:
                    steer_walkers(carla, walkers, snapshot, spots, crossings, walk_rng)
            cars = spawn_sensor_cars(carla, world, center, rng)
            saved, late, ticks, rounds = 0, 0, 0, 0
            while saved < sweeps:
                snapshot = world.wait_for_tick(10.0)
                ticks += 1
                if ticks % WALKER_STEER_TICKS == 0:
                    steer_walkers(carla, walkers, snapshot, spots, crossings, walk_rng)
                if ticks % TICKS_PER_SWEEP:
                    continue
                actors = [*world.get_actors().filter("vehicle.*"), *world.get_actors().filter("walker.pedestrian.*")]
                for car in cars:
                    measurement = car.sweep_at(snapshot.frame)
                    if measurement is None:
                        late += 1
                        continue
                    points = sweep_to_model(np.frombuffer(measurement.raw_data, dtype=np.float32))
                    save_sweep(out / f"{snapshot.frame}_{car.car.id}.npz", points,
                               actor_rows(measurement.transform, actors, snapshot, car.car.id, points))
                    saved += 1
                rounds += 1
                if rounds % RESPAWN_SWEEPS == 0 and saved < sweeps:
                    destroy_cars(carla, client, cars)
                    cars = spawn_sensor_cars(carla, world, center, rng)
                if rounds % 60 == 0:
                    lidars = len(world.get_actors().filter("sensor.lidar.*"))
                    print(f"[RECORD] Seed {seed}: {saved} of {sweeps} sweeps, {late} dropped for arriving late, {lidars} LiDARs in the world.", flush=True)
        finally:
            destroy_cars(carla, client, cars)
            client.apply_batch_sync([carla.command.DestroyActor(w.actor.id) for w in walkers])
            stop_traffic(traffic)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[*TRAIN_SEEDS, VALIDATION_SEED, TEST_SEED])
    parser.add_argument("--sweeps", type=int, default=900, help="sweeps per seed, from all the sensor cars together")
    args = parser.parse_args()
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
