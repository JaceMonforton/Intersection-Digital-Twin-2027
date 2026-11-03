"""
Trajectory prediction node: keeps 1 s of history per CARLA actor, predicts the next 3 s in each
actor's own ego frame, and draws the result in the simulator. Attaches to the world run by
central_gpu_processor.py and follows its clock.

Without models/waymo_trajectory.onnx it extrapolates at constant velocity, which is a baseline
and not intent inference: it cannot anticipate a turn or a stop.

Usage:
    python scripts/waymo_predictor_node.py
"""
import argparse
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

# --- Configuration ---
CARLA_HOST = "localhost"
CARLA_PORT = 2000

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "waymo_trajectory.onnx"
HISTORY_FRAMES = 10   # 1.0 s at 10 Hz
FUTURE_STEPS = 30     # 3.0 s at 10 Hz
STEP_SECONDS = 0.1

DRAW_HEIGHT = 0.5     # metres above the actor, so lines don't z-fight with the road
DRAW_THICKNESS = 0.1
DRAW_LIFE_TIME = 0.11  # must outlive one 0.1 s tick: at exactly 0.1 a line expires before it renders


# ---------------------------------------------------------------------------
# Pure helpers (no CARLA needed)
# ---------------------------------------------------------------------------
class HistoryBuffer:
    """Rolling [x, y, vx, vy] history for each actor, oldest first."""

    def __init__(self, length=HISTORY_FRAMES):
        self.length = length
        self.buffers = {}

    def add(self, actor_id, x, y, vx, vy):
        buf = self.buffers.setdefault(actor_id, deque(maxlen=self.length))
        buf.append((x, y, vx, vy))

    def latest(self, actor_id):
        buf = self.buffers.get(actor_id)
        return buf[-1] if buf else None

    def retain(self, active_ids):
        """Forget actors that no longer exist, so reused ids don't inherit a stale history."""
        for actor_id in set(self.buffers) - set(active_ids):
            del self.buffers[actor_id]

    def ready(self):
        return {k: list(v) for k, v in self.buffers.items() if len(v) == self.length}


def world_to_ego(points, origin_x, origin_y, yaw_deg):
    """CARLA world (y right, left-handed) to ego frame (x forward, y LEFT), the right-handed
    convention the Waymo dataset uses. points is an array of [..., (x, y)]."""
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = points[..., 0] - origin_x, points[..., 1] - origin_y
    return np.stack((c * dx + s * dy, s * dx - c * dy), axis=-1)


def ego_to_world(points, origin_x, origin_y, yaw_deg):
    """Inverse of world_to_ego."""
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    fwd, left = points[..., 0], points[..., 1]
    return np.stack((origin_x + c * fwd + s * left, origin_y + s * fwd - c * left), axis=-1)


def history_to_ego(history, yaw):
    """Model input for one actor: HISTORY_FRAMES x (x, y, vx, vy) in its current ego frame."""
    data = np.asarray(history, dtype=np.float32)
    x, y = data[-1, :2]
    positions = world_to_ego(data[:, :2], x, y, yaw)
    velocities = world_to_ego(data[:, 2:4], 0.0, 0.0, yaw)  # vectors: rotate, don't translate
    return np.concatenate((positions, velocities), axis=1).astype(np.float32)


def constant_velocity(ego_histories, steps=FUTURE_STEPS, step_seconds=STEP_SECONDS):
    """Baseline predictor: N x steps x (x, y) from each actor's latest velocity."""
    velocity = ego_histories[:, -1, 2:4]
    times = step_seconds * np.arange(1, steps + 1, dtype=np.float32)
    return velocity[:, None, :] * times[None, :, None]


class OnnxPredictor:
    """See models/README.md for the expected model contract."""

    def __init__(self, path):
        import onnxruntime

        available = onnxruntime.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = onnxruntime.InferenceSession(str(path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, ego_histories):
        output = self.session.run(None, {self.input_name: ego_histories})[0]
        if output.shape[1:] != (FUTURE_STEPS, 2):
            raise ValueError(f"Model returned shape {output.shape}, expected (N, {FUTURE_STEPS}, 2).")
        return output


def measured_velocity(vx, vy, x, y, previous, dt):
    """CARLA's velocity, or the measured one for a twin moved with set_transform, which reports zero."""
    if previous is not None and dt > 0 and math.hypot(vx, vy) < 0.01:
        return (x - previous[0]) / dt, (y - previous[1]) / dt
    return vx, vy


def predict_world_paths(ready, yaws, predictor):
    """Maps actor id -> FUTURE_STEPS x (x, y) world path, for every actor with a full history."""
    if not ready:
        return {}
    ids = list(ready)
    futures = predictor(np.stack([history_to_ego(ready[i], yaws[i]) for i in ids]))
    paths = {}
    for actor_id, future in zip(ids, futures):
        x, y = ready[actor_id][-1][:2]
        paths[actor_id] = ego_to_world(future, x, y, yaws[actor_id])
    return paths


# ---------------------------------------------------------------------------
# CARLA node
# ---------------------------------------------------------------------------
def draw_paths(carla, world, actors, paths):
    color = carla.Color(255, 0, 0)
    for actor_id, path in paths.items():
        z = actors[actor_id].get_location().z + DRAW_HEIGHT
        begin = actors[actor_id].get_location()
        begin = carla.Location(x=begin.x, y=begin.y, z=z)
        for px, py in path:
            end = carla.Location(x=float(px), y=float(py), z=z)
            world.debug.draw_line(begin, end, thickness=DRAW_THICKNESS, color=color, life_time=DRAW_LIFE_TIME)
            begin = end


def run(args):
    import carla

    if Path(args.model).exists():
        predictor = OnnxPredictor(args.model)
        print(f"[PREDICT] Using model {args.model}")
    else:
        predictor = constant_velocity
        print(f"[PREDICT] {args.model} not found: using the CONSTANT-VELOCITY baseline, not Waymo inference.")

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(20.0)
    world = client.get_world()
    print(f"[CARLA] Connected to {world.get_map().name}. Ctrl+C to exit.")

    history = HistoryBuffer()
    last_sample = None
    last_log = 0.0
    while True:
        try:
            snapshot = world.wait_for_tick(5.0)
        except RuntimeError:
            print("[CARLA] No ticks arriving. Is central_gpu_processor.py running?")
            continue
        now = snapshot.timestamp.elapsed_seconds
        if last_sample is not None and now - last_sample < STEP_SECONDS - 1e-3:
            continue
        dt = now - last_sample if last_sample is not None else 0.0
        last_sample = now

        actors = {a.id: a for a in world.get_actors().filter("vehicle.*")}
        history.retain(actors)
        yaws = {}
        for actor_id, actor in actors.items():
            tf, v = actor.get_transform(), actor.get_velocity()
            x, y = tf.location.x, tf.location.y
            vx, vy = measured_velocity(v.x, v.y, x, y, history.latest(actor_id), dt)
            history.add(actor_id, x, y, vx, vy)
            yaws[actor_id] = tf.rotation.yaw

        started = time.perf_counter()
        paths = predict_world_paths(history.ready(), yaws, predictor)
        inference_ms = (time.perf_counter() - started) * 1000.0
        draw_paths(carla, world, actors, paths)

        if time.time() - last_log >= 1.0:
            last_log = time.time()
            print(f"[PREDICT] {len(paths)}/{len(actors)} actors predicted in {inference_ms:.1f} ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
