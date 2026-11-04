"""
Trajectory prediction node: keeps 1 s of history per CARLA actor, predicts the next 3 s in each
actor's own ego frame, and draws the result in the simulator. Attaches to the world run by
central_gpu_processor.py and follows its clock.

Without models/waymo_trajectory.onnx it follows the road instead, at the vehicle's current speed:
along its lane, taking the branch its turn signal or turn rate points to, or along its current
curve when it isn't on a lane. That is a baseline and not learned inference: it cannot anticipate
a stop, or a turn nothing has signalled yet.

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
DRAW_COLOR = (70, 0, 0)  # debug lines are unlit, so full brightness blooms at high render quality
DRAW_COLOR_ML = (90, 0, 90)  # the ML car's line matches its magenta paint and box
DRAW_EVERY = 3         # draw every Nth step of other cars' lines; each one is a call to the server
DRAW_LIFE_TIME = 0.11  # must outlive one 0.1 s tick: at exactly 0.1 a line expires before it renders


LANE_STEP = 1.0        # metres between waypoints when walking a lane
BRANCH_LOOKAHEAD = 15  # waypoints walked down a branch to see where it heads
TURN_RATE_INTENT = 8.0 # deg/s; turning faster than this shows which way a vehicle is going
YAW_HISTORY = 5        # samples used to measure the turn rate
LANE_MAX_OFFSET = 3.0  # metres from a lane's centre to still count as on that lane
LANE_MAX_HEADING = 45.0  # degrees from a lane's direction to still count as following it
OTHER_TURN_PENALTY = 2.0  # lane score penalty when a lane doesn't turn the way the vehicle intends
LEFT, STRAIGHT, RIGHT = -1, 0, 1  # the sign of the yaw change in CARLA's frame


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


def yaw_difference(a, b):
    """b - a in degrees, wrapped to (-180, 180]. Negative is a left turn in CARLA's frame."""
    return (b - a + 180.0) % 360.0 - 180.0


def turn_rate(yaws, step_seconds=STEP_SECONDS):
    """Degrees per second over a list of recent yaws, oldest first."""
    if len(yaws) < 2:
        return 0.0
    return yaw_difference(yaws[0], yaws[-1]) / (step_seconds * (len(yaws) - 1))


def turn_intent(signal, rate):
    """Which way a vehicle is going next: its turn signal, or else the way it's already turning."""
    if signal != STRAIGHT:
        return signal
    if abs(rate) > TURN_RATE_INTENT:
        return LEFT if rate < 0 else RIGHT
    return STRAIGHT


def turn_class(yaw_change):
    if yaw_change < -45.0:
        return LEFT
    return RIGHT if yaw_change > 45.0 else STRAIGHT


def pick_branch(yaw_changes, intent):
    """Index of the branch to follow, given how far each one turns. The sharpest turn that way
    if the vehicle intends one and there is one; otherwise the straightest."""
    if intent != STRAIGHT:
        matching = [i for i, change in enumerate(yaw_changes) if turn_class(change) == intent]
        if matching:
            return max(matching, key=lambda i: abs(yaw_changes[i]))
    return min(range(len(yaw_changes)), key=lambda i: abs(yaw_changes[i]))


def curve_path(x, y, yaw_deg, speed, rate, steps=FUTURE_STEPS, step_seconds=STEP_SECONDS):
    """steps x (x, y) holding the current speed and turn rate, for a vehicle not on a lane."""
    points = np.zeros((steps, 2))
    yaw = yaw_deg
    for i in range(steps):
        yaw += rate * step_seconds
        x += speed * step_seconds * math.cos(math.radians(yaw))
        y += speed * step_seconds * math.sin(math.radians(yaw))
        points[i] = (x, y)
    return points


def along_polyline(points, speed, steps=FUTURE_STEPS, step_seconds=STEP_SECONDS):
    """steps x (x, y) travelling a polyline at constant speed, stopping where it ends."""
    points = np.asarray(points, dtype=float)
    segments = np.diff(points, axis=0)
    distance = np.concatenate(([0.0], np.cumsum(np.hypot(segments[:, 0], segments[:, 1]))))
    travelled = speed * step_seconds * np.arange(1, steps + 1)
    return np.stack((np.interp(travelled, distance, points[:, 0]),
                     np.interp(travelled, distance, points[:, 1])), axis=-1)


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
class LaneFollower:
    """The road baseline: where a vehicle goes if it stays on its lane and keeps its speed."""

    def __init__(self, carla, world_map):
        self.carla = carla
        self.map = world_map
        self.junction_lanes = {}  # junction id -> [(turn class, [waypoints])]

    def _lanes_of(self, junction):
        if junction.id not in self.junction_lanes:
            lanes = []
            for entry, exit_wp in junction.get_waypoints(self.carla.LaneType.Driving):
                change = yaw_difference(entry.transform.rotation.yaw, exit_wp.transform.rotation.yaw)
                lanes.append((turn_class(change), [entry, *entry.next_until_lane_end(LANE_STEP)]))
            self.junction_lanes[junction.id] = lanes
        return self.junction_lanes[junction.id]

    def _start(self, x, y, yaw, intent):
        """The waypoint this vehicle is on, or None if it isn't following a lane. Lanes overlap
        inside a junction, so the nearest one isn't enough there: score them by fit and intent."""
        here = self.carla.Location(x=x, y=y, z=0.0)
        wp = self.map.get_waypoint(here, project_to_road=True, lane_type=self.carla.LaneType.Driving)
        if wp is None:
            return None
        candidates = [(turn_class(0.0), [wp])]
        if wp.is_junction:
            candidates = self._lanes_of(wp.get_junction())
        best, best_score = None, math.inf
        for lane_class, waypoints in candidates:
            for candidate in waypoints:
                location = candidate.transform.location
                offset = math.hypot(location.x - x, location.y - y)
                heading = abs(yaw_difference(candidate.transform.rotation.yaw, yaw))
                if offset > LANE_MAX_OFFSET or heading > LANE_MAX_HEADING:
                    continue
                score = offset + heading / 30.0
                if wp.is_junction and lane_class != intent:
                    score += OTHER_TURN_PENALTY
                if score < best_score:
                    best, best_score = candidate, score
        return best

    def _branch_change(self, wp):
        end = wp
        for _ in range(BRANCH_LOOKAHEAD):
            following = end.next(LANE_STEP)
            if not following:
                break
            end = following[0]
        return end.transform.rotation.yaw

    def path(self, x, y, yaw, speed, intent):
        """FUTURE_STEPS x (x, y), or None if the vehicle isn't on a lane."""
        wp = self._start(x, y, yaw, intent)
        if wp is None:
            return None
        points = [(x, y)]
        needed = speed * STEP_SECONDS * FUTURE_STEPS
        walked = 0.0
        while walked < needed:
            following = wp.next(LANE_STEP)
            if not following:
                break
            if len(following) > 1:
                heading = wp.transform.rotation.yaw
                changes = [yaw_difference(heading, self._branch_change(f)) for f in following]
                wp = following[pick_branch(changes, intent)]
                intent = STRAIGHT  # the signal was for this junction, not the next one
            else:
                wp = following[0]
            points.append((wp.transform.location.x, wp.transform.location.y))
            walked += LANE_STEP
        return along_polyline(points, speed)


def signalled_turn(carla, actor):
    try:
        lights = int(actor.get_light_state())
    except RuntimeError:  # destroyed since this tick's actor list was taken
        return STRAIGHT
    if lights & int(carla.VehicleLightState.LeftBlinker):
        return LEFT
    if lights & int(carla.VehicleLightState.RightBlinker):
        return RIGHT
    return STRAIGHT


def baseline_paths(carla, lanes, actors, ready, yaws, recent_yaws):
    paths = {}
    for actor_id, history in ready.items():
        x, y, vx, vy = history[-1]
        speed = math.hypot(vx, vy)
        rate = turn_rate(list(recent_yaws[actor_id]))
        intent = turn_intent(signalled_turn(carla, actors[actor_id]), rate)
        path = lanes.path(x, y, yaws[actor_id], speed, intent)
        paths[actor_id] = path if path is not None else curve_path(x, y, yaws[actor_id], speed, rate)
    return paths


def draw_paths(carla, world, actors, paths):
    for actor_id, path in paths.items():
        is_ml = actors[actor_id].attributes.get("role_name") == "vehicle_ml"
        color = carla.Color(*(DRAW_COLOR_ML if is_ml else DRAW_COLOR))
        z = actors[actor_id].get_location().z + DRAW_HEIGHT
        begin = actors[actor_id].get_location()
        begin = carla.Location(x=begin.x, y=begin.y, z=z)
        for px, py in (path if is_ml else path[DRAW_EVERY - 1::DRAW_EVERY]):
            end = carla.Location(x=float(px), y=float(py), z=z)
            world.debug.draw_line(begin, end, thickness=DRAW_THICKNESS, color=color, life_time=DRAW_LIFE_TIME)
            begin = end


def run(args):
    import carla

    if Path(args.model).exists():
        predictor = OnnxPredictor(args.model)
        print(f"[PREDICT] Using model {args.model}")
    else:
        predictor = None
        print(f"[PREDICT] {args.model} not found: using the LANE-FOLLOWING baseline, not Waymo inference.")

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(20.0)
    world = client.get_world()
    print(f"[CARLA] Connected to {world.get_map().name}. Ctrl+C to exit.")

    lanes = LaneFollower(carla, world.get_map())
    recent_yaws = {}
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
        for actor_id in set(recent_yaws) - set(actors):
            del recent_yaws[actor_id]
        yaws = {}
        for actor_id, actor in actors.items():
            tf, v = actor.get_transform(), actor.get_velocity()
            x, y = tf.location.x, tf.location.y
            vx, vy = measured_velocity(v.x, v.y, x, y, history.latest(actor_id), dt)
            history.add(actor_id, x, y, vx, vy)
            yaws[actor_id] = tf.rotation.yaw
            recent_yaws.setdefault(actor_id, deque(maxlen=YAW_HISTORY)).append(tf.rotation.yaw)

        started = time.perf_counter()
        if predictor is not None:
            paths = predict_world_paths(history.ready(), yaws, predictor)
        else:
            paths = baseline_paths(carla, lanes, actors, history.ready(), yaws, recent_yaws)
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
