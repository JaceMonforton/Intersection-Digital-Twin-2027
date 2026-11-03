"""
Central perception and safety engine. Runs on the GPU machine alongside CARLA.

Pipeline: overhead LiDAR -> 3D detections + Ditto telemetry -> fused tracks -> TTC / PET / DRAC.
This is the master process for CARLA: it enables synchronous mode and drives world.tick(), so
start it BEFORE ditto_carla_agent.py and waymo_predictor_node.py, which attach to its world.
When TTC to the ML vehicle drops below 1.5 s it sets emergency_brake = true on the vehicle's
SafetyGate in Ditto, which reaches the car over MQTT. The brake latches: release it with
    python scripts/setup_entities.py --clear-brake

Usage:
    python scripts/central_gpu_processor.py              # with CARLA
    python scripts/central_gpu_processor.py --no-carla   # Ditto telemetry only
"""
import argparse
import itertools
import json
import math
import queue
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import websocket

# --- Configuration ---
CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_MAP = "Town01"
FIXED_DELTA = 0.1       # seconds of simulation per world.tick()
TRAFFIC_MANAGER_PORT = 8000
MAP_LOAD_TIMEOUT = 300.0 # seconds; a cold load of a map can take minutes
GANTRY_HEIGHT = 40.0    # metres above the intersection; camera, LiDAR and spectator sit here
CAMERA_FOV = 90.0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT_PX = 720

LIDAR_CHANNELS = 64
LIDAR_POINTS_PER_SECOND = 100000
LIDAR_RANGE = 50.0
# CARLA's default FOV (+10 to -30 degrees) suits a roof mount. From a 40 m gantry those rays would
# all run out of range before reaching the road, so aim the channels down at the intersection.
LIDAR_UPPER_FOV = -45.0
LIDAR_LOWER_FOV = -90.0
LIDAR_TIMEOUT = 1.0     # seconds to wait for the sweep belonging to a tick
MODEL_GROUND_Z = -1.8   # z of the road in the model's frame, as if the LiDAR were on a car roof
BOX_LIFE_TIME = 0.11    # must outlive one tick: at exactly FIXED_DELTA a shape expires before it renders
BOX_THICKNESS = 0.1

DITTO_API = "http://localhost:8080/api/2"
DITTO_WS = "ws://localhost:8080/ws/2"
NAMESPACE = "org.example"
INTERSECTION_THING = "org.example:intersection_01"
CONTROLLED_VEHICLE = "vehicle_ml"  # the only vehicle with a SafetyGate
VEHICLES = ("vehicle_ml", "human_vehicle")

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "centerpoint_3d.onnx"
DETECTION_THRESHOLD = 0.5

TTC_BRAKE = 1.5             # seconds; REQ-SYS-011
COLLISION_RADIUS = 2.5      # metres between centres that counts as contact (two car half-lengths)
TELEMETRY_STALE_AFTER = 1.0 # seconds without an update before a vehicle leaves the evaluation
FUSION_GATE = 3.0           # metres; max distance to associate a detection with telemetry
TELEMETRY_WEIGHT = 0.7      # share of the fused position taken from telemetry
TELEMETRY_SCALE = 1.0       # lab metres -> CARLA metres (10.0 for a 1:10 testbed)
KINEMATICS_SPEED_TO_MS = 1 / 3.6  # kuksa_to_ditto forwards VSS Vehicle.Speed, which is km/h
BRAKE_RETRY_AFTER = 0.25    # seconds before re-sending a brake Ditto has not yet confirmed
METRICS_INTERVAL = 1.0      # seconds between SafetyMetrics writes and log lines
HEADLESS_RATE = 20          # Hz, used with --no-carla


# ---------------------------------------------------------------------------
# Surrogate safety metrics (no CARLA needed)
# ---------------------------------------------------------------------------
@dataclass
class Track:
    id: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0


def time_to_collision(a, b, radius=COLLISION_RADIUS):
    """Seconds until the centres come within radius at constant velocity; inf if they never do.
    Solves |r + v t| = radius, so unlike a closing-speed ratio it handles crossing paths."""
    rx, ry = b.x - a.x, b.y - a.y
    vx, vy = b.vx - a.vx, b.vy - a.vy
    c = rx * rx + ry * ry - radius * radius
    if c <= 0:
        return 0.0  # already in contact
    qa = vx * vx + vy * vy
    qb = 2 * (rx * vx + ry * vy)
    disc = qb * qb - 4 * qa * c
    if qa < 1e-9 or disc < 0:
        return math.inf
    t = (-qb - math.sqrt(disc)) / (2 * qa)
    return t if t >= 0 else math.inf


def post_encroachment_time(a, b):
    """Predicted PET: gap between the two arrival times at the point where the paths cross.
    True PET is only known after both have passed, so this extrapolates current velocities.
    inf if the paths don't cross ahead of both actors."""
    det = a.vx * b.vy - a.vy * b.vx
    if abs(det) < 1e-9:
        return math.inf
    dx, dy = b.x - a.x, b.y - a.y
    ta = (dx * b.vy - dy * b.vx) / det
    tb = (dx * a.vy - dy * a.vx) / det
    if ta < 0 or tb < 0:
        return math.inf
    return abs(ta - tb)


def deceleration_to_avoid_crash(a, b, radius=COLLISION_RADIUS):
    """DRAC in m/s^2: closing speed^2 / (2 * gap). 0 when the actors are not closing."""
    rx, ry = b.x - a.x, b.y - a.y
    dist = math.hypot(rx, ry)
    if dist < 1e-9:
        return math.inf
    closing = -(rx * (b.vx - a.vx) + ry * (b.vy - a.vy)) / dist
    if closing <= 0:
        return 0.0
    return closing * closing / (2 * max(dist - radius, 0.01))


# ---------------------------------------------------------------------------
# State fusion (no CARLA needed)
# ---------------------------------------------------------------------------
def _dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _prop(thing, feature, field):
    v = _dig(thing, "features", feature, "properties", field)
    return float(v) if isinstance(v, (int, float)) else None


def thing_to_track(name, thing):
    """Track from a Ditto thing, or None without a position. Accepts location + velocity
    or the kinematics feature written by hardware/kuksa_to_ditto.py."""
    x = _prop(thing, "location", "x")
    y = _prop(thing, "location", "y")
    vx = _prop(thing, "velocity", "vx")
    vy = _prop(thing, "velocity", "vy")
    if x is None or y is None:
        x, y = _prop(thing, "kinematics", "x"), _prop(thing, "kinematics", "y")
    if x is None or y is None:
        return None
    if vx is None or vy is None:
        speed = (_prop(thing, "kinematics", "speed") or 0.0) * KINEMATICS_SPEED_TO_MS
        heading = _prop(thing, "kinematics", "heading")
        if heading is None:
            heading = _prop(thing, "orientation", "yaw") or 0.0
        vx = speed * math.cos(math.radians(heading))
        vy = speed * math.sin(math.radians(heading))
    return Track(name, x * TELEMETRY_SCALE, y * TELEMETRY_SCALE,
                 vx * TELEMETRY_SCALE, vy * TELEMETRY_SCALE)


def fuse(telemetry, detections, gate=FUSION_GATE):
    """Merge detections into telemetry tracks: by id when a CARLA actor is named after the vehicle,
    otherwise by nearest neighbour. Telemetry keeps its identity and its velocity unless it reports
    none. A detection with no telemetry stays as its own track, since it may be an actor that isn't
    reporting."""
    fused = []
    unmatched = list(detections)
    for t in telemetry:
        best = next((d for d in unmatched if d.id == t.id), None)
        if best is None:
            near = min(unmatched, key=lambda d: math.hypot(d.x - t.x, d.y - t.y), default=None)
            if near is not None and math.hypot(near.x - t.x, near.y - t.y) <= gate:
                best = near
        if best is None:
            fused.append(t)
            continue
        unmatched.remove(best)
        w = TELEMETRY_WEIGHT
        vx, vy = (t.vx, t.vy) if (t.vx or t.vy) else (best.vx, best.vy)
        fused.append(Track(t.id, w * t.x + (1 - w) * best.x, w * t.y + (1 - w) * best.y, vx, vy))
    return fused + unmatched


class CentroidTracker:
    """Gives per-frame detections stable ids and a velocity from successive positions."""

    def __init__(self, gate=FUSION_GATE, max_age=0.5):
        self.gate = gate
        self.max_age = max_age
        self.tracks = {}  # id -> (Track, last seen)
        self._ids = itertools.count()

    def update(self, points, now):
        self.tracks = {k: v for k, v in self.tracks.items() if now - v[1] <= self.max_age}
        free = dict(self.tracks)
        result = {}
        for x, y in points:
            best = min(free, key=lambda k: math.hypot(free[k][0].x - x, free[k][0].y - y), default=None)
            if best is not None and math.hypot(free[best][0].x - x, free[best][0].y - y) <= self.gate:
                prev, seen = free.pop(best)
                dt = now - seen
                track = Track(best, x, y, *(((x - prev.x) / dt, (y - prev.y) / dt) if dt > 0
                                            else (prev.vx, prev.vy)))
            else:
                track = Track(f"lidar:{next(self._ids)}", x, y)
            result[track.id] = (track, now)
        self.tracks.update(result)
        return [v[0] for v in result.values()]


@dataclass
class Box:
    """A 3D box in the CARLA world frame. x, y, z is its centre."""
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw: float  # degrees


def parse_lidar(raw_data):
    """CARLA's sensor.lidar.ray_cast buffer as an N x (x, y, z, intensity) float32 matrix."""
    return np.frombuffer(raw_data, dtype=np.float32).reshape(-1, 4).copy()


def lidar_to_model_frame(points, sensor_height, ground_z=MODEL_GROUND_Z):
    """CARLA sensor frame (y right) to the model's frame (x forward, y LEFT, road at ground_z)."""
    out = points.copy()
    out[:, 1] *= -1.0
    out[:, 2] += sensor_height + ground_z
    return out


def fit_model_input(points, shape):
    """Shape an N x 4 matrix for an input declared as [N, 4] or [1, N, 4]. A fixed N is met by
    evenly thinning, which keeps the scan pattern, or by padding with zero points."""
    declared_n = shape[-2]
    if isinstance(declared_n, int):
        if len(points) > declared_n:
            points = points[np.linspace(0, len(points) - 1, declared_n).astype(int)]
        elif len(points) < declared_n:
            points = np.vstack((points, np.zeros((declared_n - len(points), 4), dtype=np.float32)))
    return points[None] if len(shape) == 3 else points


def model_box_to_world(row, center_x, center_y, center_z, ground_z=MODEL_GROUND_Z):
    """(x, y, z, length, width, height, yaw radians counter-clockwise) in the model's frame -> Box."""
    x, y, z, length, width, height, yaw = (float(v) for v in row[:7])
    return Box(center_x + x, center_y - y, center_z + z - ground_z, length, width, height, -math.degrees(yaw))


# ---------------------------------------------------------------------------
# Ditto link
# ---------------------------------------------------------------------------
def apply_event(thing, path, value):
    """Apply a Ditto event (path like '/features/location/properties/x') as a merge, so a
    partial event never discards fields from an earlier one."""
    parts = [p for p in path.split("/") if p]
    node = thing
    for p in parts[:-1]:
        if not isinstance(node.get(p), dict):
            node[p] = {}
        node = node[p]
    if not parts:
        _merge(thing, value)
    elif isinstance(value, dict) and isinstance(node.get(parts[-1]), dict):
        _merge(node[parts[-1]], value)
    else:
        node[parts[-1]] = value


def _merge(dst, patch):
    if not isinstance(patch, dict):
        return
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v


class DittoLink:
    """Keeps a live copy of each vehicle thing over the WebSocket and writes hazards back."""

    def __init__(self, api=DITTO_API, ws_url=DITTO_WS):
        self.api = api
        self.ws_url = ws_url
        self.session = requests.Session()
        self.lock = threading.Lock()
        self.things = {name: {} for name in VEHICLES}
        self.updated = {}  # vehicle name -> time of the last telemetry event
        self.running = True
        self.subscribed = threading.Event()
        self.ws = None
        self._writer = ThreadPoolExecutor(max_workers=1)  # keeps metric writes off the safety loop
        self._metrics_pending = None

    def start(self):
        threading.Thread(target=self._run_ws, daemon=True).start()

    def stop(self):
        self.running = False
        if self.ws is not None:
            self.ws.close()
        self._writer.shutdown(wait=False)

    def _load_things(self):
        """Seed from REST, because events only carry changes. Seeding does not mark a vehicle
        as fresh: a position stored before this run must never count as live telemetry."""
        for name in VEHICLES:
            res = self.session.get(f"{self.api}/things/{NAMESPACE}:{name}", timeout=5)
            if res.status_code == 200:
                with self.lock:
                    self.things[name] = res.json()
            else:
                print(f"[DITTO] {name}: HTTP {res.status_code}. Run scripts/setup_entities.py first.")

    def _run_ws(self):
        while self.running:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=lambda ws: ws.send(f"START-SEND-EVENTS?namespaces={NAMESPACE}"),
                on_message=self._on_message,
                on_error=lambda ws, e: print(f"[DITTO] WebSocket error: {e}"),
            )
            self.ws.run_forever(ping_interval=20, ping_timeout=10)
            self.subscribed.clear()
            if self.running:
                print("[DITTO] WebSocket closed, reconnecting...")
                time.sleep(1.0)

    def _on_message(self, ws, message):
        if message.startswith("START-SEND-EVENTS:ACK"):
            try:
                self._load_things()
            except requests.RequestException as e:
                print(f"[DITTO] Could not load things: {e}")
            self.subscribed.set()
            return
        try:
            env = json.loads(message)
            ns, name = env["topic"].split("/")[:2]
        except (ValueError, KeyError, TypeError, AttributeError):
            return
        if ns != NAMESPACE or name not in self.things:
            return
        path = env.get("path", "/")
        with self.lock:
            apply_event(self.things[name], path, env.get("value"))
            if path.startswith(("/features/location", "/features/velocity", "/features/kinematics",
                                "/features/orientation")) or path in ("/", "/features"):
                self.updated[name] = time.time()

    def telemetry(self, now):
        """Tracks for every vehicle with fresh telemetry, plus the names of the stale ones."""
        tracks, stale = [], []
        with self.lock:
            for name, thing in self.things.items():
                track = thing_to_track(name, thing)
                if track is not None and now - self.updated.get(name, 0.0) <= TELEMETRY_STALE_AFTER:
                    tracks.append(track)
                else:
                    stale.append(name)
        return tracks, stale

    def brake_engaged(self):
        with self.lock:
            return _dig(self.things[CONTROLLED_VEHICLE], "features", "SafetyGate", "properties",
                        "emergency_brake") is True

    def send_brake(self, ttc, reason):
        """Synchronous on purpose: this write is the point of the whole loop."""
        body = {
            "emergency_brake": True,
            "reason": reason,
            "ttc": round(ttc, 3),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        url = f"{self.api}/things/{NAMESPACE}:{CONTROLLED_VEHICLE}/features/SafetyGate/properties"
        try:
            res = self.session.put(url, json=body, timeout=1.0)
        except requests.RequestException as e:
            print(f"[ALERT] Brake command failed: {e}")
            return False
        if res.status_code not in (200, 201, 204):
            print(f"[ALERT] Brake command rejected: HTTP {res.status_code} {res.text[:200]}")
            return False
        return True

    def publish_metrics(self, report):
        if self._metrics_pending is not None and not self._metrics_pending.done():
            return  # Ditto is slow; drop this sample rather than queue behind it
        url = f"{self.api}/things/{INTERSECTION_THING}/features/SafetyMetrics/properties"
        self._metrics_pending = self._writer.submit(self._put_quietly, url, report)

    def _put_quietly(self, url, body):
        try:
            self.session.put(url, json=body, timeout=2.0)
        except requests.RequestException as e:
            print(f"[DITTO] Could not write metrics: {e}")


# ---------------------------------------------------------------------------
# Safety engine
# ---------------------------------------------------------------------------
def _finite(v, digits=3):
    """JSON has no Infinity, so 'no conflict' is written as null."""
    return round(v, digits) if math.isfinite(v) else None


class SafetyEngine:
    def __init__(self, ditto):
        self.ditto = ditto
        self.last_brake_sent = 0.0
        self.last_report = 0.0
        self.warned_blind = False

    def evaluate(self, tracks):
        """Metrics for every pair, most urgent first."""
        rows = []
        for a, b in itertools.combinations(tracks, 2):
            rows.append({
                "pair": [a.id, b.id],
                "ttc": time_to_collision(a, b),
                "pet": post_encroachment_time(a, b),
                "drac": deceleration_to_avoid_crash(a, b),
            })
        rows.sort(key=lambda r: r["ttc"])
        return rows

    def step(self, now, detections=()):
        telemetry, _stale = self.ditto.telemetry(now)
        tracks = fuse(telemetry, list(detections))
        self._warn_if_ml_vehicle_blind(tracks)
        rows = self.evaluate(tracks)

        hazard = next((r for r in rows
                       if CONTROLLED_VEHICLE in r["pair"] and r["ttc"] < TTC_BRAKE), None)
        if (hazard is not None and not self.ditto.brake_engaged()
                and now - self.last_brake_sent >= BRAKE_RETRY_AFTER):
            self.last_brake_sent = now
            other = next(i for i in hazard["pair"] if i != CONTROLLED_VEHICLE)
            reason = f"TTC {hazard['ttc']:.2f} s with {other}"
            if self.ditto.send_brake(hazard["ttc"], reason):
                print(f"[ALERT] EMERGENCY BRAKE sent: {reason}")

        if rows and now - self.last_report >= METRICS_INTERVAL:
            self.last_report = now
            worst = rows[0]
            self.ditto.publish_metrics({
                "ttc": _finite(worst["ttc"]),
                "pet": _finite(worst["pet"]),
                "drac": _finite(worst["drac"]),
                "pair": worst["pair"],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            print(f"[METRICS] {'/'.join(worst['pair'])} TTC={worst['ttc']:.2f}s "
                  f"PET={worst['pet']:.2f}s DRAC={worst['drac']:.2f}m/s2")
        return rows

    def _warn_if_ml_vehicle_blind(self, tracks):
        """A vehicle with no track can't have a hazard computed, which must not pass silently."""
        visible = any(t.id == CONTROLLED_VEHICLE for t in tracks)
        if not visible and not self.warned_blind:
            print(f"[WARN] No fresh track for {CONTROLLED_VEHICLE}: it is NOT being protected.")
        self.warned_blind = not visible


# ---------------------------------------------------------------------------
# Perception (needs CARLA or the model)
# ---------------------------------------------------------------------------
class CenterPointDetector:
    """3D boxes from a LiDAR sweep. See models/README.md for the expected model contract."""

    def __init__(self, path):
        import onnxruntime

        available = onnxruntime.get_available_providers()
        providers = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider",
                                 "CPUExecutionProvider") if p in available]
        self.session = onnxruntime.InferenceSession(str(path), providers=providers)
        model_input = self.session.get_inputs()[0]
        self.shape = model_input.shape
        if len(self.shape) not in (2, 3) or self.shape[-1] != 4:
            raise ValueError(f"{path} takes input of shape {self.shape}, but this detector feeds "
                             "[N, 4] or [1, N, 4] points. See models/README.md.")
        self.input_name = model_input.name

    def detect(self, points):
        """Rows of (x, y, z, length, width, height, yaw, score, class) above the score threshold."""
        tensor = fit_model_input(points, self.shape).astype(np.float32)
        rows = self.session.run(None, {self.input_name: tensor})[0].reshape(-1, 9)
        return rows[rows[:, 7] >= DETECTION_THRESHOLD]


class CarlaPerception:
    """Owns the simulation clock and the overhead gantry at the intersection."""

    def __init__(self, junction_index, detector_mode, model_path):
        import carla

        self.carla = carla
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(60.0)
        self.world = self.client.get_world()
        if self.world.get_settings().synchronous_mode:
            print("[CARLA] Server was already synchronous, taking over as master. "
                  "If another processor is running, stop it.")
        self._leave_synchronous_mode()  # a master that died uncleanly leaves these settings behind
        if CARLA_MAP not in self.world.get_map().name:
            print(f"[CARLA] Loading map {CARLA_MAP}...")
            self.client.set_timeout(MAP_LOAD_TIMEOUT)
            self.world = self.client.load_world(CARLA_MAP)
            self.client.set_timeout(60.0)

        self.center = self._find_intersection(junction_index)
        print(f"[CARLA] Town01_Intersection_01 is junction #{junction_index} "
              f"at x={self.center.x:.1f} y={self.center.y:.1f}")
        self.gantry = carla.Location(x=self.center.x, y=self.center.y, z=self.center.z + GANTRY_HEIGHT)

        self.detector = None
        self.tracker = CentroidTracker()
        self.boxes = []
        self._previous = {}  # actor id -> (x, y, time), for actors CARLA reports no velocity for
        if detector_mode == "onnx" or (detector_mode == "auto" and Path(model_path).exists()):
            self.detector = CenterPointDetector(model_path)
            self.box_color = carla.Color(0, 255, 0)
            print(f"[LIDAR] Using model {model_path}")
        else:
            self.box_color = carla.Color(255, 255, 0)
            print("[LIDAR] No detection model: using CARLA GROUND-TRUTH actor positions, not LiDAR.")

        self.sweeps = queue.Queue(maxsize=1)
        self.frame_count = 0
        self.sensors = [self._spawn_camera(), self._spawn_lidar()]
        self.world.get_spectator().set_transform(
            carla.Transform(self.gantry, carla.Rotation(pitch=-90.0, yaw=0.0)))

        # Only after any map load: a Traffic Manager alive during load_world can abort the client
        # with "_episode != nullptr". And before ticks stop, so it can start up against a live server.
        self.traffic_manager = self.client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
        self.traffic_manager.set_synchronous_mode(True)

        # Last, so a failure above can't leave the server synchronous with no master
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA
        self.world.apply_settings(settings)

    def _leave_synchronous_mode(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

    def _find_intersection(self, index):
        junctions = {}
        for wp, _ in self.world.get_map().get_topology():
            if wp.is_junction:
                junction = wp.get_junction()
                junctions[junction.id] = junction
        if not junctions:
            raise RuntimeError(f"{CARLA_MAP} has no junctions.")
        ordered = [junctions[k] for k in sorted(junctions)]
        return ordered[index % len(ordered)].bounding_box.location

    def _spawn_camera(self):
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
        bp.set_attribute("image_size_y", str(CAMERA_HEIGHT_PX))
        bp.set_attribute("fov", str(CAMERA_FOV))
        transform = self.carla.Transform(self.gantry, self.carla.Rotation(pitch=-90.0))
        camera = self.world.spawn_actor(bp, transform)
        camera.listen(self._on_frame)
        return camera

    def _spawn_lidar(self):
        bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
        bp.set_attribute("channels", str(LIDAR_CHANNELS))
        bp.set_attribute("points_per_second", str(LIDAR_POINTS_PER_SECOND))
        bp.set_attribute("range", str(LIDAR_RANGE))
        bp.set_attribute("upper_fov", str(LIDAR_UPPER_FOV))
        bp.set_attribute("lower_fov", str(LIDAR_LOWER_FOV))
        bp.set_attribute("rotation_frequency", str(1.0 / FIXED_DELTA))  # one full sweep per tick
        bp.set_attribute("dropoff_general_rate", "0.0")  # the default silently drops 45% of points
        lidar = self.world.spawn_actor(bp, self.carla.Transform(self.gantry))
        lidar.listen(self._on_sweep)
        return lidar

    def _on_frame(self, image):
        self.frame_count += 1

    def _on_sweep(self, measurement):
        try:
            self.sweeps.get_nowait()  # only the newest sweep matters
        except queue.Empty:
            pass
        self.sweeps.put_nowait((measurement.frame, parse_lidar(measurement.raw_data)))

    def _sweep_for(self, frame):
        """The sweep from this tick, or None if it doesn't arrive in time."""
        deadline = time.time() + LIDAR_TIMEOUT
        while time.time() < deadline:
            try:
                sweep_frame, points = self.sweeps.get(timeout=LIDAR_TIMEOUT)
            except queue.Empty:
                return None
            if sweep_frame >= frame:
                return points
        return None

    def tick(self):
        """Advance the simulation one step and return the tracks detected in it."""
        frame = self.world.tick()
        now = self.world.get_snapshot().timestamp.elapsed_seconds
        return self._detect(frame, now) if self.detector else self._ground_truth(now)

    def _detect(self, frame, now):
        points = self._sweep_for(frame)
        if points is None:
            print("[LIDAR] No sweep arrived for this tick.")
            self.boxes = []
            return []
        rows = self.detector.detect(lidar_to_model_frame(points, GANTRY_HEIGHT))
        self.boxes = [model_box_to_world(r, self.center.x, self.center.y, self.center.z) for r in rows]
        return self.tracker.update([(b.x, b.y) for b in self.boxes], now)

    def _ground_truth(self, now):
        tracks, boxes, previous = [], [], {}
        for actor in self.world.get_actors().filter("vehicle.*"):
            tf = actor.get_transform()
            x, y = tf.location.x, tf.location.y
            v = actor.get_velocity()
            vx, vy = v.x, v.y
            seen = self._previous.get(actor.id)
            # A twin moved with set_transform reports zero velocity, so measure it instead
            if math.hypot(vx, vy) < 0.01 and seen is not None and now > seen[2]:
                vx, vy = (x - seen[0]) / (now - seen[2]), (y - seen[1]) / (now - seen[2])
            previous[actor.id] = (x, y, now)

            role = actor.attributes.get("role_name")
            tracks.append(Track(role if role in VEHICLES else f"carla:{actor.id}", x, y, vx, vy))
            bb = actor.bounding_box
            c = tf.transform(bb.location)
            boxes.append(Box(c.x, c.y, c.z, 2 * bb.extent.x, 2 * bb.extent.y, 2 * bb.extent.z, tf.rotation.yaw))
        self._previous = previous
        self.boxes = boxes
        return tracks

    def render(self):
        for b in self.boxes:
            self.world.debug.draw_box(
                self.carla.BoundingBox(
                    self.carla.Location(x=b.x, y=b.y, z=b.z),
                    self.carla.Vector3D(x=b.length / 2, y=b.width / 2, z=b.height / 2)),
                self.carla.Rotation(yaw=b.yaw),
                thickness=BOX_THICKNESS, color=self.box_color, life_time=BOX_LIFE_TIME)

    def cleanup(self):
        for sensor in self.sensors:
            sensor.stop()
            sensor.destroy()
        self.traffic_manager.set_synchronous_mode(False)
        self._leave_synchronous_mode()


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-carla", action="store_true", help="evaluate Ditto telemetry only")
    parser.add_argument("--junction", type=int, default=0, help="which map junction to watch, by id order")
    parser.add_argument("--detector", choices=("auto", "onnx", "groundtruth"), default="auto")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    args = parser.parse_args()

    # Without this, Ctrl+Break or a closed console skips the cleanup below, leaving the sensors
    # behind and CARLA's settings saying synchronous with no master.
    for name in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _raise_keyboard_interrupt)

    ditto = DittoLink()
    ditto.start()
    if not ditto.subscribed.wait(timeout=10.0):
        print("[DITTO] No event stream yet; continuing and retrying in the background.")
    engine = SafetyEngine(ditto)
    perception = None
    print(f"[READY] Braking {CONTROLLED_VEHICLE} when TTC < {TTC_BRAKE} s. Ctrl+C to exit.")

    try:
        if args.no_carla:
            while True:
                engine.step(time.time())
                time.sleep(1.0 / HEADLESS_RATE)
        else:
            perception = CarlaPerception(args.junction, args.detector, args.model)
            print(f"[CARLA] Master tick running at {1 / FIXED_DELTA:.0f} Hz.")
            next_tick = time.perf_counter()
            while True:
                tracks = perception.tick()
                engine.step(time.time(), tracks)
                perception.render()
                # An unthrottled tick loop would run the simulation faster than real time,
                # while telemetry and the agents' dead reckoning run on the wall clock.
                next_tick += FIXED_DELTA
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        if perception is not None:
            perception.cleanup()
        ditto.stop()


if __name__ == "__main__":
    main()
