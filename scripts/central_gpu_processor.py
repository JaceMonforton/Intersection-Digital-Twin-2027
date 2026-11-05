"""
Central perception and safety engine. Runs on the GPU machine alongside CARLA.

Pipeline: two corner LiDARs -> 3D detections + Ditto telemetry -> fused tracks -> TTC / PET / DRAC.
This is the master process for CARLA: it enables synchronous mode and drives world.tick(), so
start it BEFORE ditto_carla_agent.py and waymo_predictor_node.py, which attach to its world.
When TTC to the ML vehicle drops below 1.5 s, or the vehicles' predicted paths meet within 3 s, it
sets emergency_brake = true on the vehicle's SafetyGate in Ditto, which reaches the car over MQTT. The brake latches: release it with
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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import websocket
from model_panel import SafetySender
from waymo_predictor_node import (
    DEFAULT_MODEL as TRAJECTORY_MODEL,
)
from waymo_predictor_node import (
    STEP_SECONDS,
    YAW_HISTORY,
    HistoryBuffer,
    LaneFollower,
    OnnxPredictor,
    curve_path,
    predict_world_paths,
    signalled_turn,
    turn_intent,
    turn_rate,
)

# --- Configuration ---
CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_MAP = "Town10HD_Opt"
INTERSECTION_NAME = "Town10_Intersection_01"
# A signalized four-way with left turns across oncoming traffic and a clear view from above
INTERSECTION_JUNCTION = 189
FIXED_DELTA = 0.1       # seconds of simulation per world.tick()
TRAFFIC_MANAGER_PORT = 8000
MAP_LOAD_TIMEOUT = 300.0 # seconds; a cold load of a map can take minutes
GANTRY_HEIGHT = 40.0    # metres above the intersection; the overhead view (spectator) sits here
CAMERA_FOV = 90.0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT_PX = 720

# Each LiDAR's (x, y, z) in metres from the junction centre, in CARLA's frame. Opposite corners, so one
# sees what's hidden from the other. training/check_lidar_coverage.py compares the options.
LIDAR_MOUNTS = ((14.1, 17.2, 7.0), (-14.1, -17.2, 7.0))
LIDAR_CHANNELS = 64
LIDAR_POINTS_PER_SECOND = 300000  # each
LIDAR_RANGE = 80.0
# Aimed below the horizon, like the OS1-64s on the TUMTraf gantry. The steepest ray leaves a blind
# circle around the pole as wide as the mount is high, which a corner keeps mostly off the road.
LIDAR_UPPER_FOV = 0.0
LIDAR_LOWER_FOV = -45.0
LIDAR_TIMEOUT = 1.0     # seconds to wait for the sweep belonging to a tick
MODEL_GROUND_Z = -1.8   # z of the road in the model's frame, as if the LiDAR were on a car roof
LABEL_LIFE_TIME = 0.1  # text, unlike lines, renders at one tick's life, and longer leaves a ghost behind it
BOX_LIFE_TIME = 0.11    # must outlive one tick: at exactly FIXED_DELTA a shape expires before it renders
BOX_THICKNESS = 0.1
# Debug shapes are unlit, so full brightness blooms at high render quality
BOX_COLOR_MODEL = (0, 50, 0)
BOX_COLOR_GROUND_TRUTH = (50, 50, 0)
BOX_COLOR_ML = (55, 0, 55)  # magenta, so the ML car stands out from everything else
ML_PAINT = "170,50,170"      # its body colour, which no other car uses
# CARLA draws debug shapes into every camera, and the ML car's own cameras shouldn't show its box or
# label. So they're drawn this far above the car, out of those cameras' view, and scaled to compensate
# so that they still line up with the car from directly overhead.
ML_MARKER_HEIGHT = 7.0      # metres above the car
MAX_PLAUSIBLE_SPEED = 40.0  # m/s; a faster measured speed is a car being moved, not driving

DITTO_API = "http://localhost:8080/api/2"
DITTO_WS = "ws://localhost:8080/ws/2"
NAMESPACE = "org.example"
INTERSECTION_THING = "org.example:intersection_01"
CONTROLLED_VEHICLE = "vehicle_ml"  # the only vehicle with a SafetyGate
VEHICLES = ("vehicle_ml", "human_vehicle")

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "centerpoint_3d.onnx"
DETECTION_THRESHOLD = 0.5

TTC_BRAKE = 1.5             # seconds; REQ-SYS-011
# Straight-line TTC can't see a turn coming, and by the time it fires the car is across the lane.
# Paths are compared further out, which leaves room to stop before reaching the conflict.
PATH_BRAKE_HORIZON = 3.0    # seconds
REAR_HEADING_TOLERANCE = 45.0  # degrees either way that still counts as travelling the same way
STOPPED_SPEED = 0.5         # m/s; a car going slower than this can't be helped by braking
PATH_MOVING_SPEED = 1.0     # m/s; the path check is for crossing traffic, so slower cars are left to TTC
IN_LINE_HEADING = 30.0      # degrees between two cars that counts as following the same way
TURNING_RATE = 15.0         # degrees per second; turning faster than this, straight-line TTC means little
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
@dataclass(eq=False)  # compared by identity: path is an array, which has no single truth value
class Track:
    id: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    path: np.ndarray = None  # predicted (x, y) at each STEP_SECONDS, where one is known
    yaw: float = None        # degrees, where known
    yaw_rate: float = 0.0    # degrees per second


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


def approaching_from_behind(me, other):
    """True if other is behind me and travelling my way. Braking can't avoid that, and can make it
    worse, so it must not trigger the brake. Without headings for both, assume it might matter."""
    if me.yaw is None or other.yaw is None:
        return False
    yaw = math.radians(me.yaw)
    ahead = math.cos(yaw) * (other.x - me.x) + math.sin(yaw) * (other.y - me.y)
    heading = abs((other.yaw - me.yaw + 180.0) % 360.0 - 180.0)
    return ahead < 0.0 and heading < REAR_HEADING_TOLERANCE


def is_following(me, other):
    """True if other is simply ahead of me going the same way: ordinary traffic that I slow for."""
    if me.yaw is None or other.yaw is None:
        return False
    return abs((other.yaw - me.yaw + 180.0) % 360.0 - 180.0) < IN_LINE_HEADING


def path_check_applies(me, other):
    """The path check looks 3 s ahead, which suits a car crossing my path. Applied to a stopped car or
    one I'm following, it brakes on ordinary traffic, so those are left to the 1.5 s TTC rule."""
    return math.hypot(other.vx, other.vy) >= PATH_MOVING_SPEED and not is_following(me, other)


def ttc_check_applies(me, other):
    """Mid-turn, straight-line TTC points at whatever is in front at that instant, such as cars waiting
    at a red light, and braking for those would strand the car in the junction."""
    return abs(me.yaw_rate) <= TURNING_RATE or math.hypot(other.vx, other.vy) >= PATH_MOVING_SPEED


def path_time_to_collision(a, b, radius=COLLISION_RADIUS, step_seconds=STEP_SECONDS):
    """Seconds until two predicted paths bring the vehicles within radius at the same moment;
    inf if they don't, or if either has no path."""
    if a.path is None or b.path is None:
        return math.inf
    steps = min(len(a.path), len(b.path))
    apart = a.path[:steps] - b.path[:steps]
    close = np.flatnonzero(np.hypot(apart[:, 0], apart[:, 1]) < radius)
    return float((close[0] + 1) * step_seconds) if close.size else math.inf


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
        fused.append(Track(t.id, w * t.x + (1 - w) * best.x, w * t.y + (1 - w) * best.y, vx, vy, best.path, best.yaw, best.yaw_rate))
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
    is_ml: bool = False


def overhead_marker(box, gantry_x, gantry_y, gantry_z, marker_height=ML_MARKER_HEIGHT):
    """A flat copy of a box, raised by marker_height, that looks the same from the gantry. Anything
    nearer a camera looks bigger and further from the centre of its view, so both are scaled down by
    how much closer it is."""
    base_z = box.z - box.height / 2
    scale = (gantry_z - base_z - marker_height) / (gantry_z - base_z)
    return Box(gantry_x + (box.x - gantry_x) * scale, gantry_y + (box.y - gantry_y) * scale, base_z + marker_height,
               box.length * scale, box.width * scale, 0.02, box.yaw, box.is_ml)


def parse_lidar(raw_data):
    """CARLA's sensor.lidar.ray_cast buffer as an N x (x, y, z, intensity) float32 matrix."""
    return np.frombuffer(raw_data, dtype=np.float32).reshape(-1, 4).copy()


def lidar_to_model_frame(points, mount, ground_z=MODEL_GROUND_Z):
    """One LiDAR's sweep, in its own CARLA frame (y right), to the model's frame shared by all of them:
    centred on the junction, x forward, y LEFT, road at ground_z. The sensors aren't rotated."""
    out = points.copy()
    out[:, 0] += mount[0]
    out[:, 1] = -(out[:, 1] + mount[1])
    out[:, 2] += mount[2] + ground_z
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
        self.safety_out = SafetySender()

    def _report_ml_safety(self, rows):
        """What the checks say about the ML car this tick, for the display and the test suite."""
        mine = [r for r in rows if CONTROLLED_VEHICLE in r["pair"]]
        if not mine:
            return
        nearest = min(mine, key=lambda r: min(r["ttc"], r["path_ttc"]))
        other = next(i for i in nearest["pair"] if i != CONTROLLED_VEHICLE)
        self.safety_out.send(_finite(min(r["ttc"] for r in mine)), _finite(min(r["path_ttc"] for r in mine)),
                             _finite(min(r["pet"] for r in mine)), other, self.ditto.brake_engaged())

    def evaluate(self, tracks):
        """Metrics for every pair, most urgent first."""
        rows = []
        for a, b in itertools.combinations(tracks, 2):
            rows.append({
                "pair": [a.id, b.id],
                "tracks": (a, b),
                "ttc": time_to_collision(a, b),
                "path_ttc": path_time_to_collision(a, b),
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
        self._report_ml_safety(rows)

        hazard = next((r for r in rows if self._brake_can_help(r)
                       and (self._ttc_triggers(r) or (r["path_ttc"] < PATH_BRAKE_HORIZON and self._path_applies(r)))), None)
        if (hazard is not None and not self.ditto.brake_engaged()
                and now - self.last_brake_sent >= BRAKE_RETRY_AFTER):
            self.last_brake_sent = now
            other = next(i for i in hazard["pair"] if i != CONTROLLED_VEHICLE)
            # The sooner of the two; a path conflict alone leaves straight-line TTC infinite
            from_ttc = self._ttc_triggers(hazard)
            ttc = hazard["ttc"] if from_ttc else hazard["path_ttc"]
            source = "TTC" if from_ttc else "predicted path conflict in"
            reason = f"{source} {ttc:.2f} s with {other}"
            if self.ditto.send_brake(ttc, reason):
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

    @staticmethod
    def _ttc_triggers(row):
        a, b = row["tracks"]
        me, other = (a, b) if a.id == CONTROLLED_VEHICLE else (b, a)
        return row["ttc"] < TTC_BRAKE and ttc_check_applies(me, other)

    @staticmethod
    def _path_applies(row):
        a, b = row["tracks"]
        me, other = (a, b) if a.id == CONTROLLED_VEHICLE else (b, a)
        return path_check_applies(me, other)

    @staticmethod
    def _brake_can_help(row):
        if CONTROLLED_VEHICLE not in row["pair"]:
            return False
        a, b = row["tracks"]
        me, other = (a, b) if a.id == CONTROLLED_VEHICLE else (b, a)
        if math.hypot(me.vx, me.vy) < STOPPED_SPEED:
            return False
        return not approaching_from_behind(me, other)

    def _warn_if_ml_vehicle_blind(self, tracks):
        """A vehicle with no track can't have a hazard computed, which must not pass silently."""
        visible = any(t.id == CONTROLLED_VEHICLE for t in tracks)
        if not visible and not self.warned_blind:
            print(f"[WARN] No fresh track for {CONTROLLED_VEHICLE}: it is NOT being protected.")
        self.warned_blind = not visible


# ---------------------------------------------------------------------------
# Perception (needs CARLA or the model)
# ---------------------------------------------------------------------------
def find_junction(world, junction_id):
    junctions = {}
    for wp, _ in world.get_map().get_topology():
        if wp.is_junction:
            junction = wp.get_junction()
            junctions[junction.id] = junction
    if junction_id not in junctions:
        raise RuntimeError(f"{CARLA_MAP} has no junction {junction_id}. It has: {sorted(junctions)}")
    return junctions[junction_id]


@dataclass
class Lidar:
    sensor: object
    mount: tuple
    sweeps: queue.Queue


def spawn_lidars(carla, world, center, mounts=LIDAR_MOUNTS):
    bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
    bp.set_attribute("channels", str(LIDAR_CHANNELS))
    bp.set_attribute("points_per_second", str(LIDAR_POINTS_PER_SECOND))
    bp.set_attribute("range", str(LIDAR_RANGE))
    bp.set_attribute("upper_fov", str(LIDAR_UPPER_FOV))
    bp.set_attribute("lower_fov", str(LIDAR_LOWER_FOV))
    bp.set_attribute("rotation_frequency", str(1.0 / FIXED_DELTA))  # one full sweep per tick
    bp.set_attribute("dropoff_general_rate", "0.0")  # the default silently drops 45% of points
    lidars = []
    for x, y, z in mounts:
        location = carla.Location(x=center.x + x, y=center.y + y, z=center.z + z)
        lidar = Lidar(world.spawn_actor(bp, carla.Transform(location)), (x, y, z), queue.Queue(maxsize=1))
        lidar.sensor.listen(lambda measurement, sweeps=lidar.sweeps: _keep_newest(sweeps, measurement))
        lidars.append(lidar)
    return lidars


def _keep_newest(sweeps, measurement):
    try:
        sweeps.get_nowait()  # only the newest sweep matters
    except queue.Empty:
        pass
    sweeps.put_nowait((measurement.frame, parse_lidar(measurement.raw_data)))


def _sweep_for(sweeps, frame):
    """The sweep from this tick, or None if it doesn't arrive in time."""
    deadline = time.time() + LIDAR_TIMEOUT
    while time.time() < deadline:
        try:
            sweep_frame, points = sweeps.get(timeout=LIDAR_TIMEOUT)
        except queue.Empty:
            return None
        if sweep_frame >= frame:
            return points
    return None


def merged_sweep(lidars, frame):
    """Every LiDAR's sweep from this tick as one cloud in the model's frame. None if any is late, since
    a partial cloud would quietly hide part of the junction."""
    parts = []
    for lidar in lidars:
        points = _sweep_for(lidar.sweeps, frame)
        if points is None:
            return None
        parts.append(lidar_to_model_frame(points, lidar.mount))
    return np.vstack(parts)


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
    """Owns the simulation clock, the overhead view and the LiDARs at the intersection."""

    def __init__(self, junction_id, detector_mode, model_path):
        import carla

        self.carla = carla
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(60.0)
        self.world = self.client.get_world()
        if self.world.get_settings().synchronous_mode:
            print("[CARLA] Server was already synchronous, taking over as master. "
                  "If another processor is running, stop it.")
        self._leave_synchronous_mode()  # a master that died uncleanly leaves these settings behind
        self._clear_leftovers()
        if CARLA_MAP not in self.world.get_map().name:
            print(f"[CARLA] Loading map {CARLA_MAP}...")
            self.client.set_timeout(MAP_LOAD_TIMEOUT)
            self.world = self.client.load_world(CARLA_MAP)
            self.client.set_timeout(60.0)

        self.center = find_junction(self.world, junction_id).bounding_box.location
        print(f"[CARLA] {INTERSECTION_NAME} is junction {junction_id} "
              f"at x={self.center.x:.1f} y={self.center.y:.1f}")
        self.gantry = carla.Location(x=self.center.x, y=self.center.y, z=self.center.z + GANTRY_HEIGHT)

        self.detector = None
        self.tracker = CentroidTracker()
        self.lanes = LaneFollower(carla, self.world.get_map())
        self._recent_yaws = {}  # actor id -> recent yaws, for the turn rate
        self._ml_history = HistoryBuffer()
        self.ml_model = None
        if Path(TRAJECTORY_MODEL).exists():
            self.ml_model = OnnxPredictor(TRAJECTORY_MODEL)
            print(f"[PREDICT] {CONTROLLED_VEHICLE} is predicted by the trained model only: {TRAJECTORY_MODEL}")
        else:
            print(f"[PREDICT] {TRAJECTORY_MODEL} not found: {CONTROLLED_VEHICLE} falls back to ROAD-FOLLOWING prediction.")
        self.boxes = []
        self._previous = {}  # actor id -> (x, y, time), for actors CARLA reports no velocity for
        if detector_mode == "onnx" or (detector_mode == "auto" and Path(model_path).exists()):
            self.detector = CenterPointDetector(model_path)
            self.box_color = carla.Color(*BOX_COLOR_MODEL)
            print(f"[LIDAR] Using model {model_path}")
        else:
            self.box_color = carla.Color(*BOX_COLOR_GROUND_TRUTH)
            print("[LIDAR] No detection model: using CARLA GROUND-TRUTH actor positions, not LiDAR.")

        # Only with a detector: ray tracing them slows every tick, and ground truth doesn't read them
        self.lidars = spawn_lidars(carla, self.world, self.center) if self.detector else []
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

    def _clear_leftovers(self):
        """A crashed run leaves its cars and sensors behind, and they'd slow and interfere with this one."""
        actors = self.world.get_actors()
        leftovers = [a for pattern in ("sensor.*", "vehicle.*", "controller.*", "walker.*")
                     for a in actors.filter(pattern)]
        if leftovers:
            print(f"[CARLA] Removing {len(leftovers)} actors left over from an earlier run.")
            self.client.apply_batch_sync([self.carla.command.DestroyActor(a) for a in leftovers])

    def _leave_synchronous_mode(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

    def tick(self):
        """Advance the simulation one step and return the tracks detected in it."""
        frame = self.world.tick()
        now = self.world.get_snapshot().timestamp.elapsed_seconds
        return self._detect(frame, now) if self.detector else self._ground_truth(now)

    def _detect(self, frame, now):
        points = merged_sweep(self.lidars, frame)
        if points is None:
            print("[LIDAR] No sweep arrived for this tick.")
            self.boxes = []
            return []
        rows = self.detector.detect(points)
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
                if math.hypot(vx, vy) > MAX_PLAUSIBLE_SPEED:  # it was moved, so this history is meaningless
                    vx, vy = 0.0, 0.0
                    self._ml_history.retain([])
            previous[actor.id] = (x, y, now)

            role = actor.attributes.get("role_name")
            track_id = role if role in VEHICLES else f"carla:{actor.id}"
            path = self._predict_path(actor, tf, x, y, vx, vy)
            tracks.append(Track(track_id, x, y, vx, vy, path, tf.rotation.yaw, turn_rate(list(self._recent_yaws[actor.id]))))
            bb = actor.bounding_box
            c = tf.transform(bb.location)
            # Drawn now but shown on the next tick, so where the car will be then, or the box trails it
            boxes.append(Box(c.x + vx * FIXED_DELTA, c.y + vy * FIXED_DELTA, c.z, 2 * bb.extent.x, 2 * bb.extent.y, 2 * bb.extent.z, tf.rotation.yaw,
                             role == CONTROLLED_VEHICLE))
        self._previous = previous
        self._recent_yaws = {k: v for k, v in self._recent_yaws.items() if k in previous}
        self._ml_history.retain(previous)
        self.boxes = boxes
        return tracks

    def _predict_path(self, actor, tf, x, y, vx, vy):
        """Where this vehicle is expected to go. The ML car uses only the trained model, so until it
        has 1 s of history it has no path; the straight-line TTC check still covers it. Other cars
        use the road baseline."""
        yaws = self._recent_yaws.setdefault(actor.id, deque(maxlen=YAW_HISTORY))
        yaws.append(tf.rotation.yaw)
        if self.ml_model is not None and actor.attributes.get("role_name") == CONTROLLED_VEHICLE:
            self._ml_history.add(actor.id, x, y, vx, vy)
            ready = self._ml_history.ready()
            if actor.id not in ready:
                return None
            return predict_world_paths({actor.id: ready[actor.id]}, {actor.id: tf.rotation.yaw}, self.ml_model)[actor.id]
        speed = math.hypot(vx, vy)
        rate = turn_rate(list(yaws))
        intent = turn_intent(signalled_turn(self.carla, actor), rate)
        path = self.lanes.path(x, y, tf.rotation.yaw, speed, intent)
        return path if path is not None else curve_path(x, y, tf.rotation.yaw, speed, rate)

    def render(self):
        ml_color = self.carla.Color(*BOX_COLOR_ML)
        for b in self.boxes:
            if b.is_ml:
                b = overhead_marker(b, self.gantry.x, self.gantry.y, self.gantry.z)
                self.world.debug.draw_string(self.carla.Location(x=b.x, y=b.y, z=b.z), "ML CAR",
                                             color=self.carla.Color(170, 50, 170), life_time=LABEL_LIFE_TIME)
            self.world.debug.draw_box(
                self.carla.BoundingBox(
                    self.carla.Location(x=b.x, y=b.y, z=b.z),
                    self.carla.Vector3D(x=b.length / 2, y=b.width / 2, z=b.height / 2)),
                self.carla.Rotation(yaw=b.yaw),
                thickness=BOX_THICKNESS, color=ml_color if b.is_ml else self.box_color, life_time=BOX_LIFE_TIME)

    def cleanup(self):
        for lidar in self.lidars:
            lidar.sensor.stop()
            lidar.sensor.destroy()
        self.traffic_manager.set_synchronous_mode(False)
        self._leave_synchronous_mode()


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-carla", action="store_true", help="evaluate Ditto telemetry only")
    parser.add_argument("--junction", type=int, default=INTERSECTION_JUNCTION, help="id of the map junction to watch")
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
