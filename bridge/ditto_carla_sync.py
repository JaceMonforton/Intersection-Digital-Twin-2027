"""
Ditto -> CARLA bridge over the Ditto WebSocket (push, no polling).

Why the original didn't move the car:
  * It only reacted when path contained "/features/kinematics" AND value was a dict
    with x/y/heading. Ditto sends fine-grained events, e.g.
        path="/features/location/properties/x", value=12.3     (scalar)
        path="/attributes/location", value={"x":..,"y":..}
        action="merged" events from PATCH requests
    and the feature/attribute names in your thing are location/orientation, not kinematics.
  * Physics was left on, so gravity/collisions fought set_transform.
  * z was hard-coded to 0.5, so the car could clip into the road.
  * The socket thread died on any disconnect, with no reconnect.

This version keeps a local copy of the thing, applies EVERY event to it by path
(set / merge / delete), then extracts the pose from that state.
"""
import base64
import json
import logging
import math
import sys
import threading
import time

import carla
import websocket

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ditto_bridge")

# --- Configuration ---------------------------------------------------------
CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_MAP = None  # e.g. "Town02" to force-load; None keeps whatever is loaded

DITTO_WS_URL = "ws://192.168.0.100:8080/ws/2"
DITTO_AUTH = ("ditto", "ditto")
THING_ID = "org.ovin:my-vehicle"  # only events for this thing are applied; None = all
FILTER = None                      # optional RQL filter, e.g. 'eq(thingId,"org.ovin:my-vehicle")'

# Mapping (applied: scale -> flip -> offset)
POSITION_SCALE = 1.0      # use 10.0 only if the publisher sends lab metres for a 1:10 map
FLIP_Y = False            # True if the publisher is ROS/ENU (right-handed)
X_OFFSET = 0.0
Y_OFFSET = 0.0
YAW_IN_RADIANS = False    # ROS publishers usually send radians
YAW_OFFSET_DEG = 0.0

SNAP_TO_GROUND = True
GROUND_CLEARANCE = 0.3
LOG_INTERVAL = 1.0
RECONNECT_DELAY = 2.0


# --- Pure helpers (no CARLA needed) ---------------------------------------
def _dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _num(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def find_signal(thing, group, field):
    """Look for group.field under features.<g>.properties, features.<g>, attributes.<g>."""
    for path in (
        ("features", group, "properties", field),
        ("features", group, field),
        ("attributes", group, field),
    ):
        v = _num(_dig(thing, *path))
        if v is not None:
            return v
    return None


def parse_pose(thing):
    """Return (pose_dict, missing_list). Accepts location/orientation or kinematics."""
    x = find_signal(thing, "location", "x")
    y = find_signal(thing, "location", "y")
    z = find_signal(thing, "location", "z")
    yaw = find_signal(thing, "orientation", "yaw")
    if x is None or y is None:  # fallback: the original script's "kinematics" layout
        x = find_signal(thing, "kinematics", "x") if x is None else x
        y = find_signal(thing, "kinematics", "y") if y is None else y
    if yaw is None:
        yaw = find_signal(thing, "kinematics", "heading")
    if yaw is None:
        yaw = find_signal(thing, "location", "yaw")
    missing = [n for n, v in (("x", x), ("y", y)) if v is None]
    return {"x": x, "y": y, "z": z, "yaw": yaw}, missing


def map_to_carla(x, y, yaw):
    cx, cy = x * POSITION_SCALE, y * POSITION_SCALE
    yaw_deg = math.degrees(yaw) if YAW_IN_RADIANS else yaw
    if FLIP_Y:
        cy, yaw_deg = -cy, -yaw_deg
    return cx + X_OFFSET, cy + Y_OFFSET, yaw_deg + YAW_OFFSET_DEG


def _merge(dst, patch):
    """RFC 7396 JSON merge-patch (null deletes)."""
    if not isinstance(patch, dict):
        return patch
    if not isinstance(dst, dict):
        dst = {}
    for k, v in patch.items():
        if v is None:
            dst.pop(k, None)
        else:
            dst[k] = _merge(dst.get(k), v)
    return dst


def apply_event(thing, path, action, value):
    """Apply a Ditto event (path like '/features/location/properties/x') to the local thing."""
    parts = [p for p in path.split("/") if p]
    if action == "deleted":
        if not parts:
            thing.clear()
            return
        node = thing
        for p in parts[:-1]:
            node = node.get(p, {}) if isinstance(node, dict) else {}
        if isinstance(node, dict):
            node.pop(parts[-1], None)
        return
    if not parts:  # whole thing
        if action == "merged":
            _merge(thing, value)
        elif isinstance(value, dict):
            thing.clear()
            thing.update(value)
        return
    node = thing
    for p in parts[:-1]:
        if not isinstance(node.get(p), dict):
            node[p] = {}
        node = node[p]
    last = parts[-1]
    node[last] = _merge(node.get(last), value) if action == "merged" else value


# --- Bridge ---------------------------------------------------------------
class DittoCarlaBridge:
    def __init__(self):
        self.thing = {}
        self.lock = threading.Lock()
        self.ws = None
        self.running = True
        self.last_log = 0.0
        self.warned_missing = False
        self.dirty = False

        logger.info("Connecting to CARLA at %s:%d...", CARLA_HOST, CARLA_PORT)
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(20.0)
        self.world = (self.client.load_world(CARLA_MAP)
                      if CARLA_MAP and CARLA_MAP not in self.client.get_world().get_map().name
                      else self.client.get_world())
        self.map = self.world.get_map()

        bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        bp.set_attribute("role_name", "ego_ditto_twin")
        sp = self.map.get_spawn_points()[0]
        self.actor = self.world.spawn_actor(bp, sp)
        self.actor.set_simulate_physics(False)  # Ditto owns the pose
        logger.info("Spawned twin vehicle (id %d) at spawn point #0", self.actor.id)

    # --- websocket callbacks ---
    def on_open(self, ws):
        cmd = "START-SEND-EVENTS" + (f"?filter={FILTER}" if FILTER else "")
        logger.info("WebSocket open, sending %s", cmd)
        ws.send(cmd)

    def on_error(self, ws, error):
        logger.error("WebSocket error: %s", error)

    def on_close(self, ws, code, msg):
        logger.warning("WebSocket closed [%s]: %s", code, msg)

    def on_message(self, ws, message):
        if isinstance(message, bytes):
            message = message.decode("utf-8", "replace")
        if message.startswith("START-SEND-EVENTS:ACK"):
            logger.info("Subscription confirmed, waiting for events...")
            return
        try:
            env = json.loads(message)
        except json.JSONDecodeError:
            logger.debug("Non-JSON message: %s", message[:200])
            return
        if not isinstance(env, dict):
            return

        topic = env.get("topic", "")  # namespace/name/things/twin/events/modified
        if THING_ID and "/" in topic:
            ns, name = (topic.split("/") + ["", ""])[:2]
            if f"{ns}:{name}" != THING_ID:
                return

        action = topic.rsplit("/", 1)[-1] if topic else "modified"
        path = env.get("path", "/")
        value = env.get("value")
        with self.lock:
            apply_event(self.thing, path, action, value)
            self.dirty = True
        self.apply_pose(path)

    # --- pose -> CARLA ---
    def apply_pose(self, last_path=""):
        with self.lock:
            pose, missing = parse_pose(self.thing)
            snapshot = json.dumps(self.thing)
        if missing:
            if not self.warned_missing:
                logger.error("Event %s received but no x/y found yet. Thing so far: %s",
                             last_path, snapshot[:600])
                self.warned_missing = True
            return
        self.warned_missing = False

        cx, cy, yaw = map_to_carla(pose["x"], pose["y"], pose["yaw"] or 0.0)
        z = pose["z"] if pose["z"] is not None else 0.5
        if SNAP_TO_GROUND:
            wp = self.map.get_waypoint(carla.Location(x=cx, y=cy, z=0.0),
                                       project_to_road=True, lane_type=carla.LaneType.Any)
            if wp is not None:
                z = wp.transform.location.z + GROUND_CLEARANCE

        self.actor.set_transform(carla.Transform(
            carla.Location(x=cx, y=cy, z=z), carla.Rotation(yaw=yaw)))

        now = time.time()
        if now - self.last_log >= LOG_INTERVAL:
            self.last_log = now
            logger.info("Twin move [%s] ditto(x=%.2f y=%.2f yaw=%s) -> carla(x=%.2f y=%.2f z=%.2f yaw=%.1f)",
                        last_path, pose["x"], pose["y"], pose["yaw"], cx, cy, z, yaw)

    # --- lifecycle ---
    def _run_ws(self):
        token = base64.b64encode(f"{DITTO_AUTH[0]}:{DITTO_AUTH[1]}".encode()).decode()
        while self.running:
            self.ws = websocket.WebSocketApp(
                DITTO_WS_URL,
                header=[f"Authorization: Basic {token}"],
                on_open=self.on_open, on_message=self.on_message,
                on_error=self.on_error, on_close=self.on_close,
            )
            self.ws.run_forever(ping_interval=20, ping_timeout=10)
            if self.running:
                logger.info("Reconnecting in %.0fs...", RECONNECT_DELAY)
                time.sleep(RECONNECT_DELAY)

    def start(self):
        threading.Thread(target=self._run_ws, daemon=True).start()

    def cleanup(self):
        self.running = False
        if self.ws is not None:
            self.ws.close()
        if self.actor is not None:
            self.actor.destroy()
            logger.info("Destroyed twin vehicle.")


if __name__ == "__main__":
    bridge = DittoCarlaBridge()
    bridge.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.cleanup()
        sys.exit(0)