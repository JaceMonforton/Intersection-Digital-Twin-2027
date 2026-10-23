"""
CARLA digital twin mirror: polls Eclipse Ditto and moves a vehicle in CARLA to match.
Run AFTER the CARLA server is up. Press ESC to quit.
"""
import json
import math
import threading
import time

import carla
import numpy as np
import pygame
import requests

# --- Configuration ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
CARLA_MAP = "Town02"

DITTO_BASE = "http://192.168.0.100:8080/api/2/things/"
THING_ID = "org.ovin:my-vehicle"
DITTO_AUTH = ("ditto", "ditto")

FPS = 30
LOG_INTERVAL = 0.5  # seconds between telemetry log lines (0 = log every poll)

# --- Mapping from the physical / Ditto frame to the CARLA world ---
# Order applied: scale -> flip -> offset (offsets are in CARLA metres).
POSITION_SCALE = 1.0    # e.g. 10.0 for a 1:10 scale testbed
FLIP_Y = False          # ROS is right-handed (y left); CARLA is left-handed (y right).
                        # True also negates the yaw.
X_OFFSET = 0.0
Y_OFFSET = 0.0
YAW_IN_RADIANS = False  # True if the publisher sends yaw in radians (ROS style)
YAW_OFFSET_DEG = 0.0

# --- Ground handling ---
SNAP_TO_GROUND = True   # use the road height at (x, y) instead of the published z
GROUND_CLEARANCE = 0.3  # metres above the road surface

STALE_AFTER = 2.0       # seconds without a pose change before the HUD warns

# --- Dead reckoning (used when the thing has NO x/y, only Speed + SteeringAngle) ---
DEAD_RECKON = True
DR_SPEED_TO_MS = 1.0        # multiply Ditto "Speed" to get m/s (0.2778 if it is km/h)
DR_DISTANCE_SCALE = 1.0     # 10.0 to render a 1:10 robot as a full-size car
DR_STEER_IN_DEGREES = False # True if SteeringAngle is degrees; False = radians
DR_LEFT_POSITIVE = True     # True if positive steering = turn left (ROS convention)
DR_WHEELBASE = 2.87         # metres in CARLA (Tesla Model 3)
DR_SPEED_DEADBAND = 0.05    # ignore |speed| below this (sensor noise at standstill)


# ---------------------------------------------------------------------------
# Pure helpers (no CARLA needed)
# ---------------------------------------------------------------------------
def _dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def find_signal(thing, group, field):
    """Look up a value in a Ditto thing, trying the layouts we have seen:
      features.<group>.properties.<field>   (standard Ditto feature)
      features.<group>.<field>              (flat)
      attributes.<group>.<field>            (older attribute layout)
    Returns None if it isn't there."""
    for path in (
        ("features", group, "properties", field),
        ("features", group, field),
        ("attributes", group, field),
    ):
        value = _dig(thing, *path)
        if value is not None:
            return value
    return None


def parse_pose(thing):
    """Return (pose, missing). pose values are None where absent;
    missing lists which of x / y could not be found."""
    pose = {
        "x": find_signal(thing, "location", "x"),
        "y": find_signal(thing, "location", "y"),
        "z": find_signal(thing, "location", "z"),
        "yaw": find_signal(thing, "orientation", "yaw"),
        "speed_kmh": find_signal(thing, "kuksa", "Vehicle.Speed"),
    }
    if pose["yaw"] is None:
        pose["yaw"] = find_signal(thing, "location", "yaw")
    missing = [k for k in ("x", "y") if pose[k] is None]
    return pose, missing


def parse_drive(thing):
    """(speed, steering) from features.Speed/SteeringAngle .properties.value (either may be None)."""
    def val(name):
        v = _dig(thing, "features", name, "properties", "value")
        if v is None:
            v = _dig(thing, "features", name, "value")
        try:
            return None if v is None else float(v)
        except (TypeError, ValueError):
            return None
    return val("Speed"), val("SteeringAngle")


def step_bicycle(x, y, yaw_deg, speed, steer, dt):
    """Kinematic bicycle model in the CARLA frame. Returns new (x, y, yaw_deg)."""
    v = speed * DR_SPEED_TO_MS * DR_DISTANCE_SCALE
    if abs(speed * DR_SPEED_TO_MS) < DR_SPEED_DEADBAND:
        v = 0.0
    delta = math.radians(steer) if DR_STEER_IN_DEGREES else steer
    # CARLA yaw increases clockwise (left-handed), so a left turn lowers yaw
    sign = -1.0 if DR_LEFT_POSITIVE else 1.0
    yaw_rate = sign * (v / DR_WHEELBASE) * math.tan(delta)  # rad/s
    yaw = math.radians(yaw_deg)
    x += v * math.cos(yaw) * dt
    y += v * math.sin(yaw) * dt
    return x, y, math.degrees(yaw + yaw_rate * dt)


def map_to_carla(x, y, yaw):
    """Map a Ditto pose (x, y, yaw) into the CARLA frame; yaw returned in degrees."""
    x *= POSITION_SCALE
    y *= POSITION_SCALE
    if YAW_IN_RADIANS:
        yaw = math.degrees(yaw)
    if FLIP_Y:
        y = -y
        yaw = -yaw
    return x + X_OFFSET, y + Y_OFFSET, yaw + YAW_OFFSET_DEG


def carla_img_to_pygame_surface(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3][:, :, ::-1]  # BGRA -> RGB
    return pygame.surfarray.make_surface(array.swapaxes(0, 1))


# ---------------------------------------------------------------------------
class DittoTelemetryConsumer(threading.Thread):
    """Polls Ditto in the background so Pygame rendering never stalls."""

    def __init__(self, thing_id, base_url, auth):
        super().__init__(daemon=True)
        self.url = f"{base_url}{thing_id}"
        self.session = requests.Session()
        self.session.auth = auth
        self.running = True
        self.last_error = None
        self._last_log = 0.0
        self._last_pose = None
        self.latest_telemetry = {
            "x": 0.0, "y": 0.0, "z": 0.5, "yaw": 0.0,
            "speed_kmh": 0.0,
            "connected": False,   # last HTTP poll succeeded
            "has_pose": False,    # payload contained location x/y
            "last_change": 0.0,   # time the pose last changed
            "has_drive": False,   # payload contained Speed / SteeringAngle
            "drive_speed": 0.0,
            "drive_steer": 0.0,
        }

    def _log_error(self, msg):
        if msg != self.last_error:  # only print when the message changes
            print(f"[DITTO POLL] {msg}")
            self.last_error = msg

    def run(self):
        t = self.latest_telemetry
        while self.running:
            try:
                response = self.session.get(self.url, timeout=(3.0, 5.0))
                if response.status_code == 200:
                    thing = response.json()
                    if not t["connected"]:
                        print("[DITTO POLL] Connected, receiving telemetry.")
                        print(f"[DITTO RAW] {json.dumps(thing)[:1000]}")
                    t["connected"] = True

                    speed, steer = parse_drive(thing)
                    if speed is not None and steer is not None:
                        t["drive_speed"], t["drive_steer"] = speed, steer
                        t["has_drive"] = True
                        t["last_change"] = time.time()

                    pose, missing = parse_pose(thing)
                    if missing:
                        t["has_pose"] = False
                        if t["has_drive"] and DEAD_RECKON:
                            now = time.time()
                            if now - self._last_log >= LOG_INTERVAL:
                                print(f"[DITTO RX] no x/y -> dead reckoning: "
                                      f"speed={speed} steer={steer}")
                                self._last_log = now
                        else:
                            self._log_error(
                                f"Connected, but no {'/'.join(missing)} found in the thing. "
                                f"Payload: {json.dumps(thing)[:600]}"
                            )
                    else:
                        self.last_error = None
                        t["has_pose"] = True
                        for key in ("x", "y", "z", "yaw", "speed_kmh"):
                            if pose[key] is not None:
                                t[key] = float(pose[key])

                        current = (t["x"], t["y"], t["yaw"])
                        if current != self._last_pose:
                            self._last_pose = current
                            t["last_change"] = time.time()

                        now = time.time()
                        if now - self._last_log >= LOG_INTERVAL:
                            print(
                                f"[DITTO RX] x={t['x']:.2f} y={t['y']:.2f} z={t['z']:.2f} "
                                f"yaw={t['yaw']:.2f} speed={t['speed_kmh']:.1f} km/h"
                            )
                            self._last_log = now
                else:
                    t["connected"] = False
                    self._log_error(f"HTTP {response.status_code}: {response.text[:200]}")
            except Exception as e:  # noqa: BLE001
                t["connected"] = False
                self._log_error(f"{type(e).__name__}: {e}")

            time.sleep(0.05)

    def stop(self):
        self.running = False


class DigitalTwinAgent:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("CARLA Digital Twin - Eclipse Ditto Mirror")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 18)

        self.client = None
        self.world = None
        self.carla_map = None
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None

        self.current_surface = None
        self.last_collision_time = 0
        self.collision_intensity = 0.0
        self.actor_list = []
        self._dr = None
        self._last_step = 0.0

        self.telemetry_consumer = DittoTelemetryConsumer(THING_ID, DITTO_BASE, DITTO_AUTH)

    def setup_world(self):
        print(f"[CARLA] Connecting to {CARLA_HOST}:{CARLA_PORT}...")
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(60.0)  # map loads can be slow

        current_map = self.client.get_world().get_map().name
        if CARLA_MAP not in current_map:
            print(f"[CARLA] Loading map {CARLA_MAP}...")
            self.world = self.client.load_world(CARLA_MAP)
        else:
            self.world = self.client.get_world()
        self.carla_map = self.world.get_map()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / FPS
        self.world.apply_settings(settings)

        # Handy for choosing publisher start coordinates
        spawn_points = self.carla_map.get_spawn_points()
        print("[CARLA] First spawn points (use for --x/--y/--yaw on the publisher):")
        for i, sp in enumerate(spawn_points[:5]):
            print(f"   #{i}: x={sp.location.x:.1f} y={sp.location.y:.1f} yaw={sp.rotation.yaw:.1f}")

    def spawn_ego_vehicle(self):
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        vehicle_bp.set_attribute("role_name", "ego_ditto_twin")

        spawn_points = self.carla_map.get_spawn_points()
        candidates = spawn_points if spawn_points else [carla.Transform()]
        for sp in candidates:
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, sp)
            if self.vehicle:
                break
        if not self.vehicle:
            raise RuntimeError("Could not spawn vehicle at any spawn point.")

        # Ditto drives the pose; stop physics from fighting set_transform
        self.vehicle.set_simulate_physics(False)
        self.actor_list.append(self.vehicle)
        print(f"[CARLA] Spawned vehicle ID {self.vehicle.id}.")

    def attach_sensors(self):
        bp_lib = self.world.get_blueprint_library()

        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(WINDOW_WIDTH))
        camera_bp.set_attribute("image_size_y", str(WINDOW_HEIGHT))
        camera_bp.set_attribute("fov", "90")
        cam_tf = carla.Transform(carla.Location(x=-5.0, z=2.2), carla.Rotation(pitch=-12.0))
        self.camera = self.world.spawn_actor(camera_bp, cam_tf, attach_to=self.vehicle)
        self.camera.listen(self.on_camera_feed)
        self.actor_list.append(self.camera)

        col_bp = bp_lib.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(self.on_collision)
        self.actor_list.append(self.collision_sensor)

    def on_camera_feed(self, image):
        self.current_surface = carla_img_to_pygame_surface(image)

    def on_collision(self, event):
        i = event.normal_impulse
        self.collision_intensity = math.sqrt(i.x ** 2 + i.y ** 2 + i.z ** 2)
        self.last_collision_time = time.time()
        print(f"[ALERT] Collision with {event.other_actor.type_id} ({self.collision_intensity:.2f})")

    def update_vehicle_from_ditto(self):
        t = self.telemetry_consumer.latest_telemetry
        now = time.time()
        dt = min(now - self._last_step, 0.25) if self._last_step else 0.0
        self._last_step = now

        if not t["connected"]:
            return
        if t["has_pose"]:
            x, y, yaw = map_to_carla(t["x"], t["y"], t["yaw"])
            self._dr = None  # re-seed dead reckoning from the real pose if pose later vanishes
        elif DEAD_RECKON and t["has_drive"]:
            if self._dr is None:
                tf = self.vehicle.get_transform()
                self._dr = (tf.location.x, tf.location.y, tf.rotation.yaw)
            x, y, yaw = step_bicycle(*self._dr, t["drive_speed"], t["drive_steer"], dt)
            self._dr = (x, y, yaw)
        else:
            return

        if SNAP_TO_GROUND:
            wp = self.carla_map.get_waypoint(
                carla.Location(x=x, y=y, z=0.0),
                project_to_road=True,
                lane_type=carla.LaneType.Any,
            )
            z = (wp.transform.location.z if wp else 0.0) + GROUND_CLEARANCE
        else:
            z = t["z"] * POSITION_SCALE

        self.vehicle.set_transform(
            carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
            )
        )

    def render_hud(self):
        if not self.vehicle:
            return
        t = self.telemetry_consumer.latest_telemetry
        if not t["connected"]:
            status, status_color = "OFFLINE / SEARCHING", (255, 80, 80)
        elif not t["has_pose"] and DEAD_RECKON and t["has_drive"]:
            status, status_color = "DEAD RECKONING (speed + steering only)", (255, 220, 0)
        elif not t["has_pose"]:
            status, status_color = "CONNECTED - NO POSE IN PAYLOAD", (255, 170, 0)
        else:
            status, status_color = "SYNCED", (0, 255, 120)
        tf = self.vehicle.get_transform()

        age = time.time() - t["last_change"] if t["last_change"] else None
        if age is None:
            age_text, age_color = "never", (255, 170, 0)
        else:
            stale = age > STALE_AFTER
            age_text = f"{age:4.1f} s ago" + ("  (STALE)" if stale else "")
            age_color = (255, 170, 0) if stale else (255, 255, 255)

        lines = [
            ("MODE:      DITTO TWIN MIRROR", (0, 200, 255)),
            (f"Status:    {status}", status_color),
            (f"Target ID: {THING_ID}", (255, 255, 255)),
            (f"Speed:     {t['speed_kmh']:5.1f} km/h (Ditto)", (255, 255, 255)),
            (f"Ditto raw: X={t['x']:.2f}, Y={t['y']:.2f}, yaw={t['yaw']:.2f}", (255, 255, 255)),
            (f"CARLA pos: X={tf.location.x:.1f}, Y={tf.location.y:.1f}, Z={tf.location.z:.1f}",
             (255, 255, 255)),
            (f"Pose changed: {age_text}", age_color),
        ]

        bg = pygame.Surface((470, len(lines) * 22 + 15))
        bg.set_alpha(180)
        bg.fill((0, 0, 0))
        self.display.blit(bg, (10, 10))

        for i, (line, color) in enumerate(lines):
            self.display.blit(self.font.render(line, True, color), (20, 18 + i * 22))

        if time.time() - self.last_collision_time < 1.5:
            warn = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            warn.set_alpha(80)
            warn.fill((255, 0, 0))
            self.display.blit(warn, (0, 0))
            msg = f"!!! COLLISION (Impulse: {self.collision_intensity:.1f}) !!!"
            self.display.blit(self.font.render(msg, True, (255, 255, 255)), (WINDOW_WIDTH // 2 - 200, 30))

    def run(self):
        try:
            self.setup_world()
            self.spawn_ego_vehicle()
            self.attach_sensors()
            self.telemetry_consumer.start()

            print("\n[READY] Listening to Ditto. Press ESC or close the window to exit.\n")

            running = True
            while running:
                self.update_vehicle_from_ditto()
                self.world.tick()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        running = False

                if self.current_surface is not None:
                    self.display.blit(self.current_surface, (0, 0))
                self.render_hud()
                pygame.display.flip()
                self.clock.tick(FPS)
        finally:
            self.cleanup()

    def cleanup(self):
        print("[CLEANUP] Stopping threads and removing actors...")
        self.telemetry_consumer.stop()

        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

        for actor in reversed(self.actor_list):
            try:
                if actor is not None and actor.is_alive:
                    if hasattr(actor, "stop"):
                        actor.stop()
                    actor.destroy()
            except Exception:  # noqa: BLE001, S110
                pass

        pygame.quit()
        print("[CLEANUP] Done.")


if __name__ == "__main__":
    DigitalTwinAgent().run()