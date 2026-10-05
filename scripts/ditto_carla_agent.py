"""
CARLA digital twin mirror: polls Eclipse Ditto and moves a vehicle in CARLA to match.
Run AFTER the CARLA server is up. Press ESC to quit.
"""
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

DITTO_BASE = "http://localhost:8080/api/2/things/"
THING_ID = "org.example:vehicle_01"
DITTO_AUTH = ("ditto", "ditto")

FPS = 30


def carla_img_to_pygame_surface(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3][:, :, ::-1]  # BGRA -> RGB
    return pygame.surfarray.make_surface(array.swapaxes(0, 1))


class DittoTelemetryConsumer(threading.Thread):
    """Polls Ditto in the background so Pygame rendering never stalls."""

    def __init__(self, thing_id, base_url, auth):
        super().__init__(daemon=True)
        self.url = f"{base_url}{thing_id}"
        self.session = requests.Session()
        self.session.auth = auth
        self.running = True
        self.last_error = None
        self.latest_telemetry = {
            "x": 0.0, "y": 0.0, "z": 0.5, "yaw": 0.0,
            "speed_kmh": 0.0, "connected": False,
        }

    def _log_error(self, msg):
        if msg != self.last_error:  # only print when the error changes
            print(f"[DITTO POLL] {msg}")
            self.last_error = msg

    def run(self):
        t = self.latest_telemetry
        while self.running:
            try:
                response = self.session.get(self.url, timeout=2.0)
                if response.status_code == 200:
                    attrs = response.json().get("attributes", {})
                    location = attrs.get("location", {})
                    orientation = attrs.get("orientation", {})
                    kuksa_speed = attrs.get("kuksa", {}).get("Vehicle.Speed")

                    t["x"] = float(location.get("x", t["x"]))
                    t["y"] = float(location.get("y", t["y"]))
                    t["z"] = float(location.get("z", t["z"]))
                    t["yaw"] = float(orientation.get("yaw", t["yaw"]))
                    if kuksa_speed is not None:
                        t["speed_kmh"] = float(kuksa_speed)
                    else:
                        t["speed_kmh"] = float(attrs.get("speed_kmh", 0.0))

                    if not t["connected"]:
                        print("[DITTO POLL] Connected, receiving telemetry.")
                    t["connected"] = True
                    self.last_error = None
                else:
                    t["connected"] = False
                    self._log_error(f"HTTP {response.status_code}: {response.text[:200]}")
            except Exception as e: # noqa: BLE001
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
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None

        self.current_surface = None
        self.last_collision_time = 0
        self.collision_intensity = 0.0
        self.actor_list = []

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

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / FPS
        self.world.apply_settings(settings)

        # Handy for choosing publisher start coordinates
        spawn_points = self.world.get_map().get_spawn_points()
        print("[CARLA] First spawn points (use for --x/--y/--yaw on the publisher):")
        for i, sp in enumerate(spawn_points[:5]):
            print(f"   #{i}: x={sp.location.x:.1f} y={sp.location.y:.1f} yaw={sp.rotation.yaw:.1f}")

    def spawn_ego_vehicle(self):
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        vehicle_bp.set_attribute("role_name", "ego_ditto_twin")

        spawn_points = self.world.get_map().get_spawn_points()
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
        if not t["connected"]:
            return
        self.vehicle.set_transform(
            carla.Transform(
                carla.Location(x=t["x"], y=t["y"], z=t["z"]),
                carla.Rotation(pitch=0.0, yaw=t["yaw"], roll=0.0),
            )
        )

    def render_hud(self):
        if not self.vehicle:
            return
        t = self.telemetry_consumer.latest_telemetry
        status = "SYNCED" if t["connected"] else "OFFLINE / SEARCHING"
        status_color = (0, 255, 120) if t["connected"] else (255, 80, 80)
        tf = self.vehicle.get_transform()

        lines = [
            "MODE:      DITTO TWIN MIRROR",
            f"Status:    {status}",
            f"Target ID: {THING_ID}",
            f"Speed:     {t['speed_kmh']:5.1f} km/h (Ditto)",
            f"Ditto pos: X={t['x']:.1f}, Y={t['y']:.1f}",
            f"CARLA pos: X={tf.location.x:.1f}, Y={tf.location.y:.1f}, Z={tf.location.z:.1f}",
        ]

        bg = pygame.Surface((400, len(lines) * 22 + 15))
        bg.set_alpha(180)
        bg.fill((0, 0, 0))
        self.display.blit(bg, (10, 10))

        for i, line in enumerate(lines):
            color = (255, 255, 255)
            if i == 0:
                color = (0, 200, 255)
            elif i == 1:
                color = status_color
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
                    if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
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
            except Exception: # noqa: BLE001, S110
                pass

        pygame.quit()
        print("[CLEANUP] Done.")


if __name__ == "__main__":
    DigitalTwinAgent().run()