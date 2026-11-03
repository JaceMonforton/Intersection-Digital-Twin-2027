"""
A window following the ML car: first person on the left, third person on the right.

It attaches to the running simulation and waits for vehicle_ml to appear, and it picks the car up
again whenever a new stage respawns it. Closing the window only closes the view. run_simulation.py
opens it automatically.

Usage:
    python scripts/ml_car_view.py
"""
import math
import signal
import sys

import carla
import numpy as np
import pygame
import requests

# --- Configuration ---
CARLA_HOST = "localhost"
CARLA_PORT = 2000
GATE_URL = "http://localhost:8080/api/2/things/org.example:vehicle_ml/features/SafetyGate/properties"
ML_ROLE = "vehicle_ml"

VIEW_WIDTH = 800
VIEW_HEIGHT = 450
DIVIDER = 4           # pixels between the two views
BANNER_HEIGHT = 44    # pixels of status along the bottom
CAMERA_FOV = "90"
FPS = 30

# Where each camera sits, relative to the car: x forward, y right, z up (metres)
FIRST_PERSON = {"location": {"x": 0.15, "y": -0.35, "z": 1.25}, "pitch": -3.0, "label": "1ST PERSON"}
THIRD_PERSON = {"location": {"x": -6.5, "y": 0.0, "z": 2.8}, "pitch": -12.0, "label": "3RD PERSON"}

GATE_POLL_SECONDS = 0.5
TEXT_COLOR = (255, 255, 255)
ML_COLOR = (255, 0, 255)
BRAKE_COLOR = (255, 60, 60)
OK_COLOR = (0, 220, 120)
BANNER_COLOR = (20, 20, 20)
BACKGROUND = (0, 0, 0)


# ---------------------------------------------------------------------------
# Pure helpers (no CARLA needed)
# ---------------------------------------------------------------------------
def bgra_to_rgb(raw_data, width, height):
    """A CARLA camera image's BGRA bytes as a height x width x 3 RGB array."""
    array = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width, 4)
    return array[:, :, :3][:, :, ::-1]


def window_size():
    return 2 * VIEW_WIDTH + DIVIDER, VIEW_HEIGHT + BANNER_HEIGHT


def speed_from_positions(previous, current, seconds):
    """m/s between two (x, y) positions. The ML car is moved directly, so CARLA reports zero for it."""
    if previous is None or seconds <= 0.0:
        return 0.0
    return math.hypot(current[0] - previous[0], current[1] - previous[1]) / seconds


def status_text(speed, braking):
    state = "EMERGENCY BRAKE" if braking else "driving" if speed >= 0.5 else "stopped"
    return f"{speed * 3.6:5.1f} km/h   {state}"


# ---------------------------------------------------------------------------
# CARLA
# ---------------------------------------------------------------------------
class ChaseCamera:
    """One camera attached to the car, holding on to its latest frame."""

    def __init__(self, world, vehicle, spec):
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(VIEW_WIDTH))
        bp.set_attribute("image_size_y", str(VIEW_HEIGHT))
        bp.set_attribute("fov", CAMERA_FOV)
        transform = carla.Transform(carla.Location(**spec["location"]), carla.Rotation(pitch=spec["pitch"]))
        self.actor = world.spawn_actor(bp, transform, attach_to=vehicle)
        self.label = spec["label"]
        self.frame = None
        self.actor.listen(self._on_image)

    def _on_image(self, image):
        self.frame = bgra_to_rgb(image.raw_data, image.width, image.height)

    def destroy(self):
        self.actor.stop()
        self.actor.destroy()


class MlCarView:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(window_size())
        pygame.display.set_caption("ML Car View")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 20)
        self.big_font = pygame.font.SysFont("Consolas", 26, bold=True)

        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(30.0)
        self.world = client.get_world()
        self.session = requests.Session()
        self.vehicle = None
        self.cameras = []
        self.previous = None
        self.previous_time = 0.0
        self.speed = 0.0
        self.braking = False
        self.last_gate_poll = 0.0

    def _find_ml_car(self):
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name") == ML_ROLE:
                return actor
        return None

    def _drop_cameras(self):
        for camera in self.cameras:
            try:
                camera.destroy()
            except RuntimeError:
                pass  # went with its car
        self.cameras = []

    def _follow(self, vehicle):
        self._drop_cameras()
        self.vehicle = vehicle
        self.previous = None
        self.speed = 0.0
        if vehicle is not None:
            self.cameras = [ChaseCamera(self.world, vehicle, FIRST_PERSON),
                            ChaseCamera(self.world, vehicle, THIRD_PERSON)]
            print(f"[VIEW] Following vehicle_ml (actor {vehicle.id}).", flush=True)

    def _update_status(self, sim_time):
        location = self.vehicle.get_location()
        position = (location.x, location.y)
        elapsed = sim_time - self.previous_time
        if elapsed > 0.0:
            self.speed = speed_from_positions(self.previous, position, elapsed)
            self.previous, self.previous_time = position, sim_time
        if sim_time - self.last_gate_poll >= GATE_POLL_SECONDS:
            self.last_gate_poll = sim_time
            try:
                self.braking = self.session.get(GATE_URL, timeout=0.3).json().get("emergency_brake") is True
            except (requests.RequestException, ValueError):
                pass  # keep the last known state; this window is only a display

    def _draw_view(self, camera, left):
        if camera.frame is not None:
            self.display.blit(pygame.surfarray.make_surface(camera.frame.swapaxes(0, 1)), (left, 0))
        label = self.font.render(camera.label, True, TEXT_COLOR)
        backing = pygame.Surface((label.get_width() + 16, label.get_height() + 8))
        backing.set_alpha(170)
        self.display.blit(backing, (left + 10, 10))
        self.display.blit(label, (left + 18, 14))

    def _draw(self):
        self.display.fill(BACKGROUND)
        width, height = window_size()
        if not self.cameras:
            message = self.big_font.render("Waiting for the ML car...", True, ML_COLOR)
            self.display.blit(message, message.get_rect(center=(width // 2, height // 2)))
        else:
            self._draw_view(self.cameras[0], 0)
            self._draw_view(self.cameras[1], VIEW_WIDTH + DIVIDER)
            pygame.draw.rect(self.display, BANNER_COLOR, (0, VIEW_HEIGHT, width, BANNER_HEIGHT))
            self.display.blit(self.big_font.render("ML CAR", True, ML_COLOR), (16, VIEW_HEIGHT + 8))
            color = BRAKE_COLOR if self.braking else OK_COLOR
            self.display.blit(self.big_font.render(status_text(self.speed, self.braking), True, color),
                              (160, VIEW_HEIGHT + 8))
        pygame.display.flip()

    def run(self):
        print("[VIEW] Waiting for the ML car. Close the window to exit.", flush=True)
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
            try:
                snapshot = self.world.wait_for_tick(0.2)
            except RuntimeError:
                snapshot = None  # between ticks, or the simulation has paused; keep the window alive
            if snapshot is not None:
                vehicle = self._find_ml_car()
                if (vehicle.id if vehicle else None) != (self.vehicle.id if self.vehicle else None):
                    self._follow(vehicle)
                if self.vehicle is not None:
                    self._update_status(snapshot.timestamp.elapsed_seconds)
            self._draw()
            self.clock.tick(FPS)

    def cleanup(self):
        self._drop_cameras()
        pygame.quit()


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    for name in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _raise_keyboard_interrupt)
    view = MlCarView()
    try:
        view.run()
    except KeyboardInterrupt:
        pass
    finally:
        view.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
