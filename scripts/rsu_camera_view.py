"""
A live window of the two roadside cameras, for the camera_and_rsu config: what a camera at each roadside mount sees,
with the roadside camera detector's boxes (models/rsu_camera_detector.candidate.onnx, run by the processor on its
own cameras in the same places) drawn over it. CARLA's true boxes are drawn dimly underneath, for comparing by eye.

    python run_simulation.py --camera_and_rsu
    python scripts/rsu_camera_view.py     # alongside a processor started with --detector camera --lidar-view

It attaches to the running simulation, waits for a tick, then spawns its own two cameras; it never ticks the world.
Closing the window only closes the view.
"""
import math
import signal
import sys
import time

import numpy as np
from camera_features import FOV, IMAGE_HEIGHT, IMAGE_WIDTH, aim_at_centre, project
from central_gpu_processor import CARLA_HOST, CARLA_PORT, LIDAR_MOUNTS, find_junction
from detection_features import MODEL_GROUND_Z, PEDESTRIAN
from lidar_view import FEED_NAME, STOPPED_AFTER, SweepFeed

JUNCTION_ID = 189
HEADER = 64             # pixels of text above the images
FPS = 20
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
TRACK_CAR, TRACK_PERSON = (70, 230, 120), (240, 90, 220)
TRUE_BOX, BAD, TEXT, TITLE, BACKGROUND = (120, 120, 125), (240, 90, 80), (205, 205, 205), (240, 240, 240), (16, 16, 18)


def box_corners_world(row, center):
    """The 8 corners of a box row in the roadside model's frame (x, y left, z, length, width, height, yaw radians
    counter-clockwise), in CARLA's frame: bottom four, then top four. center is the junction's (x, y, z)."""
    x, y, z = center[0] + float(row[0]), center[1] - float(row[1]), center[2] + float(row[2]) - MODEL_GROUND_Z
    half_length, half_width, half_height, yaw = float(row[3]) / 2, float(row[4]) / 2, float(row[5]) / 2, -float(row[6])
    cos, sin = math.cos(yaw), math.sin(yaw)
    flat = [(a, b) for a, b in ((half_length, half_width), (half_length, -half_width), (-half_length, -half_width), (-half_length, half_width))]
    return np.array([(x + a * cos - b * sin, y + a * sin + b * cos, z + c) for c in (-half_height, half_height) for a, b in flat])


class RoadsideCamera:
    """One camera at a roadside mount, holding on to its latest frame."""

    def __init__(self, carla, world, center, mount):
        from ml_car_view import bgra_to_rgb

        self.bgra_to_rgb = bgra_to_rgb
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        for key, value in (("image_size_x", IMAGE_WIDTH), ("image_size_y", IMAGE_HEIGHT), ("fov", FOV)):
            bp.set_attribute(key, str(value))
        pitch, yaw = aim_at_centre(mount)
        self.transform = carla.Transform(carla.Location(x=center.x + mount[0], y=center.y + mount[1], z=center.z + mount[2]),
                                         carla.Rotation(pitch=pitch, yaw=yaw))
        self.world_to_camera = np.array(self.transform.get_inverse_matrix())
        self.actor = world.spawn_actor(bp, self.transform)
        self.frame = None
        self.actor.listen(self._on_image)

    def _on_image(self, image):
        self.frame = self.bgra_to_rgb(image.raw_data, image.width, image.height)

    def destroy(self):
        self.actor.stop()
        self.actor.destroy()


class Window:
    def __init__(self, pygame, carla):
        self.pygame = pygame
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(30.0)
        world = client.get_world()
        world.wait_for_tick(30.0)  # a new client reads the world wrongly until its first tick
        location = find_junction(world, JUNCTION_ID).bounding_box.location
        self.center = (location.x, location.y, location.z)
        self.cameras = [RoadsideCamera(carla, world, location, mount) for mount in LIDAR_MOUNTS]
        pygame.init()
        self.screen = pygame.display.set_mode((IMAGE_WIDTH * len(self.cameras), IMAGE_HEIGHT + HEADER))
        pygame.display.set_caption("Roadside cameras: camera detector")
        self.small, self.large = pygame.font.SysFont("segoeui", 14), pygame.font.SysFont("segoeui", 19, bold=True)
        self.clock = pygame.time.Clock()
        self.feed = None
        self.state = None
        self.updated = None

    def poll(self, now):
        self.feed = self.feed or SweepFeed.attach(FEED_NAME)
        state = self.feed.read() if self.feed else None
        if state is not None:
            self.state, self.updated = state, now

    def _boxes(self, rows, camera, offset, colour, width):
        for row in rows:
            pixels = project(box_corners_world(row, self.center), camera.world_to_camera)
            for a, b in EDGES:
                if not np.isnan(pixels[[a, b]]).any():
                    self.pygame.draw.line(self.screen, colour, (pixels[a, 0] + offset, pixels[a, 1] + HEADER), (pixels[b, 0] + offset, pixels[b, 1] + HEADER), width)

    def draw(self, now):
        pygame = self.pygame
        self.screen.fill(BACKGROUND)
        title = "Roadside cameras: the camera detection model's boxes"
        self.screen.blit(self.large.render(title, True, TITLE), (16, 8))
        if self.state is None:
            words, colour = "Waiting for a processor started with --detector camera --lidar-view.", BAD
        elif now - self.updated > STOPPED_AFTER:
            words, colour = f"NO NEW DETECTIONS for {now - self.updated:.0f} s", BAD
        else:
            detections = self.state["detections"]
            people = int((detections[:, 8] == PEDESTRIAN).sum()) if len(detections) else 0
            words, colour = (f"{len(detections) - people} vehicles and {people} people reported, of {len(self.state['true_boxes'])} in CARLA. "
                             "Green: detected vehicle, magenta: detected person, grey: CARLA's true box. Esc closes."), TEXT
        self.screen.blit(self.small.render(words, True, colour), (16, 38))
        for index, camera in enumerate(self.cameras):
            offset = index * IMAGE_WIDTH
            if camera.frame is not None:
                self.screen.blit(pygame.surfarray.make_surface(camera.frame.swapaxes(0, 1)), (offset, HEADER))
            if self.state is not None:
                self._boxes(self.state["true_boxes"], camera, offset, TRUE_BOX, 1)
                for row in self.state["detections"]:
                    self._boxes([row], camera, offset, TRACK_PERSON if int(row[8]) == PEDESTRIAN else TRACK_CAR, 2)
        pygame.display.flip()

    def run(self):
        print("[CAMERA VIEW] Close the window to exit.", flush=True)
        while True:
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT or (event.type == self.pygame.KEYDOWN and event.key == self.pygame.K_ESCAPE):
                    return
            now = time.monotonic()
            self.poll(now)
            self.draw(now)
            self.clock.tick(FPS)

    def close(self):
        for camera in self.cameras:
            camera.destroy()
        self.pygame.quit()
        if self.feed:
            self.feed.close()


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    import carla
    import pygame

    for name in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _raise_keyboard_interrupt)
    window = Window(pygame, carla)
    try:
        window.run()
    except KeyboardInterrupt:
        pass
    finally:
        window.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
