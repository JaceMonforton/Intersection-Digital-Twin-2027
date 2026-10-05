import math
import time

import carla
import numpy as np
import pygame

# --- Configuration Constants ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
FPS = 30


def carla_img_to_pygame_surface(image):
    """Converts a CARLA raw RGBA image buffer into a Pygame Surface."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]          # Drop Alpha channel (BGRA -> BGR)
    array = array[:, :, ::-1]        # Convert BGR to RGB
    # Swap axes because Pygame surfaces are (width, height, channels)
    return pygame.surfarray.make_surface(array.swapaxes(0, 1))


class AutonomousAgent:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("CARLA Digital Twin - Autonomous Agent View")
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

    def setup_world(self):
        """Connects to CARLA and configures synchronous mode."""
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # Enable Synchronous Mode for smooth Pygame rendering
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / FPS
        self.world.apply_settings(settings)

    def spawn_ego_vehicle(self):
        """Spawns the main autonomous vehicle and sets up Traffic Manager autopilot."""
        blueprint_lib = self.world.get_blueprint_library()
        vehicle_bp = blueprint_lib.find('vehicle.tesla.model3')
        vehicle_bp.set_attribute('role_name', 'ego')

        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[0] if spawn_points else carla.Transform()

        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if not self.vehicle:
            # Fallback if first spawn point is occupied by traffic
            for sp in spawn_points[1:]:
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, sp)
                if self.vehicle:
                    break

        self.actor_list.append(self.vehicle)

        # Enable Traffic Manager Autopilot
        traffic_manager = self.client.get_trafficmanager(8000)
        self.vehicle.set_autopilot(True, traffic_manager.get_port())

        # Configure Traffic Manager safety behaviors for this vehicle
        traffic_manager.distance_to_leading_vehicle(self.vehicle, 3.0)
        traffic_manager.vehicle_percentage_speed_difference(self.vehicle, -10) # Drive 10% faster than speed limit
        traffic_manager.ignore_lights_percentage(self.vehicle, 0) # Obey red lights

    def attach_sensors(self):
        """Attaches an RGB camera behind the vehicle and a collision detector."""
        blueprint_lib = self.world.get_blueprint_library()

        # 1. Third-Person RGB Camera
        camera_bp = blueprint_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        camera_bp.set_attribute('fov', '90')

        # Mounting position: 4.5m behind, 2.0m above, tilted downward slightly
        camera_transform = carla.Transform(
            carla.Location(x=-5.0, z=2.2),
            carla.Rotation(pitch=-12.0)
        )
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda img: self.on_camera_feed(img))
        self.actor_list.append(self.camera)

        # 2. Collision Sensor
        col_bp = blueprint_lib.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.on_collision(event))
        self.actor_list.append(self.collision_sensor)

    def on_camera_feed(self, image):
        """Callback for processing RGB camera frames."""
        self.current_surface = carla_img_to_pygame_surface(image)

    def on_collision(self, event):
        """Callback for collision events."""
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.last_collision_time = time.time()
        self.collision_intensity = intensity
        print(f"[SAFETY ALERT] Collision detected with {event.other_actor.type_id}! Intensity: {intensity:.2f}")

    def render_hud(self):
        """Draws real-time telemetry and state overlays on the Pygame screen."""
        if not self.vehicle:
            return

        # Fetch telemetry
        v = self.vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        transform = self.vehicle.get_transform()
        control = self.vehicle.get_control()

        # Render Information Strings
        hud_lines = [
            "AUTOPILOT: ACTIVE (Traffic Manager)",
            f"Speed:     {speed_kmh:5.1f} km/h",
            f"Throttle:  {control.throttle * 100:3.0f}%",
            f"Brake:     {control.brake * 100:3.0f}%",
            f"Steer:     {control.steer:5.2f}",
            f"Heading:   {transform.rotation.yaw:5.1f}°",
            f"Position:  X={transform.location.x:.1f}, Y={transform.location.y:.1f}"
        ]

        # Draw dark background box for text
        hud_surface = pygame.Surface((320, len(hud_lines) * 22 + 15))
        hud_surface.set_alpha(180)
        hud_surface.fill((0, 0, 0))
        self.display.blit(hud_surface, (10, 10))

        # Render Text
        for i, line in enumerate(hud_lines):
            color = (0, 255, 120) if i == 0 else (255, 255, 255)
            text_sf = self.font.render(line, True, color)
            self.display.blit(text_sf, (20, 18 + i * 22))

        # Draw Collision Warning Overlay (flashes red for 1.5 seconds)
        if time.time() - self.last_collision_time < 1.5:
            warn_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            warn_surface.set_alpha(80)
            warn_surface.fill((255, 0, 0))
            self.display.blit(warn_surface, (0, 0))

            warn_text = self.font.render(f"!!! COLLISION DETECTED (Impulse: {self.collision_intensity:.1f}) !!!", True, (255, 255, 255))
            self.display.blit(warn_text, (WINDOW_WIDTH // 2 - 200, 30))

    def run(self):
        """Main execution loop."""
        try:
            self.setup_world()
            self.spawn_ego_vehicle()
            self.attach_sensors()

            print("\n[SUCCESS] Autonomous Agent running in Pygame.")
            print("Press ESC or close the Pygame window to exit.\n")

            running = True
            while running:
                # 1. Tick CARLA World
                self.world.tick()

                # 2. Handle Pygame Events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False

                # 3. Draw Camera Surface
                if self.current_surface is not None:
                    self.display.blit(self.current_surface, (0, 0))

                # 4. Render Overlay HUD
                self.render_hud()

                # 5. Flip Pygame Buffer
                pygame.display.flip()
                self.clock.tick(FPS)

        finally:
            self.cleanup()

    def cleanup(self):
        """Restores world settings and destroys spawned actors."""
        print("[CLEANUP] Destroying actors and resetting world settings...")
        
        # Reset Synchronous Mode
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

        # Destroy actors in reverse order
        for actor in reversed(self.actor_list):
            if actor is not None and actor.is_alive:
                actor.destroy()

        pygame.quit()
        print("[CLEANUP] Complete.")


if __name__ == '__main__':
    agent = AutonomousAgent()
    agent.run()