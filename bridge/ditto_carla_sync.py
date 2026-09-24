# import json
# import logging
# import sys
# import threading
# import time

# import carla
# import websocket

# # Set up clean logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# class DittoCarlaBridge:
#     def __init__(
#         self,
#         carla_host: str = "localhost",
#         carla_port: int = 2000,
#         ditto_ws_url: str = "ws://localhost:8080/ws/2",
#         scale_factor: float = 10.0,
#     ):
#         self.scale_factor = scale_factor
#         self.ditto_ws_url = ditto_ws_url
#         self.actor = None
#         self.running = True

#         # 1. Connect to CARLA UE4 Server
#         logging.info("Connecting to CARLA Server at %s:%d...", carla_host, carla_port)
#         self.client = carla.Client(carla_host, carla_port)
#         self.client.set_timeout(10.0)
#         self.world = self.client.get_world()

#         # 2. Spawn Digital Twin Vehicle Actor
#         blueprint_library = self.world.get_blueprint_library()
#         bp = blueprint_library.find("vehicle.tesla.model3")
#         bp.set_attribute("role_name", "digital_twin")

#         # Pick origin or default spawn point
#         spawn_points = self.world.get_map().get_spawn_points()
#         spawn_point = spawn_points[0] if spawn_points else carla.Transform()

#         self.actor = self.world.spawn_actor(bp, spawn_point)
#         logging.info("Spawned CARLA Digital Twin Actor (ID: %d)", self.actor.id)

#     def update_actor_transform(self, x_phys: float, y_phys: float, heading_deg: float):
#         """Converts physical coordinates to CARLA spatial transform & moves actor."""
#         # Coordinate Transformation: Physical (Right-Handed ENU) -> CARLA (Left-Handed)
#         carla_x = x_phys * self.scale_factor
#         carla_y = -y_phys * self.scale_factor
#         carla_yaw = -heading_deg

#         location = carla.Location(x=carla_x, y=carla_y, z=0.5)
#         rotation = carla.Rotation(pitch=0.0, yaw=carla_yaw, roll=0.0)
#         transform = carla.Transform(location, rotation)

#         if self.actor:
#             # 1. Update vehicle position in 3D world
#             self.actor.set_transform(transform)

#             # 2. Auto-track spectator camera behind vehicle
#             spectator = self.world.get_spectator()
#             camera_location = location + carla.Location(x=-6.0, y=0.0, z=3.0)
#             spectator.set_transform(
#                 carla.Transform(
#                     camera_location,
#                     carla.Rotation(pitch=-15.0, yaw=carla_yaw, roll=0.0),
#                 )
#             )

#     def on_ditto_message(self, ws, message):
#         """Processes incoming WebSocket event stream from Eclipse Ditto."""
#         try:
#             payload = json.loads(message)
#             # Filter for kinematics state modifications
#             if payload.get("path") == "/features/kinematics/properties":
#                 kinematics = payload.get("value", {})
#                 x = float(kinematics.get("x", 0.0))
#                 y = float(kinematics.get("y", 0.0))
#                 heading = float(kinematics.get("heading", 0.0))

#                 logging.info("Twin Update -> x: %.2f, y: %.2f, heading: %.1f°", x, y, heading)
#                 self.update_actor_transform(x, y, heading)
#         except Exception as err:
#             logging.error("Failed to parse Ditto message: %s", err)

#     def start(self):
#         """Connects WebSocket listener thread."""
#         logging.info("Opening WebSocket connection to Ditto at %s...", self.ditto_ws_url)
#         self.ws = websocket.WebSocketApp(
#             self.ditto_ws_url,
#             on_message=self.on_ditto_message,
#             header=["Authorization: Basic ZGl0dG86ZGl0dG8="],  # Default ditto:ditto basic auth
#         )
#         ws_thread = threading.Thread(target=self.ws.run_forever)
#         ws_thread.daemon = True
#         ws_thread.start()
#         logging.info("Bridge active! Waiting for live telemetry updates...")

#     def cleanup(self):
#         """Cleanly destroys spawned actor on shutdown."""
#         logging.info("Shutting down bridge...")
#         if hasattr(self, "ws"):
#             self.ws.close()
#         if self.actor:
#             self.actor.destroy()
#             logging.info("CARLA Digital Twin actor destroyed cleanly.")


# if __name__ == "__main__":
#     bridge = DittoCarlaBridge()
#     bridge.start()

#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         bridge.cleanup()
#         sys.exit(0)