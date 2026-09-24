import json
import logging
import sys
import threading
import time

import carla
import websocket

# Set up clean terminal logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- System Configuration ---
CARLA_HOST = "localhost"
CARLA_PORT = 2000
DITTO_WS_URL = "ws://localhost:8080/ws/2"
SCALE_FACTOR = 10.0


class DittoCarlaBridge:
    def __init__(self):
        self.actor = None
        self.world = None

        # 1. Connect to CARLA
        logging.info("Connecting to CARLA server at %s:%d...", CARLA_HOST, CARLA_PORT)
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # 2. Spawn Digital Twin Vehicle Actor
        bp_library = self.world.get_blueprint_library()
        vehicle_bp = bp_library.find("vehicle.tesla.model3")
        spawn_point = self.world.get_map().get_spawn_points()[0]

        self.actor = self.world.spawn_actor(vehicle_bp, spawn_point)
        logging.info("Spawned Digital Twin Vehicle (Actor ID: %d)", self.actor.id)

    def on_open(self, ws):
        logging.info("WebSocket connected to Ditto! Sending event subscription command...")
        # MANDATORY: Ditto protocol requires plain-text 'START-SEND-EVENTS' to begin streaming
        ws.send("START-SEND-EVENTS")

    def on_error(self, ws, error):
        logging.error("WebSocket Error: %s", error)

    def on_close(self, ws, close_status, close_msg):
        logging.warning("WebSocket Closed [Code %s]: %s", close_status, close_msg)

    def on_message(self, ws, message):
        # Handle Ditto subscription acknowledgment
        if message.startswith("START-SEND-EVENTS:ACK"):
            logging.info("✅ Ditto Event Subscription Confirmed! Waiting for twin telemetry...")
            return

        try:
            data = json.loads(message)
            path = data.get("path", "")
            value = data.get("value", {})

            # Match kinematics update path
            if "/features/kinematics" in path and isinstance(value, dict):
                x_phys = float(value.get("x", 0.0))
                y_phys = float(value.get("y", 0.0))
                heading_deg = float(value.get("heading", 0.0))

                # Frame Transform: Physical Right-Handed ENU -> CARLA Left-Handed
                carla_x = x_phys * SCALE_FACTOR
                carla_y = -y_phys * SCALE_FACTOR
                carla_yaw = -heading_deg

                location = carla.Location(x=carla_x, y=carla_y, z=0.5)
                rotation = carla.Rotation(pitch=0.0, yaw=carla_yaw, roll=0.0)
                transform = carla.Transform(location, rotation)

                if self.actor:
                    self.actor.set_transform(transform)
                    logging.info(
                        "Twin Move -> Phys(x=%.2f, y=%.2f) | CARLA(x=%.2f, y=%.2f, yaw=%.1f°)",
                        x_phys, y_phys, carla_x, carla_y, carla_yaw
                    )

        except json.JSONDecodeError:
            logging.debug("Non-JSON message received: %s", message)
        except Exception as err:
            logging.error("Error processing update: %s", err)

    def start(self):
        logging.info("Connecting to Ditto WebSocket at %s...", DITTO_WS_URL)
        self.ws = websocket.WebSocketApp(
            DITTO_WS_URL,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            header=["Authorization: Basic ZGl0dG86ZGl0dG8="],  # Default ditto:ditto basic auth
        )
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

    def cleanup(self):
        if hasattr(self, "ws"):
            self.ws.close()
        if self.actor is not None:
            self.actor.destroy()
            logging.info("Cleaned up CARLA vehicle actor.")


if __name__ == "__main__":
    bridge = DittoCarlaBridge()
    bridge.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        bridge.cleanup()
        sys.exit(0)