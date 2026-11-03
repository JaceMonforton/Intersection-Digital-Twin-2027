"""
Onboard safety gate for the physical vehicle (Raspberry Pi / Jetson).

Listens to device/vehicle_ml/commands and cuts the motors when the SafetyGate feature reports
emergency_brake == True. It fails safe: the motors are also held off before the first state
arrives and whenever the broker link is lost, because a vehicle that cannot hear the gate
cannot be braked by it. Only an explicit emergency_brake == False releases the hold.

Needs only paho-mqtt, plus Rosmaster_Lib or RPi.GPIO for the motors.

Usage:
    python scripts/physical_ml_subscriber.py --broker 192.168.0.100
    python scripts/physical_ml_subscriber.py --driver sim   # no motors, for bench testing
"""
import argparse
import json
import threading
import time

import paho.mqtt.client as mqtt

# --- Configuration ---
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_KEEPALIVE = 2      # seconds; sets how quickly a dead Wi-Fi link is noticed
COMMAND_TOPIC = "device/vehicle_ml/commands"
GATE_PREFIX = ("features", "SafetyGate", "properties")

REASSERT_HZ = 20        # re-send the stop while held, in case another process commands the motors
PWM_FREQUENCY = 1000
DEFAULT_PWM_PINS = "18,13"


# ---------------------------------------------------------------------------
# Motor backends
# ---------------------------------------------------------------------------
class RosmasterMotors:
    """Yahboom Rosmaster R2, through its expansion board."""
    name = "rosmaster"

    def __init__(self):
        from Rosmaster_Lib import Rosmaster

        self.bot = Rosmaster()

    def stop(self):
        self.bot.set_motor(0, 0, 0, 0)
        self.bot.set_car_motion(0, 0, 0)


class GpioPwmMotors:
    """Motors on PWM pins owned by this process. Jetson.GPIO provides the same module name."""
    name = "gpio"

    def __init__(self, pins):
        from RPi import GPIO

        GPIO.setmode(GPIO.BCM)
        self.channels = []
        for pin in pins:
            GPIO.setup(pin, GPIO.OUT)
            pwm = GPIO.PWM(pin, PWM_FREQUENCY)
            pwm.start(0)
            self.channels.append(pwm)

    def stop(self):
        for pwm in self.channels:
            pwm.ChangeDutyCycle(0)


class SimMotors:
    """Records stops instead of driving hardware."""
    name = "sim"

    def __init__(self):
        self.stops = 0

    def stop(self):
        self.stops += 1


def make_motors(driver, pins):
    if driver == "rosmaster":
        return RosmasterMotors()
    if driver == "gpio":
        return GpioPwmMotors(pins)
    if driver == "sim":
        return SimMotors()
    for factory in (RosmasterMotors, lambda: GpioPwmMotors(pins)):
        try:
            return factory()
        except (ImportError, RuntimeError, OSError) as e:
            print(f"[MOTOR] Backend unavailable: {e}")
    print("[WARN] NO MOTOR HARDWARE FOUND. Using the simulated driver: a brake will stop NOTHING.")
    return SimMotors()


# ---------------------------------------------------------------------------
# Safety gate
# ---------------------------------------------------------------------------
def parse_gate_event(payload):
    """emergency_brake from a Ditto Protocol event, or None if it doesn't carry one.
    Ditto reports the change at whichever depth was written, so rebuild the thing from
    path + value and read the field back out."""
    try:
        env = json.loads(payload)
        path = [p for p in env["path"].split("/") if p]
        value = env["value"]
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    node = {}
    cursor = node
    for part in path[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    if path:
        cursor[path[-1]] = value
    else:
        node = value
    for key in (*GATE_PREFIX, "emergency_brake"):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node if isinstance(node, bool) else None


class SafetyGateSubscriber:
    def __init__(self, motors, broker=MQTT_BROKER, port=MQTT_PORT, topic=COMMAND_TOPIC):
        self.motors = motors
        self.broker = broker
        self.port = port
        self.topic = topic
        self.lock = threading.Lock()
        self.held = False
        self.reason = None
        self.running = False
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"safety-gate-{id(self):x}")
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=2)
        self._engage("waiting for first SafetyGate state")

    def _engage(self, reason):
        with self.lock:
            changed = not self.held
            self.held = True
            self.reason = reason
        self.motors.stop()  # before logging, which can block
        if changed:
            print(f"[SAFETY] MOTORS CUT: {reason}")

    def _release(self):
        with self.lock:
            changed = self.held
            self.held = False
            self.reason = None
        if changed:
            print("[SAFETY] Hold released.")

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            print(f"[MQTT] Connection refused: {reason_code}")
            return
        print(f"[MQTT] Connected to {self.broker}:{self.port}, subscribing to {self.topic}")
        client.subscribe(self.topic, qos=1)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        if self.running:
            self._engage(f"broker link lost ({reason_code})")

    def _on_message(self, client, userdata, msg):
        self.handle_payload(msg.payload)

    def handle_payload(self, payload):
        brake = parse_gate_event(payload)
        if brake is True:
            self._engage("emergency_brake == True")
        elif brake is False:
            self._release()

    def _reassert_loop(self):
        while self.running:
            with self.lock:
                held = self.held
            if held:
                try:
                    self.motors.stop()
                except Exception as e:  # noqa: BLE001  # this thread must outlive a driver hiccup
                    print(f"[MOTOR] Stop failed: {e}")
            time.sleep(1.0 / REASSERT_HZ)

    def start(self):
        self.running = True
        threading.Thread(target=self._reassert_loop, daemon=True).start()
        self.client.connect_async(self.broker, self.port, keepalive=MQTT_KEEPALIVE)
        self.client.loop_start()

    def stop(self):
        self.running = False
        self.client.disconnect()
        self.client.loop_stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker", default=MQTT_BROKER)
    parser.add_argument("--port", type=int, default=MQTT_PORT)
    parser.add_argument("--driver", choices=("auto", "rosmaster", "gpio", "sim"), default="auto")
    parser.add_argument("--pwm-pins", default=DEFAULT_PWM_PINS, help="BCM pins for --driver gpio")
    args = parser.parse_args()

    motors = make_motors(args.driver, [int(p) for p in args.pwm_pins.split(",")])
    print(f"[MOTOR] Driver: {motors.name}")
    gate = SafetyGateSubscriber(motors, args.broker, args.port)
    gate.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        gate.stop()
        motors.stop()


if __name__ == "__main__":
    main()
