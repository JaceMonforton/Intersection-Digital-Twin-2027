"""
Onboard safety gate for the physical vehicle (Raspberry Pi / Jetson).

Listens to device/vehicle_ml/commands and cuts the motors when the SafetyGate feature reports
emergency_brake == True. It fails safe: the motors are also held off before the first state
arrives and whenever the broker link is lost, because a vehicle that cannot hear the gate
cannot be braked by it. Only an explicit emergency_brake == False releases the hold.

It also holds the motors when the driver goes quiet. Once the driver has sent its first valid command on
device/vehicle_ml/drive, the gate expects another within WATCHDOG_TIMEOUT by its own clock, whether to drive or
to hold. Silence means a dead link, a dead processor or a stalled simulation, and the broker's keepalive alone
took 3.8 s to notice a silent Wi-Fi drop. A trip latches like a broker loss, and a release disarms the watchdog
until the next valid command, so it can't trip again on the silence it was released from.

    held because        set by                       cleared by
    no state yet        start                        emergency_brake == False
    broker link lost    a disconnect                 emergency_brake == False
    emergency_brake     emergency_brake == True      emergency_brake == False
    watchdog            WATCHDOG_TIMEOUT of silence  emergency_brake == False

A drive command counts only if it is valid JSON with a session id, a sequence number above the last one from that
session (a new session may start again from 0, after a driver restart) and either hold or finite steer and speed.
Retained commands, and any before the first gate state, are ignored. Timestamps from the driver's clock are never
compared with this one. Nothing here drives the motors yet: the command is kept for the drive path to come.

Needs only paho-mqtt, plus Rosmaster_Lib or RPi.GPIO for the motors.

Usage:
    python scripts/physical_ml_subscriber.py --broker 192.168.0.100
    python scripts/physical_ml_subscriber.py --driver sim   # no motors, for bench testing
"""
import argparse
import json
import math
import threading
import time

import paho.mqtt.client as mqtt

# --- Configuration ---
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_KEEPALIVE = 2      # seconds; sets how quickly a dead Wi-Fi link is noticed
COMMAND_TOPIC = "device/vehicle_ml/commands"
DRIVE_TOPIC = "device/vehicle_ml/drive"
GATE_PREFIX = ("features", "SafetyGate", "properties")
# shortcut: 0.3 s is three missed commands at 10 Hz, chosen before any real Wi-Fi was measured; tune it on the car
WATCHDOG_TIMEOUT = 0.3  # seconds of driver silence, by this machine's clock, that hold the motors

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


def parse_drive_command(payload):
    """A drive or hold command as a dict, or None if it isn't a valid one. See the module docstring."""
    try:
        command = json.loads(payload)
    except (ValueError, TypeError):
        return None
    if not isinstance(command, dict) or not isinstance(command.get("session"), str):
        return None
    seq = command.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        return None
    if command.get("hold") is True:
        return {"session": command["session"], "seq": seq, "hold": True}
    values = [command.get("steer"), command.get("speed")]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
        return None
    return {"session": command["session"], "seq": seq, "hold": False, "steer": float(values[0]), "speed": float(values[1])}


class SafetyGateSubscriber:
    def __init__(self, motors, broker=MQTT_BROKER, port=MQTT_PORT, topic=COMMAND_TOPIC, drive_topic=DRIVE_TOPIC,
                 watchdog_timeout=WATCHDOG_TIMEOUT, clock=time.monotonic):
        self.motors = motors
        self.broker = broker
        self.port = port
        self.topic = topic
        self.drive_topic = drive_topic
        self.watchdog_timeout = watchdog_timeout
        self.clock = clock
        self.lock = threading.Lock()
        self.held = False
        self.reason = None
        self.running = False
        self.have_state = False   # a SafetyGate state has arrived; drive commands before it are ignored
        self.command = None       # the newest valid drive or hold command
        self.last_command_at = None  # clock time it arrived; None while the watchdog is disarmed
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
            self.last_command_at = None  # disarmed until the next valid command, so the old silence can't trip it
        if changed:
            print("[SAFETY] Hold released.")

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            print(f"[MQTT] Connection refused: {reason_code}")
            return
        print(f"[MQTT] Connected to {self.broker}:{self.port}, subscribing to {self.topic} and {self.drive_topic}")
        client.subscribe(self.topic, qos=1)
        client.subscribe(self.drive_topic, qos=0)  # a late drive command is worse than none, so no redelivery

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        if self.running:
            self._engage(f"broker link lost ({reason_code})")

    def _on_message(self, client, userdata, msg):
        if msg.topic == self.drive_topic:
            self.handle_drive(msg.payload, retained=bool(msg.retain))
        else:
            self.handle_payload(msg.payload)

    def handle_payload(self, payload):
        brake = parse_gate_event(payload)
        if brake is not None:
            self.have_state = True
        if brake is True:
            self._engage("emergency_brake == True")
        elif brake is False:
            self._release()

    def handle_drive(self, payload, retained=False):
        """Takes a drive or hold command and feeds the watchdog with it. Returns whether it was accepted. It never
        releases a hold: only emergency_brake == False does."""
        command = None if retained else parse_drive_command(payload)
        with self.lock:
            if command is None or not self.have_state:
                return False
            last = self.command
            if last is not None and last["session"] == command["session"] and command["seq"] <= last["seq"]:
                return False  # a replay, or one overtaken by a newer command
            self.command = command
            self.last_command_at = self.clock()
        return True

    def check_watchdog(self):
        """Holds the motors once an armed watchdog has heard nothing valid for longer than its timeout."""
        with self.lock:
            stale = self.last_command_at is not None and self.clock() - self.last_command_at > self.watchdog_timeout
            if stale:
                self.last_command_at = None  # one trip per silence; the hold itself is latched
        if stale:
            self._engage(f"watchdog: no drive command for over {self.watchdog_timeout:.2f} s")

    def _reassert_loop(self):
        while self.running:
            self.check_watchdog()
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
    parser.add_argument("--watchdog", type=float, default=WATCHDOG_TIMEOUT,
                        help="seconds of drive-command silence that hold the motors, once commands have started")
    args = parser.parse_args()
    if not 0.05 <= args.watchdog <= 5.0:
        parser.error("--watchdog must be between 0.05 and 5 seconds")

    motors = make_motors(args.driver, [int(p) for p in args.pwm_pins.split(",")])
    print(f"[MOTOR] Driver: {motors.name}")
    gate = SafetyGateSubscriber(motors, args.broker, args.port, watchdog_timeout=args.watchdog)
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
