"""
Drives a real (or fake) lab car through its twin: the trained driving model, the planner and the car's safety
monitor decide as they do for the ML car in CARLA, but the decision goes to the car as a drive command instead of
to CARLA's pedals. The car's own speed controller works its motors, behind its safety gate.

    python scripts/remote_driver.py                 # drive vehicle_ml
    python scripts/remote_driver.py --dry-run       # print the commands, send nothing: the first step with a real car

Where the car is comes from its own telemetry (telemetry/<car>, the pose that scripts/fake_lab_car.py and the overhead
camera publish), mapped into CARLA through config/lab_frame.json, the same mapping the processor and the twin mirror
use. Everything else (other cars, people, the lights) comes from CARLA's true positions: this driver doesn't run the
detectors. It attaches to the simulation and never ticks it, deciding once per tick.

It holds the car whenever it has any doubt: no pose yet, a pose that stopped arriving or stopped changing
(POSE_MAX_AGE, inside the gate's own 0.3 s watchdog), the gate or the safety monitor holding, a decision that failed
or wasn't finite. It never releases a brake: only an operator does. A hold is sent on the way out, but the car's
watchdog is what stops it when this process dies.
"""
import argparse
import json
import math
import random
import sys
import threading
import time
import uuid

import lab_frame

POSE_MAX_AGE = 0.25  # s since a new pose last arrived; inside the gate's 0.3 s, at the fake car's 20 Hz


class PoseTracker:
    """The car's latest pose from its telemetry, in CARLA's frame. A pose counts only while new ones keep arriving:
    one whose t hasn't moved on doesn't refresh it, so a stuck publisher repeating itself goes stale too."""

    def __init__(self, frame, clock=time.monotonic):
        self.frame, self.clock = frame, clock
        self.lock = threading.Lock()
        self.pose = self.received = self.last_t = None

    def feed(self, payload):
        try:
            message = json.loads(payload)
        except (ValueError, TypeError):
            return
        if not isinstance(message, dict) or message.get("path") != "/features/pose/properties" or not isinstance(message.get("value"), dict):
            return
        pose = lab_frame.carla_pose({"features": {"pose": {"properties": message["value"]}}}, self.frame)
        if pose is None or pose["t"] is None:
            return
        with self.lock:
            if self.last_t is not None and pose["t"] <= self.last_t:
                return  # not new
            self.pose, self.received, self.last_t = pose, self.clock(), pose["t"]

    def ego(self):
        """(x, y, vx, vy, yaw) in CARLA's frame, or None with no fresh pose."""
        with self.lock:
            if self.pose is None or self.clock() - self.received > POSE_MAX_AGE:
                return None
            p = self.pose
        return p["x"], p["y"], p["vx"], p["vy"], p["yaw"]


class CommandStream:
    """Drive and hold commands for the car's gate, numbered within this run's own session."""

    def __init__(self):
        self.session = uuid.uuid4().hex
        self.seq = 0

    def _next(self, **fields):
        self.seq += 1
        return json.dumps({"session": self.session, "seq": self.seq, **fields}).encode()

    def hold(self):
        return self._next(hold=True)

    def drive(self, steer, speed):
        return self._next(steer=steer, speed=speed)


def next_command(stream, tracker, decide, frame):
    """This tick's command. decide(ego) is ModelDriver.decide for this tick's world; it isn't asked without a pose."""
    ego = tracker.ego()
    if ego is None:
        return stream.hold()
    try:
        _, _, drive, allowed = decide(ego)
        steer, speed = float(drive[0]), float(drive[1])
    except Exception as e:  # noqa: BLE001  # whatever went wrong, the car holds rather than keeping its last command
        print(f"[REMOTE] Decision failed, holding: {e}", flush=True)
        return stream.hold()
    if not allowed or not (math.isfinite(steer) and math.isfinite(speed)):
        return stream.hold()
    # Steering keeps its sign: the lab-to-CARLA flip leaves a right turn a right turn
    return stream.drive(steer, max(0.0, speed / frame.scale))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--car", default="vehicle_ml", help="the Thing name: its telemetry and command topics")
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--seconds", type=float, default=0.0, help="how long to drive; 0 until Ctrl+C")
    parser.add_argument("--dry-run", action="store_true", help="print each command instead of sending it")
    args = parser.parse_args()

    import carla
    import numpy as np
    import paho.mqtt.client as mqtt
    import requests
    from driving_features import INPUT_SIZE
    from ml_driver import DrivingPolicy, ModelDriver, actor_state, brake_is_off
    from planner import Planner
    from safety_monitor import SafetyMonitor
    from scenario_suite import JUNCTION
    from traffic_monitor import GATE_URL, LaneDriver, RedLightWatch

    frame = lab_frame.load()
    if not frame.calibrated:
        print("[REMOTE] config/lab_frame.json isn't calibrated yet: the twin is placed by a guess.", flush=True)
    tracker, stream = PoseTracker(frame), CommandStream()
    drive_topic = f"device/{args.car}/drive"
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"remote-driver-{stream.session[:8]}")
    client.on_connect = lambda c, u, f, rc, p: c.subscribe(f"telemetry/{args.car}", qos=0)
    client.on_message = lambda c, u, msg: tracker.feed(msg.payload)
    client.connect(args.broker, args.port, keepalive=2)
    client.loop_start()

    def send(command):
        if args.dry_run:
            print(f"[REMOTE] would send {command.decode()}", flush=True)
        else:
            client.publish(drive_topic, command, qos=0, retain=False)

    world = carla.Client("localhost", 2000)
    world.set_timeout(30.0)
    world = world.get_world()
    world.wait_for_tick(10.0)
    world_map, lights, session = world.get_map(), RedLightWatch(carla, world), requests.Session()
    policy, rng = DrivingPolicy(), random.Random(0)
    policy(np.zeros(INPUT_SIZE, dtype=np.float32))  # the first run is slow, and would outlast the gate's watchdog

    def find_waypoint(x, y):
        return world_map.get_waypoint(carla.Location(x=float(x), y=float(y), z=0.0), project_to_road=True, lane_type=carla.LaneType.Driving)

    driver = None
    started = time.monotonic()
    print(f"[REMOTE] Driving {args.car} through its twin{' (dry run)' if args.dry_run else ''}. Session {stream.session}.", flush=True)
    try:
        while args.seconds <= 0 or time.monotonic() - started < args.seconds:
            world.wait_for_tick(10.0)  # a stalled simulation stops the commands, and the car's watchdog then holds it
            ego = tracker.ego()
            if ego is not None and driver is None:
                driver = ModelDriver(policy, LaneDriver(find_waypoint(ego[0], ego[1]), rng), find_waypoint, rng)
                driver.planner, driver.monitor = Planner(JUNCTION, has_roadside=False), SafetyMonitor()
            if driver is None:
                # Nothing is sent until it can drive: the first command arms the gate's watchdog, and setting up
                # took longer than its 0.3 s. Until then the gate isn't armed and nothing drives the car.
                continue
            cars = [actor_state(a) for a in world.get_actors().filter("vehicle.*") if a.attributes.get("role_name") != args.car]
            walkers = [(w.get_location().x, w.get_location().y, w.get_velocity().x, w.get_velocity().y)
                       for w in world.get_actors().filter("walker.pedestrian.*")]
            try:
                released = brake_is_off(session.get(GATE_URL, timeout=0.2).json())
            except (requests.RequestException, ValueError):
                released = False  # can't read the gate, so don't drive

            def decide(ego, driver=driver, cars=cars, walkers=walkers, released=released):
                return driver.decide(ego, lights.not_green(), cars, walkers, released)

            send(next_command(stream, tracker, decide, frame))
    except KeyboardInterrupt:
        pass
    finally:
        send(stream.hold())  # a courtesy: the car's watchdog is what stops it if this never arrives
        time.sleep(0.2)
        client.loop_stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
