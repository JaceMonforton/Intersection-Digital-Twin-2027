"""
How hard the ML car really brakes in CARLA, against what the planner assumes (COMFORT_BRAKE and HARD_BRAKE in
scripts/planner.py). The ML car's Tesla Model 3 is brought to each speed on a straight road and then stopped three ways:

    full      brake 1.0 straight away: the hardest the car can stop, the safety gate's and the monitor's hold
    stop      the speed controller given a target of 0: what the planner's "hard_stop" really gets
    comfort   the speed controller given a target falling at COMFORT_BRAKE: what the planner's "stop" really gets

    python training/measure_braking.py              # starts its own processor; needs CARLA and nothing else using it

It starts and stops central_gpu_processor.py itself, as only the processor may tick the world.
"""
import argparse
import json
import math
import queue
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from central_gpu_processor import CARLA_HOST, CARLA_PORT
from driving_features import TICK, SpeedController
from planner import COMFORT_BRAKE, HARD_BRAKE

SPEEDS = (5.0, 10.0, 14.0)   # m/s; 14 is the fastest the scenarios' cars go
STRAIGHT = 120.0             # m of straight road looked for ahead of the start; the longest there is gets used
STOPPED = 0.3                # m/s
SETTLE_TICKS = 10            # ticks after spawning before the car drives off
UP_TO_SPEED_TICKS = 400      # most ticks to reach and hold a speed
STEADY_TICKS = 10            # ticks within 0.3 m/s of it to count as held
OUT = ROOT / "docs" / "results" / "braking.json"


def braking(speeds, tick=TICK):
    """From one tick's speeds while stopping: (mean deceleration until stopped, hardest one-tick deceleration,
    s until stopped, m travelled until stopped). Mean is the start speed over the time taken, the planner's sense."""
    speeds = np.asarray(speeds, dtype=float)
    under = np.nonzero(speeds < STOPPED)[0]
    end = int(under[0]) if len(under) else len(speeds) - 1
    seconds = end * tick
    metres = float(np.sum((speeds[:end] + speeds[1:end + 1]) / 2.0) * tick)
    return (speeds[0] - speeds[end]) / seconds if seconds else 0.0, float(np.max(-np.diff(speeds[:end + 1]))) / tick if end else 0.0, seconds, metres


def straight_ahead(world_map, point, most=STRAIGHT):
    """m of road ahead of a spawn point, up to most, before it bends 2 degrees. Junctions count: the road is empty."""
    waypoint = world_map.get_waypoint(point.location)
    yaw = waypoint.transform.rotation.yaw
    for step in range(int(most / 2.0)):
        ahead = waypoint.next(2.0)
        if not ahead or abs((ahead[0].transform.rotation.yaw - yaw + 180) % 360 - 180) > 2.0:
            return step * 2.0
        waypoint = ahead[0]
    return most


def run_once(carla, world, ticks, point, speed, how):
    car = world.spawn_actor(world.get_blueprint_library().find("vehicle.tesla.model3"), point)
    try:
        for _ in range(SETTLE_TICKS):
            car.apply_control(carla.VehicleControl(brake=1.0))
            ticks.get(timeout=10)
        # Driven up to speed on its own wheels: set_target_velocity leaves them still, and they skid on the first tick
        controller, steady = SpeedController(), 0
        for _ in range(UP_TO_SPEED_TICKS):
            v = car.get_velocity()
            now = math.hypot(v.x, v.y)
            steady = steady + 1 if abs(now - speed) < 0.3 else 0
            if steady >= STEADY_TICKS:
                break
            _, throttle, brake = controller.step((0.0, speed), now)
            car.apply_control(carla.VehicleControl(throttle=float(throttle), brake=float(brake)))
            ticks.get(timeout=10)
        else:
            raise RuntimeError(f"the car didn't hold {speed} m/s within {UP_TO_SPEED_TICKS} ticks")
        speed = now
        controller, speeds = SpeedController(), []
        for n in range(200):
            v = car.get_velocity()
            now = math.hypot(v.x, v.y)
            speeds.append(now)
            if now < STOPPED and n:
                break
            if how == "full":
                control = carla.VehicleControl(brake=1.0)
            else:
                target = 0.0 if how == "stop" else max(0.0, speed - COMFORT_BRAKE * (n + 1) * TICK)
                _, throttle, brake = controller.step((0.0, target), now)
                control = carla.VehicleControl(throttle=float(throttle), brake=float(brake))
            car.apply_control(control)
            ticks.get(timeout=10)
        return speeds
    finally:
        car.destroy()
        ticks.get(timeout=10)


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    import carla
    from hidden_hazards import start_processor, stop_processor

    process, log = start_processor("truth", ROOT / "models" / "rsu_detector.candidate.onnx", ROOT / "output" / "braking_processor.log")
    try:
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(20.0)
        world = client.get_world()
        ticks = queue.Queue()
        listening = world.on_tick(ticks.put)
        world_map = world.get_map()
        point = max(world_map.get_spawn_points(), key=lambda p: straight_ahead(world_map, p))
        print(f"Straight road ahead of the start: {straight_ahead(world_map, point):.0f} m", flush=True)
        results = []
        for speed in SPEEDS:
            for how in ("full", "stop", "comfort"):
                speeds = run_once(carla, world, ticks, point, speed, how)
                mean, hardest, seconds, metres = braking(speeds)
                results.append({"from": round(speeds[0], 2), "how": how, "mean_decel": round(mean, 2), "hardest_tick": round(hardest, 2),
                                "seconds": round(seconds, 1), "metres": round(metres, 1), "speeds": [round(v, 2) for v in speeds]})
                print(f"from {speeds[0]:4.1f} m/s, {how:7s}: {mean:4.1f} m/s2 on average, {hardest:4.1f} at most in one tick, "
                      f"stopped in {seconds:.1f} s and {metres:.1f} m", flush=True)
        world.remove_on_tick(listening)
        OUT.write_text(json.dumps({"assumed": {"comfort": COMFORT_BRAKE, "hard": HARD_BRAKE}, "measured": results}, indent=2) + "\n")
        print(f"Saved {OUT}")
    finally:
        stop_processor(process, log)


if __name__ == "__main__":
    main()
