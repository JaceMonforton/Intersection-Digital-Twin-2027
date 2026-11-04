"""
The driving model's pass test: seeded drives in traffic, measured against the autopilot cars around it.

It passes if its collisions and red lights per km are at most 1.25 times autopilot's, and it covers at
least 90% of the distance autopilot does in the same time. Autopilot is measured at the same time and
the same way, with its contacts with the ML car left out, since those are the model's.

Run it once with the brake ignored, which measures the model alone, and once with it active. The brake
latches, so after a short hold this releases it, standing in for an operator, as traffic_monitor.py does.

Needs the processor and test_traffic.py running, and the main Python environment, which has carla.
    python training/pass_test.py --ignore-brake
    python training/pass_test.py
"""
import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from ml_driver import (
    CARLA_HOST,
    CARLA_PORT,
    DrivingPolicy,
    actor_state,
    drive,
    is_new_contact,
)
from traffic_monitor import BRAKE_HOLD, OUTPUT_DIR, RedLightWatch

# --- Configuration ---
RATE_LIMIT = 1.25          # times autopilot's collisions and red lights per km
MIN_DISTANCE_SHARE = 0.9   # of the distance autopilot covers in the same time
AUTOPILOT = "autopilot"
ML_ROLE = "vehicle_ml"
TELEPORT = 5.0             # m in a tick that isn't driving


# ---------------------------------------------------------------------------
# The verdict (no CARLA needed)
# ---------------------------------------------------------------------------
def verdict(ml, autopilot):
    """Each is {km, collisions, red_runs, metres_per_car_second}. Returns the rates and the result of each check."""
    def per_km(side, name):
        return side[name] / max(side["km"], 1e-6)

    result = {"checks": {}}
    for name in ("collisions", "red_runs"):
        limit = RATE_LIMIT * per_km(autopilot, name)
        result["checks"][name] = {"ml_per_km": per_km(ml, name), "autopilot_per_km": per_km(autopilot, name),
                                  "limit_per_km": limit, "passed": per_km(ml, name) <= limit}
    share = ml["metres_per_car_second"] / max(autopilot["metres_per_car_second"], 1e-6)
    result["checks"]["distance"] = {"share_of_autopilot": share, "limit": MIN_DISTANCE_SHARE, "passed": share >= MIN_DISTANCE_SHARE}
    result["passed"] = all(check["passed"] for check in result["checks"].values())
    return result


def summarise(reports):
    """The ML car's totals over its drives, in the form verdict() takes."""
    metres = sum(r["distance"] for r in reports)
    seconds = sum(r["planned_seconds"] for r in reports)
    kinds = {}
    for report in reports:
        for hit in report["collisions"]:
            kind = hit["with"].split(".")[0]
            kinds[kind] = kinds.get(kind, 0) + 1
    return {"km": metres / 1000.0, "collisions": sum(len(r["collisions"]) for r in reports), "collisions_by_kind": kinds,
            "red_runs": sum(len(r["red_light_runs"]) for r in reports), "metres_per_car_second": metres / max(seconds, 1e-6),
            "times_lost": sum(r["times_lost"] for r in reports), "got_stuck": sum(r["got_stuck"] for r in reports),
            "brake_requests": sum(r["brake_requests"] for r in reports), "missed_ticks": sum(r["missed_ticks"] for r in reports)}


def counts_as_autopilot_collision(other_role):
    """A contact with the ML car is the model's, whichever car's sensor felt it."""
    return other_role != ML_ROLE


# ---------------------------------------------------------------------------
# Measuring autopilot (needs carla)
# ---------------------------------------------------------------------------
def watch_autopilot(seconds, out):
    import carla

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(30.0)
    world = client.get_world()
    world.wait_for_tick(10.0)
    lights = RedLightWatch(carla, world)
    bp = world.get_blueprint_library().find("sensor.other.collision")
    sensors, hits, last, contacts = {}, [], {}, {}
    metres, car_seconds, collisions, kinds = 0.0, 0.0, 0, {}
    started = now = None
    try:
        while started is None or now - started < seconds:
            snapshot = world.wait_for_tick(10.0)
            now = snapshot.timestamp.elapsed_seconds
            started = now if started is None else started
            states = {}
            for car in world.get_actors().filter("vehicle.*"):
                if car.attributes.get("role_name") != AUTOPILOT:
                    continue
                states[car.id] = actor_state(car)
                if car.id not in sensors:
                    sensor = world.spawn_actor(bp, carla.Transform(), attach_to=car)
                    sensor.listen(lambda e, me=car.id: hits.append((me, e.other_actor.id, e.other_actor.type_id,
                                                                    e.other_actor.attributes.get("role_name"))))
                    sensors[car.id] = sensor
                if car.id in last:
                    step = math.hypot(states[car.id][0] - last[car.id][0], states[car.id][1] - last[car.id][1])
                    metres += step if step < TELEPORT else 0.0
                last[car.id] = states[car.id][:2]
            car_seconds += len(states) * snapshot.timestamp.delta_seconds
            lights.update(states, now - started, None, "AUTOPILOT")
            while hits:
                me, other, kind, role = hits.pop()
                if counts_as_autopilot_collision(role) and is_new_contact(contacts, (min(me, other), max(me, other)), now):
                    collisions += 1
                    name = kind.split(".")[0]
                    kinds[name] = kinds.get(name, 0) + 1
    finally:
        for sensor in sensors.values():
            sensor.stop()
            sensor.destroy()
    Path(out).write_text(json.dumps({"km": metres / 1000.0, "collisions": collisions, "collisions_by_kind": kinds,
                                     "red_runs": len(lights.runs), "metres_per_car_second": metres / max(car_seconds, 1e-6),
                                     "cars": len(states), "seconds": round(now - started)}))


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------
def print_verdict(result, ml, autopilot, mode):
    checks = result["checks"]
    print("\n" + "=" * 60)
    print(f"PASS TEST, {mode}: {'PASSED' if result['passed'] else 'FAILED'}")
    print("=" * 60)
    print(f"Driven              model {ml['km']:.1f} km, autopilot {autopilot['km']:.1f} km")
    for name, label in (("collisions", "Collisions per km"), ("red_runs", "Red lights per km")):
        c = checks[name]
        print(f"{label:<20}model {c['ml_per_km']:.3f}, autopilot {c['autopilot_per_km']:.3f}, limit {c['limit_per_km']:.3f}: {'pass' if c['passed'] else 'FAIL'}")
    d = checks["distance"]
    print(f"Distance            {100 * d['share_of_autopilot']:.0f}% of autopilot's, limit {100 * d['limit']:.0f}%: {'pass' if d['passed'] else 'FAIL'}")
    print(f"Model's collisions  {ml['collisions_by_kind']}; lost its lane {ml['times_lost']} times; stuck {ml['got_stuck']} times; "
          f"{ml['brake_requests']} brake requests; {ml['missed_ticks']} ticks missed")
    print(f"Autopilot's         {autopilot['collisions_by_kind']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--minutes", type=float, default=5.0, help="simulated minutes per run")
    parser.add_argument("--first-seed", type=int, default=21)
    parser.add_argument("--ignore-brake", action="store_true", help="measure the model alone")
    parser.add_argument("--watch-autopilot", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--watch-out", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.watch_autopilot is not None:
        watch_autopilot(args.watch_autopilot, args.watch_out)
        return 0

    policy = DrivingPolicy()
    import carla

    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    mode = "brake ignored" if args.ignore_brake else "brake active"
    seconds = args.minutes * 60.0
    watch_path = OUTPUT_DIR / f"pass_test_autopilot_{stamp}.json"
    watcher = subprocess.Popen([sys.executable, "-u", __file__, "--watch-autopilot", str(seconds * args.runs),
                                "--watch-out", str(watch_path)])
    time.sleep(2.0)

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(30.0)
    world = client.get_world()
    reports = []
    print(f"[PASS TEST] {args.runs} runs of {args.minutes:g} simulated minutes, {mode}.", flush=True)
    for seed in range(args.first_seed, args.first_seed + args.runs):
        report = asdict(drive(carla, world, policy, seconds, seed, args.ignore_brake, release_hold=BRAKE_HOLD))
        report["planned_seconds"] = seconds
        reports.append(report)
        print(f"[PASS TEST] seed {seed}: {report['distance']:.0f} m, {len(report['collisions'])} collisions, "
              f"{len(report['red_light_runs'])} red lights, lost lane {report['times_lost']}, stuck {report['got_stuck']}, "
              f"{report['brake_requests']} brake requests", flush=True)
    watcher.wait(timeout=seconds * args.runs + 120.0)

    autopilot = json.loads(watch_path.read_text())
    ml = summarise(reports)
    result = verdict(ml, autopilot)
    print_verdict(result, ml, autopilot, mode)
    path = OUTPUT_DIR / f"pass_test_{'alone' if args.ignore_brake else 'with_brake'}_{stamp}.json"
    path.write_text(json.dumps({"mode": mode, "result": result, "model": ml, "autopilot": autopilot, "runs": reports}, indent=2))
    print(f"\nReport: {path}", flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
