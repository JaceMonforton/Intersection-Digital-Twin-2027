"""
Measures the roadside unit's tracks against CARLA's true positions, live, in traffic. Unlike eval_detection.py this
covers the whole chain: the sensors, the detector, the tracker, and the message to the car.

Needs the processor running with --detector onnx. It starts its own traffic and pedestrians, and only reads the
world otherwise.
    python training/eval_tracking.py --seconds 180 --seed 11 --output output/rsu_tracking.json

--traffic-only starts the same traffic and scores nothing, which leaves the roadside port free for the ML car:
    python training/eval_tracking.py --seconds 300 --seed 12 --traffic-only
"""
import argparse
import json
import math
import signal
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "training")]
from central_gpu_processor import (
    CARLA_HOST,
    CARLA_PORT,
    INTERSECTION_JUNCTION,
    find_junction,
)
from record_detection import (
    WALKER_STEER_TICKS,
    spawn_walkers,
    start_traffic,
    steer_walkers,
    stop_traffic,
    traffic_level,
)
from roadside_link import RoadsideReceiver

MATCH = {"cars": 3.0, "walkers": 1.5}  # m between a track and an actor for them to be the same one
NEAR = 30.0             # m from the junction centre, where the coverage is complete
MOVING = 1.0            # m/s; speed and heading errors only mean something for what's moving


def score(tracks, actors, reach, center_xy):
    """Matches tracks to actors, nearest first. Returns (per actor: (position error, speed error, heading error) or None, ghosts).
    tracks and actors are (x, y, vx, vy[, yaw]). A heading error is None unless the actor is moving."""
    free = list(tracks)
    found = []
    for actor in actors:
        best = min(free, key=lambda t: math.hypot(t[0] - actor[0], t[1] - actor[1]), default=None)
        if best is None or math.hypot(best[0] - actor[0], best[1] - actor[1]) > reach:
            found.append(None)
            continue
        free.remove(best)
        speed = math.hypot(actor[2], actor[3])
        heading = None
        if speed >= MOVING and len(best) > 4:
            heading = abs((best[4] - math.degrees(math.atan2(actor[3], actor[2])) + 180.0) % 360.0 - 180.0)
        found.append((math.hypot(best[0] - actor[0], best[1] - actor[1]), math.hypot(best[2] - actor[2], best[3] - actor[3]) if speed >= MOVING else None, heading))
    near_ghosts = [t for t in free if math.hypot(t[0] - center_xy[0], t[1] - center_xy[1]) <= NEAR]
    # How far each is from the nearest actor at all: close means a duplicate of a real track, far means a false one
    apart = [min((math.hypot(t[0] - a[0], t[1] - a[1]) for a in actors), default=math.inf) for t in near_ghosts]
    return found, apart


def summarise(found, near, apart, ticks):
    """Totals from every (matched result or None), for actors within NEAR of the centre."""
    hit = [f for f, n in zip(found, near) if f is not None and n]
    speeds = [f[1] for f in hit if f[1] is not None]
    headings = [f[2] for f in hit if f[2] is not None]
    total = sum(near)
    return {"actor_ticks_near": total, "tracked_share": round(len(hit) / total, 3) if total else None,
            "position_error_median_m": round(float(np.median([f[0] for f in hit])), 3) if hit else None,
            "position_error_p95_m": round(float(np.percentile([f[0] for f in hit], 95)), 3) if hit else None,
            "speed_error_median_ms": round(float(np.median(speeds)), 3) if speeds else None,
            "speed_error_p95_ms": round(float(np.percentile(speeds, 95)), 3) if speeds else None,
            "heading_error_median_deg": round(float(np.median(headings)), 1) if headings else None,
            "heading_error_p95_deg": round(float(np.percentile(headings, 95)), 1) if headings else None,
            "ghost_tracks_per_second": round(len(apart) / max(ticks * 0.1, 1e-6), 3),
            "ghosts_within_3m_of_an_actor": round(float(np.mean(np.array(apart) <= 3.0)), 3) if apart else None,
            "ghost_distance_to_actor_median_m": round(float(np.median(apart)), 2) if apart else None}


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output")
    parser.add_argument("--traffic-only", action="store_true", help="don't listen or score, so scripts/ml_driver.py --perception rsu can listen")
    args = parser.parse_args()
    for name in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _raise_keyboard_interrupt)

    import carla

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(60.0)
    world = client.get_world()
    center = find_junction(world, INTERSECTION_JUNCTION).bounding_box.location
    vehicles, pedestrians = traffic_level(args.seed)
    log_path = ROOT / "output" / "eval_tracking_traffic.log"
    log_path.parent.mkdir(exist_ok=True)
    receiver = None if args.traffic_only else RoadsideReceiver()
    walkers = []
    results = {"cars": [], "walkers": []}
    near = {"cars": [], "walkers": []}
    ghosts = {"cars": [], "walkers": []}
    stale, ticks = 0, 0
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        traffic = start_traffic(args.seed, vehicles, log)
        try:
            while len(world.get_actors().filter("vehicle.*")) < 0.8 * vehicles:
                world.wait_for_tick(10.0)
            walkers, spots, crossings, rng = spawn_walkers(carla, client, world, center, pedestrians, args.seed)
            warm = 0
            while warm < 300:
                snapshot = world.wait_for_tick(10.0)
                if warm % WALKER_STEER_TICKS == 0:
                    steer_walkers(carla, walkers, snapshot, spots, crossings, rng)
                warm += 1
            while ticks < args.seconds * 10:
                snapshot = world.wait_for_tick(10.0)
                ticks += 1
                if ticks % WALKER_STEER_TICKS == 0:
                    steer_walkers(carla, walkers, snapshot, spots, crossings, rng)
                if receiver is None:
                    continue
                message = receiver.poll()
                now = snapshot.timestamp.elapsed_seconds
                if message is None or abs(now - message["t"]) > 0.5:
                    stale += 1
                    continue
                actors = {"cars": world.get_actors().filter("vehicle.*"), "walkers": world.get_actors().filter("walker.pedestrian.*")}
                for kind, group in actors.items():
                    states = []
                    for actor in group:
                        state = snapshot.find(actor.id)
                        if state is None:
                            continue
                        place, v = state.get_transform().location, state.get_velocity()
                        states.append((place.x, place.y, v.x, v.y))
                    found, extra = score(message[kind], states, MATCH[kind], (center.x, center.y))
                    results[kind] += found
                    near[kind] += [math.hypot(s[0] - center.x, s[1] - center.y) <= NEAR for s in states]
                    ghosts[kind] += extra
        finally:
            client.apply_batch_sync([carla.command.DestroyActor(w.actor.id) for w in walkers])
            stop_traffic(traffic)
            if receiver is not None:
                receiver.close()
    if args.traffic_only:
        return
    report = {"seconds": ticks * 0.1, "seed": args.seed, "vehicles": vehicles, "pedestrians": pedestrians, "stale_ticks": stale,
              "cars": summarise(results["cars"], near["cars"], ghosts["cars"], ticks), "walkers": summarise(results["walkers"], near["walkers"], ghosts["walkers"], ticks)}
    text = json.dumps(report, indent=1)
    print(text)
    if args.output:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
