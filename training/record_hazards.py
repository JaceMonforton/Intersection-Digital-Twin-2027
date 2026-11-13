"""
Teaches the driving model to give way the way the safety gate would, but in time: to slow and stop short of where a
car or person is about to cross its route, rather than carry on until the gate brakes it in the junction.

The safety gate brakes on a predicted path conflict within 3 s, which is often too late to stop short of it, and in
the simulation it no longer holds the car, so the model must do this itself. A rule labels the three scenarios
(left_turn, oncoming_turner, hidden_pedestrian): from exactly what the model is given, it finds the nearest point on
the route that a car or person, keeping its speed and heading, will reach before the ML car has got past it, and asks
for a speed that stops short of it at a steady deceleration. On a left turn the left-turn rule of
record_left_turns.py decides about oncoming cars, as it already does. What the car saw and the speed the rule asked
for become training samples; train_hazards.py learns them.

This rule only teaches. It isn't used when the model drives. --model-share runs let the model drive while the rule
still labels, which records the mistakes the model itself makes. Needs the processor running with --detector
groundtruth, so that the lessons see every actor; nothing else may be using CARLA.
    python training/record_hazards.py --runs 60
    python training/record_hazards.py --runs 60 --model-share 0.5
"""
import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_professor_test_suite as suite_module
from driving_features import OUTPUT_SIZE, decode_inputs
from ml_driver import DrivingPolicy
from record_driving import CONVERTED_DIR
from record_left_turns import teacher_target
from run_professor_test_suite import Console
from run_rsu_comparison import LEADS, Comparison
from waymo_predictor_node import LEFT, world_to_ego

# --- Configuration ---
HAZARD_FILES = "hazard_"  # train_hazards.py draws a set share from files with this prefix
LOOK = 4.0          # s ahead that an actor's path is followed
STEP = 0.1          # s
REACH = {"car": 3.0, "walker": 2.0}  # m from a route point that an actor counts as on it
AFTER = 2.0         # s after the ML car would pass a point that an actor reaching it still makes it wait
SAME_WAY = 0.7      # cos of the angle under which a car is going along the route: following it is the general driver's job
MOVING = 1.0        # m/s
# m from the car's centre to where it stops short of a conflict: half its length and a margin, more for a car, whose
# body and turning sweep reach further. At 4 m for cars it stopped inside the oncoming turner's path and was hit.
STOP_SHORT = {"car": 6.5, "walker": 4.0}
BRAKING = 3.0       # m/s^2; firm but not an emergency stop
CLOSE = 0.5         # m of room under which it asks to stop now
MATCH = 1.5         # m between a seen car and a CARLA one that makes them the same
SCENARIOS = ("left_turn", "oncoming_turner", "hidden_pedestrian")


def route_distances(lane):
    """Metres along the route from the car to each of its points. lane is (x, y) in the car's frame, nearest first."""
    points = np.vstack(([0.0, 0.0], np.asarray(lane, float)))
    return np.cumsum(np.hypot(*np.diff(points, axis=0).T))


def conflict_distance(seen, cars=True, kinds=("car", "walker")):
    """Metres along the route to the nearest point that an actor will be on within AFTER s of when the ML car gets
    there, or inf. Each actor is (x, y, vx, vy) in the car's frame, or (x, y, vx, vy, yaw rate in rad/s) for one that's
    turning, which keeps turning at that rate: a car turning across the route only points across it once it's late."""
    lane = np.asarray(seen["lane"], float)
    if len(lane) < 2:
        return math.inf
    distances = route_distances(lane)
    tangents = np.gradient(lane, axis=0)
    tangents /= np.maximum(np.hypot(tangents[:, 0], tangents[:, 1]), 1e-6)[:, None]
    times = np.arange(0.0, LOOK + STEP / 2, STEP)
    passes = distances / max(seen["speed"], MOVING)  # when the ML car would be at each point, at its speed
    best = math.inf
    for kind, actors in (("car", seen["cars"] if cars and "car" in kinds else []), ("walker", seen["walkers"] if "walker" in kinds else [])):
        for actor in actors:
            x, y, vx, vy = actor[:4]
            path = predicted_path(x, y, vx, vy, actor[4] if len(actor) > 4 else 0.0, times)
            gaps = np.hypot(path[:, None, 0] - lane[None, :, 0], path[:, None, 1] - lane[None, :, 1])
            for point in np.flatnonzero((gaps < REACH[kind]).any(axis=0)):
                speed = math.hypot(vx, vy)
                if kind == "car" and speed > MOVING and (tangents[point] @ (vx, vy)) / speed > SAME_WAY:
                    continue
                on = np.flatnonzero(gaps[:, point] < REACH[kind])
                arrives = times[on[0]]
                leaves = math.inf if on[-1] == len(times) - 1 else times[on[-1]]  # still there at the end: assume it stays
                if arrives <= passes[point] + AFTER and leaves >= passes[point] - AFTER:
                    best = min(best, float(distances[point]))
    return best


def predicted_path(x, y, vx, vy, turning, times):
    """(x, y) at each time, keeping speed and turning at a steady rate (rad/s, positive to the left)."""
    if abs(turning) < 1e-3:
        return np.c_[x + vx * times, y + vy * times]
    speed, heading = math.hypot(vx, vy), math.atan2(vy, vx)
    angles = heading + turning * times
    return np.c_[x + speed / turning * (np.sin(angles) - math.sin(heading)), y - speed / turning * (np.cos(angles) - math.cos(heading))]


def with_turning(cars, world):
    """The seen cars with each one's yaw rate from CARLA, matched by position, in the car's frame. Only the teacher
    knows this; the model has to judge a turn from where a car is and the way it's heading."""
    if world is None or not cars:
        return cars
    vehicles = list(world.get_actors().filter("vehicle.*"))
    ml = next((v for v in vehicles if v.attributes.get("role_name") == "vehicle_ml"), None)
    if ml is None:
        return cars
    tf = ml.get_transform()
    others = [v for v in vehicles if v.id != ml.id]
    if not others:
        return cars
    places = world_to_ego(np.array([(v.get_location().x, v.get_location().y) for v in others]), tf.location.x, tf.location.y, tf.rotation.yaw)
    turned = []
    for car in cars:
        gaps = np.hypot(places[:, 0] - car[0], places[:, 1] - car[1])
        nearest = int(np.argmin(gaps))
        # CARLA is left-handed with y right, so its yaw rate is the other way round to the car's frame
        rate = -math.radians(others[nearest].get_angular_velocity().z) if gaps[nearest] < MATCH else 0.0
        turned.append((*car[:4], rate))
    return turned


def hazard_target(seen, model_target):
    """The speed to ask for 1 s from now: the left-turn rule's, then no faster than stops short of a conflict."""
    target = teacher_target(seen, model_target)
    room = min(conflict_distance(seen, cars=seen["route"] != LEFT, kinds=(kind,)) - STOP_SHORT[kind] for kind in STOP_SHORT)
    if room < CLOSE:
        return 0.0
    return min(target, math.sqrt(2.0 * BRAKING * room))


class HazardTeacher:
    """Runs the model and records what the rule would ask for on every route. Drives with the rule's speed unless
    model_drives."""

    def __init__(self, policy):
        self.policy = policy
        self.world = None  # set to the CARLA world, so that turning cars are followed round their turn
        self.model_drives = False
        self.reset()

    def reset(self):
        self.inputs, self.targets, self.routes = [], [], []
        self.slowed = 0

    def __call__(self, inputs):
        output = np.array(self.policy(inputs), dtype=np.float32)
        seen = decode_inputs(inputs)
        target = hazard_target({**seen, "cars": with_turning(seen["cars"], self.world)}, float(output[1]))
        self.slowed += target < float(output[1]) - 0.5
        self.inputs.append(np.array(inputs))
        self.targets.append((output[0], target))
        self.routes.append(seen["route"])
        return output if self.model_drives else np.array((output[0], target), dtype=np.float32)


def run_one(suite, name, lead, rng):
    if name == "left_turn":
        # So that it meets gaps near the limit, and traffic at different moments, as record_left_turns.py does
        suite_module.T2_START_BACK = rng.uniform(5.0, 15.0)
        suite_module.T2_FIRST_WAVE = rng.uniform(0.0, 30.0)
        suite_module.T2_PREFILL_TO = rng.uniform(0.0, 100.0)
        suite_module.T2_HEADWAY = (2.0, rng.uniform(4.0, 10.0))
        suite_module.T2_GAP = (rng.uniform(4.0, 11.0), 14.0)
        return suite.test_left_turn()
    return (suite.run_turner if name == "oncoming_turner" else suite.run_pedestrian)(lead)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=60, help="runs in all, taking the three scenarios in turn")
    parser.add_argument("--model-share", type=float, default=0.0, help="share of runs where the model drives and the rule only labels")
    parser.add_argument("--traffic", type=int, nargs=2, default=(0, 15), metavar=("LOW", "HIGH"), help="background cars, drawn per run")
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS), help="taken in turn")
    args = parser.parse_args()

    import carla

    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    teacher = HazardTeacher(DrivingPolicy(args.model) if args.model else DrivingPolicy())
    suite = Comparison(carla, client, teacher, Console(quiet=True), perception="truth")
    teacher.world = suite.world
    rng = random.Random(args.seed)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    clean = 0
    try:
        for run in range(args.runs):
            name = args.scenarios[run % len(args.scenarios)]
            teacher.reset()
            teacher.model_drives = rng.random() < args.model_share
            suite.seed = rng.randrange(1 << 30)
            suite.traffic = rng.randint(*args.traffic)
            lead = rng.choice(LEADS) + rng.uniform(-0.3, 0.3)
            try:
                result = run_one(suite, name, lead, rng)
            except RuntimeError as error:
                print(f"[TEACH] run {run + 1}/{args.runs} {name} couldn't run: {error}", flush=True)
                continue
            safe = not result.collisions
            clean += safe
            print(f"[TEACH] run {run + 1}/{args.runs} {name}, {suite.traffic} cars ({'model' if teacher.model_drives else 'rule'} drove): "
                  f"{'no collision' if safe else 'COLLISION'}, {len(teacher.inputs)} samples, {teacher.slowed} slowed", flush=True)
            if teacher.inputs:
                np.savez_compressed(CONVERTED_DIR / f"{HAZARD_FILES}{stamp}_{run:03d}.npz", inputs=np.stack(teacher.inputs),
                                    targets=np.array(teacher.targets, np.float32).reshape(-1, OUTPUT_SIZE), routes=np.array(teacher.routes, np.int8))
    finally:
        suite.close()
    print(f"[TEACH] {clean} of {args.runs} runs had no collision.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
