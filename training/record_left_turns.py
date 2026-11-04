"""
Teaches the driving model to wait for a safe gap before turning left across traffic.

Autopilot can't teach this here: this junction gives one approach a green at a time, so in the recordings
no car ever turns left across moving traffic. Instead, a rule built on time-to-collision labels the left-turn
test: it waits just inside the junction, short of the oncoming lanes, until the next car will reach the route
later than the car needs to get across, plus a margin. If the light changes meanwhile it finishes the turn on
the same terms. What the car saw and the speed the rule asked for become training samples. Each run gets
different traffic.

This rule only teaches. It isn't used when the model drives. --model-share runs let the model drive while the
rule still labels, which records the mistakes the model itself makes. Needs the processor running.
    python training/record_left_turns.py --runs 20
    python training/record_left_turns.py --runs 20 --model-share 1.0
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
from run_professor_test_suite import Console, Suite
from waymo_predictor_node import LEFT

# --- Configuration ---
YIELD_FILES = "yield_"  # train_driving.py draws a set share from files with this prefix
HOLD_INSIDE = 9.0     # m past the start of the turn to wait; the nearest oncoming lane is reached at 16 m
COMMITTED_INSIDE = 12.5  # m past the start of the turn with the bumper near that lane, so it must get across
CLEAR_INSIDE = 28.0   # m past the start of the turn that the car has left the far oncoming lane
GO_SPEED = 4.5        # m/s through the turn
PULL_AWAY = 1.5       # m/s^2 the car manages from a stop
START_LAG = 0.8       # s before a stopped car really gets moving
SET_OFF_MARGIN = 3.0  # s between clearing the crossing and the next car reaching it, to set off from a stop
MOVING_MARGIN = 1.5   # s to keep going once moving; smaller, so that it doesn't change its mind halfway
MOVING = 1.0          # m/s
YIELD_BRAKE = 1.0     # m/s^2 for slowing to the waiting point, gently since the speed controller lags
STOP_WITHIN = 1.0     # m of the waiting point that counts as there


def time_to_clear(distance, speed, top=GO_SPEED, acceleration=PULL_AWAY):
    """Seconds to cover this distance from this speed, speeding up to top."""
    speed = min(speed, top)
    speeding_up = (top - speed) / acceleration
    covered = (speed + top) / 2.0 * speeding_up
    if covered >= distance:
        return (math.sqrt(speed ** 2 + 2.0 * acceleration * distance) - speed) / acceleration
    return speeding_up + (distance - covered) / top


def is_safe_to_go(ttc, inside, speed):
    """ttc is seconds until the first car reaches the route, or None. inside is metres past the start of the turn."""
    if ttc is None:
        return True
    stopped = speed < MOVING
    return ttc > time_to_clear(CLEAR_INSIDE - inside, speed) + (START_LAG + SET_OFF_MARGIN if stopped else MOVING_MARGIN)


def teacher_target(seen, model_target):
    """The speed to ask for 1 s from now on this left turn, given what the model wanted."""
    if seen["route"] != LEFT or seen["stop_gap"] is not None:
        return model_target
    inside = -seen["turn_gap"]
    safe = is_safe_to_go(seen["crossing_ttc"], inside, seen["speed"])
    if inside >= COMMITTED_INSIDE and not safe and seen["speed"] < MOVING:
        return 0.0  # stopped further on than it should be: going now would be worse than staying
    if inside >= COMMITTED_INSIDE or safe:
        return max(model_target, GO_SPEED) if inside >= 0.0 else model_target
    room = HOLD_INSIDE - inside
    limit = 0.0 if room < STOP_WITHIN else math.sqrt(2.0 * YIELD_BRAKE * room)
    return min(limit, max(model_target, GO_SPEED))


class TeacherPolicy:
    """Runs the model and records what the rule would ask for. Drives with the rule's speed unless model_drives."""

    def __init__(self, policy):
        self.policy = policy
        self.model_drives = False
        self.reset()

    def reset(self):
        self.inputs, self.targets, self.routes = [], [], []
        self.waited = 0

    def __call__(self, inputs):
        output = np.array(self.policy(inputs), dtype=np.float32)
        seen = decode_inputs(inputs)
        if seen["route"] != LEFT:
            return output
        target = teacher_target(seen, float(output[1]))
        self.waited += target < 0.5
        self.inputs.append(np.array(inputs))
        self.targets.append((output[0], target))
        self.routes.append(seen["route"])
        return output if self.model_drives else np.array((output[0], target), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--model-share", type=float, default=0.0, help="share of runs where the model drives and the rule only labels")
    parser.add_argument("--light-change-share", type=float, default=0.0, help="share of runs where the light turns yellow while it waits")
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import carla

    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    teacher = TeacherPolicy(DrivingPolicy(args.model) if args.model else DrivingPolicy())
    suite = Suite(carla, client, teacher, Console(quiet=True))
    rng = random.Random(args.seed)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    clean = 0
    try:
        for run in range(args.runs):
            # So that it sees gaps near the limit, and meets traffic at different moments and speeds
            suite_module.T2_START_BACK = rng.uniform(5.0, 15.0)
            suite_module.T2_FIRST_WAVE = rng.uniform(0.0, 30.0)
            suite_module.T2_PREFILL_TO = rng.uniform(0.0, 100.0)
            suite_module.T2_HEADWAY = (2.0, rng.uniform(4.0, 10.0))
            suite_module.T2_GAP = (rng.uniform(4.0, 11.0), 14.0)
            teacher.reset()
            teacher.model_drives = rng.random() < args.model_share
            suite.seed = rng.randrange(1 << 30)
            result = suite.test_left_turn(light_change=rng.random() < args.light_change_share)
            clean += result.passed
            print(f"[TEACH] run {run + 1}/{args.runs} ({'model' if teacher.model_drives else 'rule'} drove): {'PASS' if result.passed else 'FAIL'} "
                  f"{[name for name, ok in result.checks.items() if not ok]} {result.notes} {len(teacher.inputs)} samples, {teacher.waited} waiting", flush=True)
            if teacher.inputs:
                np.savez_compressed(CONVERTED_DIR / f"{YIELD_FILES}{stamp}_{run:03d}.npz", inputs=np.stack(teacher.inputs),
                                    targets=np.array(teacher.targets, np.float32).reshape(-1, OUTPUT_SIZE), routes=np.array(teacher.routes, np.int8))
    finally:
        suite.close()
    print(f"[TEACH] {clean} of {args.runs} runs passed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
