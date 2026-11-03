"""
Converts Waymo Open Motion Dataset tf_example shards into samples for train_trajectory.py.

Each sample is one vehicle at one moment, in that vehicle's own frame (x forward, y left):
    inputs:  10 x (x, y, vx, vy), the last 1 s, oldest first, ending on the current frame
    targets: 30 x (x, y), the next 3 s

Usage:
    python training/convert_waymo.py data/waymo/training data/waymo/converted/training.npz
    python training/convert_waymo.py data/waymo/validation data/waymo/converted/validation.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HISTORY = 10
FUTURE = 30
STEP_SECONDS = 0.1
VEHICLE_TYPE = 1
MAX_AGENTS = 128
PAST_FRAMES = 10
FUTURE_FRAMES = 80

PARKED_METRES = 1.0      # a vehicle moving less than this over the whole sample counts as parked
PARKED_KEEP_SHARE = 0.05 # parked cars swamp the data, but the model must still learn to hold still
TURN_DEGREES = 30.0      # heading change over the 3 s future that counts as a turn
TURN_MIN_METRES = 5.0    # a sample must travel this far for its heading change to mean anything
STRAIGHT, LEFT, RIGHT = 0, 1, 2

SIGNALS = ("x", "y", "velocity_x", "velocity_y", "bbox_yaw")


# ---------------------------------------------------------------------------
# Pure logic (no TensorFlow needed)
# ---------------------------------------------------------------------------
def to_agent_frame(points, origin_x, origin_y, yaw):
    """Waymo's world frame is already right-handed, so this only rotates. Don't reuse the predictor's
    world_to_ego here: it also flips y, because CARLA's frame is left-handed, which would mirror
    the data and teach the model that left turns go right. yaw is in radians."""
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = points[..., 0] - origin_x, points[..., 1] - origin_y
    return np.stack((c * dx + s * dy, -s * dx + c * dy), axis=-1)


def rotate_to_agent_frame(vectors, yaw):
    return to_agent_frame(vectors, 0.0, 0.0, yaw)


def classify_turn(target):
    """STRAIGHT, LEFT or RIGHT, from how far the vehicle's direction of travel has swung by the end
    of a 30 x (x, y) target in its own frame. Positive y is left."""
    if np.hypot(*target[-1]) < TURN_MIN_METRES:
        return STRAIGHT
    end = target[-1] - target[-6]
    if np.hypot(*end) < 0.5:  # stopped by the end, so judge by where it ended up
        end = target[-1]
    heading = np.degrees(np.arctan2(end[1], end[0]))
    if heading > TURN_DEGREES:
        return LEFT
    return RIGHT if heading < -TURN_DEGREES else STRAIGHT


def scene_samples(past, current, future, valid_past, valid_current, valid_future, types, rng):
    """Samples from one scene. past, current and future map a signal name to an array of
    [agents, frames]; the valid_ arrays are masks of the same shape; types is [agents]."""
    inputs, targets, turns = [], [], []
    frames = {k: np.concatenate((past[k][:, 1:], current[k]), axis=1) for k in SIGNALS}
    valid = np.concatenate((valid_past[:, 1:], valid_current, valid_future[:, :FUTURE]), axis=1)
    for agent in range(len(types)):
        if types[agent] != VEHICLE_TYPE or not valid[agent].all():
            continue
        x, y = frames["x"][agent, -1], frames["y"][agent, -1]
        yaw = frames["bbox_yaw"][agent, -1]
        history_points = np.stack((frames["x"][agent], frames["y"][agent]), axis=-1)
        velocity = np.stack((frames["velocity_x"][agent], frames["velocity_y"][agent]), axis=-1)
        future_points = np.stack((future["x"][agent, :FUTURE], future["y"][agent, :FUTURE]), axis=-1)

        ego_history = to_agent_frame(history_points, x, y, yaw)
        ego_velocity = rotate_to_agent_frame(velocity, yaw)
        ego_future = to_agent_frame(future_points, x, y, yaw)

        moved = max(np.hypot(*ego_history[0]), np.max(np.hypot(ego_future[:, 0], ego_future[:, 1])))
        if moved < PARKED_METRES and rng.random() > PARKED_KEEP_SHARE:
            continue
        inputs.append(np.concatenate((ego_history, ego_velocity), axis=1))
        targets.append(ego_future)
        turns.append(classify_turn(ego_future))
    return inputs, targets, turns


# ---------------------------------------------------------------------------
# Reading TFRecords
# ---------------------------------------------------------------------------
def feature_spec(tf):
    spec = {"state/type": tf.io.FixedLenFeature([MAX_AGENTS], tf.float32)}
    for section, frames in (("past", PAST_FRAMES), ("current", 1), ("future", FUTURE_FRAMES)):
        for signal in SIGNALS:
            spec[f"state/{section}/{signal}"] = tf.io.FixedLenFeature([MAX_AGENTS, frames], tf.float32)
        spec[f"state/{section}/valid"] = tf.io.FixedLenFeature([MAX_AGENTS, frames], tf.int64)
    return spec


def convert(source, destination, limit=None, seed=0):
    import tensorflow as tf

    files = sorted(str(p) for p in Path(source).glob("*tfrecord*"))
    if not files:
        raise FileNotFoundError(f"No tfrecord files in {source}")
    spec = feature_spec(tf)
    rng = np.random.default_rng(seed)
    dataset = tf.data.TFRecordDataset(files).map(lambda r: tf.io.parse_single_example(r, spec),
                                                 num_parallel_calls=tf.data.AUTOTUNE)
    inputs, targets, turns = [], [], []
    scenes = 0
    for scene in dataset.as_numpy_iterator():
        sections = {name: {k: scene[f"state/{name}/{k}"] for k in SIGNALS} for name in ("past", "current", "future")}
        new_inputs, new_targets, new_turns = scene_samples(
            sections["past"], sections["current"], sections["future"],
            scene["state/past/valid"].astype(bool), scene["state/current/valid"].astype(bool),
            scene["state/future/valid"].astype(bool), scene["state/type"], rng)
        inputs += new_inputs
        targets += new_targets
        turns += new_turns
        scenes += 1
        if scenes % 500 == 0:
            print(f"[CONVERT] {scenes} scenes, {len(inputs)} samples", flush=True)
        if limit is not None and scenes >= limit:
            break

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    turn_array = np.asarray(turns, dtype=np.int8)
    np.savez(destination, inputs=np.asarray(inputs, dtype=np.float32),
             targets=np.asarray(targets, dtype=np.float32), turns=turn_array)
    print(f"[CONVERT] {scenes} scenes -> {len(inputs)} samples: {(turn_array == STRAIGHT).sum()} straight, "
          f"{(turn_array == LEFT).sum()} left, {(turn_array == RIGHT).sum()} right. Wrote {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="folder of tf_example tfrecord files")
    parser.add_argument("destination", help="output .npz")
    parser.add_argument("--limit", type=int, help="stop after this many scenes")
    args = parser.parse_args()
    convert(args.source, args.destination, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
