"""
Measures the roadside detector on a recorded seed. Run it with a seed that wasn't trained on. Seed 28 is for
choosing the thresholds, and seed 29 is for the final figures, so seed 29 should only be used once a model is
chosen. Needs only numpy and onnxruntime.

    python training/eval_detection.py models/rsu_detector.candidate.onnx --seed 28
    python training/eval_detection.py models/rsu_detector.candidate.onnx --seed 29 --threshold 0.4 0.3

"Seen" actors are those with at least MIN_POINTS LiDAR points on them. A box beside an actor with fewer points
isn't counted as a mistake, nor is missing that actor. "On the road" is approximate: it's within ROAD_REACH of
where any recorded vehicle has been.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from detection_features import (
    CLASSES,
    GRID_HALF,
    MIN_POINTS,
    PEDESTRIAN,
    VEHICLE,
    decode,
    load_sweep,
    rasterize,
)

RAW_DIR = ROOT / "data" / "detection" / "raw"
MATCH_DISTANCE = {VEHICLE: 2.0, PEDESTRIAN: 1.0}  # m between a box's centre and an actor's
NEAR = 30.0             # m from the junction centre, where the coverage is complete
ROAD_REACH = 2.0        # m
ROAD_CELL = 1.0         # m
LOW_THRESHOLD = 0.05    # scores are kept down to here, so that any threshold can be measured afterwards
CLASS_NAMES = {VEHICLE: "vehicles", PEDESTRIAN: "pedestrians"}


def match(detections, labels, kind, distance=None):
    """Matches boxes of one class to actors of that class, best score first. Returns (scores, is_true) for every
    box that counts, plus found, the score each seen actor was found with, or 0. A box near an actor too hidden to
    count is left out, as it's neither right nor wrong."""
    distance = MATCH_DISTANCE[kind] if distance is None else distance
    boxes = detections[detections[:, 8] == kind]
    boxes = boxes[np.argsort(-boxes[:, 7])]
    actors = labels[labels[:, 7] == kind]
    seen, hidden = actors[actors[:, 8] >= MIN_POINTS], actors[actors[:, 8] < MIN_POINTS]
    found = np.zeros(len(seen), dtype=np.float32)
    errors = np.full(len(seen), np.nan, dtype=np.float32)
    scores, is_true = [], []
    for box in boxes:
        gaps = np.hypot(seen[:, 0] - box[0], seen[:, 1] - box[1])
        gaps[found > 0] = np.inf
        best = int(np.argmin(gaps)) if len(gaps) else -1
        if best >= 0 and gaps[best] <= distance:
            found[best] = box[7]
            errors[best] = gaps[best]
            scores.append(box[7])
            is_true.append(True)
        elif len(hidden) == 0 or np.hypot(hidden[:, 0] - box[0], hidden[:, 1] - box[1]).min() > distance:
            scores.append(box[7])
            is_true.append(False)
    return np.array(scores, np.float32), np.array(is_true, bool), seen, found, errors


def average_precision(scores, is_true, total):
    """The area under the precision and recall curve, with precision taken as the best at that recall or above."""
    if total == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    hits = np.cumsum(is_true[order])
    precision = hits / np.arange(1, len(order) + 1)
    recall = hits / total
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * envelope))


def road_mask(labels_by_sweep):
    """Cells within ROAD_REACH of anywhere a vehicle has been in these recordings."""
    size = round(2 * GRID_HALF / ROAD_CELL)
    mask = np.zeros((size, size), dtype=bool)
    vehicles = np.vstack(labels_by_sweep)
    vehicles = vehicles[vehicles[:, 7] == VEHICLE]
    cells = np.clip(((vehicles[:, :2] + GRID_HALF) / ROAD_CELL).astype(int), 0, size - 1)
    mask[cells[:, 0], cells[:, 1]] = True
    reach = round(ROAD_REACH / ROAD_CELL)
    grown = np.zeros_like(mask)
    for dr in range(-reach, reach + 1):
        for dc in range(-reach, reach + 1):
            grown |= np.roll(mask, (dr, dc), axis=(0, 1))
    return grown


def on_road(mask, xy):
    cells = np.clip(((xy + GRID_HALF) / ROAD_CELL).astype(int), 0, mask.shape[0] - 1)
    return mask[cells[:, 0], cells[:, 1]]


def choose_threshold(scores, is_true, total):
    """The score with the best F1, which weighs a missed actor and a false box evenly."""
    order = np.argsort(-scores)
    hits = np.cumsum(is_true[order])
    precision, recall = hits / np.arange(1, len(order) + 1), hits / max(total, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return float(scores[order][int(np.argmax(f1))]) if len(order) else 0.5


def evaluate(session, files, thresholds=None, road=None):
    name = session.get_inputs()[0].name
    collected = {kind: {"scores": [], "true": [], "seen": [], "found": [], "errors": []} for kind in range(CLASSES)}
    frames, seconds = 0, []
    for path in files:
        points, labels = load_sweep(path)
        started = time.perf_counter()
        output = session.run(None, {name: rasterize(points)[None]})[0][0]
        detections = decode(output, LOW_THRESHOLD)
        seconds.append(time.perf_counter() - started)
        frames += 1
        for kind, into in collected.items():
            scores, is_true, seen, found, errors = match(detections, labels, kind)
            into["scores"].append(scores)
            into["true"].append(is_true)
            into["seen"].append(seen)
            into["found"].append(found)
            into["errors"].append(errors)

    result = {"sweeps": frames, "milliseconds_median": float(np.median(seconds[1:] if frames > 1 else seconds) * 1000)}
    for kind, into in collected.items():
        scores, is_true = np.concatenate(into["scores"]), np.concatenate(into["true"])
        seen, found, errors = np.vstack(into["seen"]), np.concatenate(into["found"]), np.concatenate(into["errors"])
        threshold = thresholds[kind] if thresholds else choose_threshold(scores, is_true, len(seen))
        hit = found >= threshold
        distance = np.hypot(seen[:, 0], seen[:, 1])
        groups = {"all": np.ones(len(seen), bool), f"within_{NEAR:.0f}m": distance <= NEAR}
        if road is not None:
            groups["on_road"] = on_road(road, seen[:, :2])
            groups["off_road"] = ~groups["on_road"]
            groups[f"on_road_within_{NEAR:.0f}m"] = groups["on_road"] & (distance <= NEAR)
        groups["5_or_fewer_points"] = seen[:, 8] <= 5
        groups["over_20_points"] = seen[:, 8] > 20
        kept = scores >= threshold
        result[CLASS_NAMES[kind]] = {
            "threshold": threshold, "seen": len(seen), "average_precision": average_precision(scores, is_true, len(seen)),
            "recall": {name: {"actors": int(group.sum()), "recall": float(hit[group].mean()) if group.any() else None}
                       for name, group in groups.items()},
            "precision": float(is_true[kept].mean()) if kept.any() else None,
            "false_boxes_per_sweep": float((~is_true[kept]).sum() / frames),
            "median_position_error_m": float(np.nanmedian(errors[hit])) if hit.any() else None,
            "p90_position_error_m": float(np.nanpercentile(errors[hit], 90)) if hit.any() else None,
        }
    return result


def seed_files(seed):
    files = sorted((RAW_DIR / f"seed_{seed}").glob("*.npz"), key=lambda path: int(path.stem))
    if not files:
        sys.exit(f"Nothing recorded for seed {seed}.")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--threshold", type=float, nargs=CLASSES, help="vehicle and pedestrian; chosen for the best F1 if left out")
    parser.add_argument("--output", help="also save the results to this JSON file")
    args = parser.parse_args()

    import onnxruntime

    session = onnxruntime.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    files = seed_files(args.seed)
    road = road_mask([load_sweep(path)[1] for path in sorted(RAW_DIR.glob("seed_*/*.npz"))])
    result = evaluate(session, files, args.threshold, road)
    text = json.dumps({"seed": args.seed, **result}, indent=1)
    print(text)
    if args.output:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
