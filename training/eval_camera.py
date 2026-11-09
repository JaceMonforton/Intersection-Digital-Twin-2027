"""
Measures the roadside camera detector on a recorded seed, both cameras together, as the processor runs it: each
camera's boxes placed on the road, then merged. Seed 28 chooses the thresholds; seed 29 is for the final figures only.
Scoring is training/eval_detection.py's: an actor counts as seen with at least MIN_PIXELS pixels in either camera,
and a box beside one that's hidden or barely shows is neither right nor wrong.

    python training/eval_camera.py models/rsu_camera_detector.candidate.onnx --seed 28
    python training/eval_camera.py models/rsu_camera_detector.candidate.onnx --seed 29 --threshold 0.4 0.3
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from camera_features import CLASSES, MIN_PIXELS, decode, merge_cameras, pad_image
from detection_features import LABEL_SIZE, MIN_POINTS
from eval_detection import (
    CLASS_NAMES,
    LOW_THRESHOLD,
    NEAR,
    average_precision,
    choose_threshold,
    match,
)

RAW_DIR = ROOT / "data" / "camera" / "raw"


def truth(parts):
    """Both cameras' labels as one set in eval_detection's label format, an actor once each, seen (MIN_POINTS in the
    points column) if it shows with MIN_PIXELS in either camera. The same actor has the same position in both."""
    actors = {}
    for labels in parts:
        for row in labels:
            key = (round(float(row[2]), 3), round(float(row[3]), 3), int(row[9]))
            actors[key] = max(actors.get(key, (0.0, row))[0], float(row[10])), row
    out = np.zeros((len(actors), LABEL_SIZE), np.float32)
    for i, (pixels, row) in enumerate(actors.values()):
        out[i, :8] = row[2:10]
        out[i, 8] = MIN_POINTS if pixels >= MIN_PIXELS else 0
    return out


def frames(seed):
    """{frame: [path for camera 0, path for camera 1]} for a seed, with its meta."""
    folder = RAW_DIR / f"seed_{seed}"
    by_frame = defaultdict(dict)
    for path in folder.glob("*.npz"):
        frame, camera = path.stem.split("_")
        by_frame[int(frame)][int(camera)] = path
    meta = json.loads((folder / "meta.json").read_text())
    count = len(meta["cameras"])
    whole = {f: [c[i] for i in range(count)] for f, c in sorted(by_frame.items()) if len(c) == count}
    if not whole:
        sys.exit(f"Nothing recorded for seed {seed}.")
    return whole, meta


def evaluate(session, seed, thresholds=None):
    name = session.get_inputs()[0].name
    files, meta = frames(seed)
    to_world = [np.array(c["camera_to_world"]) for c in meta["cameras"]]
    center = meta["center"]
    collected = {kind: {"scores": [], "true": [], "seen": [], "found": [], "errors": []} for kind in range(CLASSES)}
    seconds = []
    for paths in files.values():
        loaded = [np.load(p) for p in paths]
        started = time.perf_counter()
        outputs = session.run(None, {name: np.stack([pad_image(d["image"]) for d in loaded])})[0]
        detections = merge_cameras([decode(o, LOW_THRESHOLD, m, center) for o, m in zip(outputs, to_world)])
        seconds.append(time.perf_counter() - started)
        labels = truth([d["labels"] for d in loaded])
        for kind, into in collected.items():
            for key, value in zip(("scores", "true", "seen", "found", "errors"), match(detections, labels, kind)):
                into[key].append(value)

    result = {"seed": seed, "frames": len(files), "milliseconds_median_both_cameras": float(np.median(seconds[1:]) * 1000)}
    for kind, into in collected.items():
        scores, is_true = np.concatenate(into["scores"]), np.concatenate(into["true"])
        seen, found, errors = np.vstack(into["seen"]), np.concatenate(into["found"]), np.concatenate(into["errors"])
        threshold = thresholds[kind] if thresholds else choose_threshold(scores, is_true, len(seen))
        hit, kept = found >= threshold, scores >= threshold
        distance = np.hypot(seen[:, 0], seen[:, 1])
        groups = {"all": np.ones(len(seen), bool), f"within_{NEAR:.0f}m": distance <= NEAR, f"beyond_{NEAR:.0f}m": distance > NEAR}
        result[CLASS_NAMES[kind]] = {
            "threshold": threshold, "seen": len(seen), "average_precision": average_precision(scores, is_true, len(seen)),
            "recall": {g: {"actors": int(m.sum()), "recall": float(hit[m].mean()) if m.any() else None} for g, m in groups.items()},
            "precision": float(is_true[kept].mean()) if kept.any() else None,
            "false_boxes_per_frame": float((~is_true[kept]).sum() / len(files)),
            "median_position_error_m": {g: float(np.nanmedian(errors[hit & m])) if (hit & m).any() else None for g, m in groups.items()},
            "p90_position_error_m": float(np.nanpercentile(errors[hit], 90)) if hit.any() else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--threshold", type=float, nargs=CLASSES, help="vehicle and pedestrian; chosen for the best F1 if left out")
    parser.add_argument("--output", help="also save the results to this JSON file")
    args = parser.parse_args()

    import onnxruntime

    session = onnxruntime.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    text = json.dumps(evaluate(session, args.seed, args.threshold), indent=1)
    print(text)
    if args.output:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
