"""
Trains the ML car's driving model on samples from record_driving.py and exports it for the driver.

This is behaviour cloning: the model learns to copy the steering and speed of CARLA's autopilot. Good
numbers here mean it copies well on driving like the recording. They do not show it can drive, because
its own small errors take it to places the recording never went. Only driving it in CARLA shows that.

Whole recording files are held out for validation, because neighbouring ticks are almost identical and
a random split would mostly test memory. Input scaling lives inside the model, so the exported ONNX
file takes metres and m/s directly, as models/README.md specifies.

Usage (in .venv, which has torch):
    python training/train_driving.py
    python training/train_driving.py --epochs 5 --no-export   # a quick trial
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from driving_features import (
    ACTOR_FEATURES,
    CROSSING_PRESENT,
    INPUT_SCALE,
    INPUT_SIZE,
    LANE_POINTS,
    MAX_CARS,
    MAX_WALKERS,
    OUTPUT_SIZE,
    STOP_GAP,
    STOP_PRESENT,
)
from waymo_predictor_node import LEFT, RIGHT

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "driving" / "converted"
MODEL_OUT = ROOT / "models" / "driving_policy.onnx"  # train_left_turns.py builds on what this writes, so pass --output models/driving_general.onnx

VALIDATION_EVERY = 6    # one recording file in this many is held out
LEFT_WEIGHT = 8.0       # left turns are the rarest thing in the recording
RIGHT_WEIGHT = 4.0
CORRECTION_SHARE = 0.2  # of what's drawn comes from autopilot's takeovers, which are few but are the point
CORRECTION_FILES = "corrections_"
LESSON_FILES = "yield_"  # left-turn lessons, for train_left_turns.py; here they only made other driving worse
HOLD_WEIGHT = 0.2       # standing still on the brake is 30% of the recording and one easy lesson
CAR_BRAKE_WEIGHT = 8.0  # the same for slowing behind a car, the model's most common crash
WALKER_BRAKE_WEIGHT = 16.0  # and for a pedestrian, which is rarer still
RED_BRAKE_WEIGHT = 8.0  # slowing for a line takes a moment, and running one is the worst thing it can learn
CURVE_METRES = 5.0      # the route 20 m ahead this far to one side counts as a turn
STOPPED_SPEED = 0.5     # m/s
MOVING_SPEED = 2.0      # m/s
SLOWING = 0.3           # m/s below its current speed that a target counts as slowing down
SPEED_LOSS_SCALE = 7.0  # m/s, about cruising speed; keeps the speed error near +/-1 in the loss
STEER_LOSS_WEIGHT = 5.0  # steering is mostly under 0.1 and speed spans 1 after scaling, so it would be drowned out
LOSS_BETA = 0.05        # below this the loss is quadratic; a wide one would ignore small steering errors
ONNX_OPSET = 17

SPEED = 0
LANE_END_Y = 6 + 2 * (LANE_POINTS + 1) - 1
CARS = LANE_END_Y + 1


class DrivingModel(nn.Module):
    """[N, 76] as models/README.md lays out -> [N, 2] of (steer, target speed in m/s)."""

    def __init__(self, hidden=256):
        super().__init__()
        self.register_buffer("scale", torch.from_numpy(INPUT_SCALE.copy()))
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, OUTPUT_SIZE),
        )

    def forward(self, inputs):
        output = self.net(inputs / self.scale)
        return torch.stack((output[:, 0], output[:, 1] * SPEED_LOSS_SCALE), dim=1)


def split_files(paths):
    """(training, validation) file lists. Sorted by name, so files added later never move earlier ones."""
    paths = sorted(paths)
    held_out = {i for i in range(len(paths)) if i % VALIDATION_EVERY == VALIDATION_EVERY - 2}
    return [p for i, p in enumerate(paths) if i not in held_out], [p for i, p in enumerate(paths) if i in held_out]


def correction_mask(paths, prefix=CORRECTION_FILES):
    """Which samples, in load order, come from recordings with this prefix: takeovers, by default."""
    return np.concatenate([np.full(len(np.load(path)["routes"]), Path(path).name.startswith(prefix)) for path in paths])


def load(paths):
    parts = [np.load(path) for path in paths]
    inputs = np.concatenate([p["inputs"] for p in parts])
    if inputs.shape[1:] != (INPUT_SIZE,):
        raise ValueError(f"Inputs have shape {inputs.shape[1:]}, not ({INPUT_SIZE},). Re-run record_driving.py --convert.")
    return inputs, np.concatenate([p["targets"] for p in parts]), np.concatenate([p["routes"] for p in parts])


def actor_close_ahead(inputs, start, count, reach, side):
    """Is a car or pedestrian within reach metres ahead and side metres to either side?"""
    slots = inputs[:, start:start + count * ACTOR_FEATURES].reshape(-1, count, ACTOR_FEATURES)
    return ((slots[:, :, 0] == 1.0) & (slots[:, :, 1] > 0.0) & (slots[:, :, 1] < reach) & (np.abs(slots[:, :, 2]) < side)).any(axis=1)


def car_close_ahead(inputs):
    return actor_close_ahead(inputs, CARS, MAX_CARS, 15.0, 2.0)


def walker_close_ahead(inputs):
    return actor_close_ahead(inputs, CARS + MAX_CARS * ACTOR_FEATURES, MAX_WALKERS, 12.0, 3.0)


def turning(inputs, routes):
    """(left, right) masks: a turn on the route, or the road itself curving ahead."""
    return (routes == LEFT) | (inputs[:, LANE_END_Y] > CURVE_METRES), (routes == RIGHT) | (inputs[:, LANE_END_Y] < -CURVE_METRES)


def holding_still(inputs, targets):
    return (inputs[:, SPEED] < STOPPED_SPEED) & (targets[:, 1] < STOPPED_SPEED)


def sample_weights(inputs, targets, routes):
    """How often to draw each sample: more turns, and less standing on the brake. Pulling away stays in full."""
    left, right = turning(inputs, routes)
    weights = np.ones(len(inputs), dtype=np.float32)
    weights[right] = RIGHT_WEIGHT
    weights[left] = LEFT_WEIGHT
    weights[holding_still(inputs, targets)] *= HOLD_WEIGHT
    slowing = (inputs[:, SPEED] >= STOPPED_SPEED) & (targets[:, 1] < inputs[:, SPEED] - SLOWING)
    weights[slowing & car_close_ahead(inputs)] = CAR_BRAKE_WEIGHT
    weights[slowing & (inputs[:, STOP_PRESENT] == 1.0)] = RED_BRAKE_WEIGHT
    weights[slowing & walker_close_ahead(inputs)] = WALKER_BRAKE_WEIGHT
    return weights


def with_correction_share(weights, is_correction, share=CORRECTION_SHARE):
    """Scales the takeover samples so that they make up this share of what's drawn."""
    if not is_correction.any():
        return weights
    weights = weights.copy()
    others = weights[~is_correction].sum()
    weights[is_correction] *= share / (1.0 - share) * others / weights[is_correction].sum()
    return weights


def situations(inputs, targets, routes):
    """Masks for the situations worth reporting one by one, since an overall average hides them."""
    speed = inputs[:, SPEED]
    moving, stopped = speed > MOVING_SPEED, speed < STOPPED_SPEED
    left, right = turning(inputs, routes)
    near_red = (inputs[:, STOP_PRESENT] == 1.0) & (inputs[:, STOP_GAP] < 8.0)
    return {
        "all": np.ones(len(inputs), dtype=bool),
        "straight road": moving & ~left & ~right,
        "left route": routes == LEFT,
        "car due, left": (routes == LEFT) & (inputs[:, CROSSING_PRESENT] == 1.0),
        "right route": routes == RIGHT,
        "curving left": moving & left,
        "curving right": moving & right,
        "at a red light": near_red,
        "nearing a red": moving & (inputs[:, STOP_PRESENT] == 1.0) & (inputs[:, STOP_GAP] < 15.0),
        "car close ahead": moving & car_close_ahead(inputs),
        "person close ahead": moving & walker_close_ahead(inputs),
        "holding still": holding_still(inputs, targets),
        "pulling away": stopped & ~holding_still(inputs, targets),
    }


def predict(model, inputs, device, batch=65536):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(inputs[i:i + batch]).to(device)).cpu().numpy()
                               for i in range(0, len(inputs), batch)])


def report(predicted, inputs, targets, routes):
    """Per situation: steering and target speed error, whether it rightly chooses to slow down or not, and
    the same errors for a do-nothing baseline (steer straight, keep the current speed)."""
    table = {}
    for name, mask in situations(inputs, targets, routes).items():
        if mask.any():
            p, t, speed = predicted[mask], targets[mask], inputs[mask, SPEED]
            table[name] = {
                "samples": int(mask.sum()),
                "steer_error": float(np.abs(p[:, 0] - t[:, 0]).mean()),
                "steer_error_baseline": float(np.abs(t[:, 0]).mean()),
                "speed_error": float(np.abs(p[:, 1] - t[:, 1]).mean()),
                "speed_error_baseline": float(np.abs(speed - t[:, 1]).mean()),
                "slowing_right": float(((p[:, 1] < speed - SLOWING) == (t[:, 1] < speed - SLOWING)).mean()),
            }
    return table


def print_report(title, table):
    print(f"[TRAIN] {title}")
    print(f"          {'':<16}{'samples':>8}  {'steer err (base)':>17}  {'speed err (base)':>17}  slowing or not right")
    for name, row in table.items():
        print(f"          {name:<16}{row['samples']:>8}  {row['steer_error']:>8.3f} ({row['steer_error_baseline']:.3f}) "
              f" {row['speed_error']:>9.3f} ({row['speed_error_baseline']:.3f})  {100 * row['slowing_right']:>5.1f}%")


def loss_function(predicted, targets):
    steer = nn.functional.smooth_l1_loss(predicted[:, 0], targets[:, 0], beta=LOSS_BETA, reduction="none")
    speed = nn.functional.smooth_l1_loss(predicted[:, 1] / SPEED_LOSS_SCALE, targets[:, 1] / SPEED_LOSS_SCALE, beta=LOSS_BETA, reduction="none")
    return STEER_LOSS_WEIGHT * steer + speed


def export(model, path):
    """Exports to ONNX and checks onnxruntime gives the same answer as PyTorch."""
    import onnxruntime

    model = model.to("cpu").eval()
    scale = torch.from_numpy(INPUT_SCALE.copy())
    path.parent.mkdir(exist_ok=True)
    torch.onnx.export(model, (torch.randn(4, INPUT_SIZE) * scale,), str(path), input_names=["inputs"], output_names=["controls"],
                      dynamic_axes={"inputs": {0: "N"}, "controls": {0: "N"}}, opset_version=ONNX_OPSET, dynamo=False)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    other_batch = (torch.randn(7, INPUT_SIZE) * scale).numpy()
    from_onnx = session.run(None, {"inputs": other_batch})[0]
    with torch.no_grad():
        from_torch = model(torch.from_numpy(other_batch)).numpy()
    worst = float(np.abs(from_onnx - from_torch).max())
    if worst > 1e-3:
        raise RuntimeError(f"ONNX output differs from PyTorch by up to {worst}.")
    print(f"[TRAIN] Exported {path} ({path.stat().st_size // 1024} KB); matches PyTorch within {worst:.1e}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--output", default=str(MODEL_OUT))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TRAIN] Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    train_files, validation_files = split_files(p for p in Path(args.data).glob("*.npz") if not p.name.startswith(LESSON_FILES))
    if not validation_files:
        raise SystemExit(f"[TRAIN] Too few recordings in {args.data}. Run record_driving.py first.")
    inputs, targets, routes = load(train_files)
    val_inputs, val_targets, val_routes = load(validation_files)
    print(f"[TRAIN] {len(inputs)} training samples from {len(train_files)} files, {len(val_inputs)} validation samples "
          f"from {[p.stem for p in validation_files]}.")

    is_correction = correction_mask(train_files)
    val_is_correction = correction_mask(validation_files)
    weights = with_correction_share(sample_weights(inputs, targets, routes), is_correction)
    print(f"[TRAIN] {int(is_correction.sum())} training samples and {int(val_is_correction.sum())} validation samples are takeovers.")
    left, right = turning(inputs, routes)
    total_weight = float(weights.sum())
    print(f"[TRAIN] Share of what's drawn: left {100 * weights[left].sum() / total_weight:.0f}%, "
          f"right {100 * weights[right & ~left].sum() / total_weight:.0f}%, "
          f"standing still {100 * weights[holding_still(inputs, targets)].sum() / total_weight:.0f}%.")

    train_in, train_out = torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)
    draw = torch.from_numpy(weights).to(device)
    val_in, val_out = torch.from_numpy(val_inputs).to(device), torch.from_numpy(val_targets).to(device)
    val_weights = torch.from_numpy(with_correction_share(sample_weights(val_inputs, val_targets, val_routes), val_is_correction)).to(device)

    model = DrivingModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    steps_per_epoch = max(1, len(inputs) // args.batch)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * steps_per_epoch)

    best, best_state, best_epoch = float("inf"), None, 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(steps_per_epoch):
            picks = torch.multinomial(draw, args.batch, replacement=True)
            loss = loss_function(model(train_in[picks]), train_out[picks]).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            schedule.step()
            total += float(loss)

        model.eval()
        with torch.no_grad():
            # Weighted like training, so that the epoch picked isn't just the best at straight roads
            val_loss = float((loss_function(model(val_in), val_out) * val_weights).sum() / val_weights.sum())
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"[TRAIN] epoch {epoch:>3}/{args.epochs}  loss {total / steps_per_epoch:.4f}  val {val_loss:.4f}  "
                  f"best {best:.4f} @ {best_epoch}  ({time.time() - started:.0f} s)", flush=True)

    model.load_state_dict(best_state)
    training_table = report(predict(model, inputs, device), inputs, targets, routes)
    validation_table = report(predict(model, val_inputs, device), val_inputs, val_targets, val_routes)
    print_report("Training files:", training_table)
    print_report(f"Held-out files (best epoch {best_epoch}):", validation_table)
    if val_is_correction.any():
        predicted = predict(model, val_inputs[val_is_correction], device)
        print_report("Held-out takeovers only:", report(predicted, val_inputs[val_is_correction], val_targets[val_is_correction],
                                                        val_routes[val_is_correction]))

    if not args.no_export:
        output = Path(args.output)
        export(model, output)
        summary = {"validation": validation_table, "training": training_table, "best_epoch": best_epoch, "epochs": args.epochs,
                   "training_files": [p.stem for p in train_files], "validation_files": [p.stem for p in validation_files]}
        output.with_suffix(".metrics.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
