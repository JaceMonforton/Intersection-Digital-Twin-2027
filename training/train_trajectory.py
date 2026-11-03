"""
Trains the trajectory model on samples from convert_waymo.py and exports it for the predictor node.

The model keeps a vehicle at constant velocity and learns a correction on top, so its starting point
is the baseline it has to beat. Input scaling lives inside the model, so the exported ONNX file
takes metres and m/s directly, as models/README.md specifies.

Usage:
    python training/train_trajectory.py
    python training/train_trajectory.py --epochs 5 --no-export   # a quick trial
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from convert_waymo import FUTURE, HISTORY, LEFT, RIGHT, STEP_SECONDS, STRAIGHT
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DATA = ROOT / "data" / "waymo" / "converted" / "training.npz"
VALIDATION_DATA = ROOT / "data" / "waymo" / "converted" / "validation.npz"
MODEL_OUT = ROOT / "models" / "waymo_trajectory.onnx"

POSITION_SCALE = 20.0  # metres; keeps inputs near +/-1
SPEED_SCALE = 10.0     # m/s
TURN_WEIGHT = 4.0      # turns are rare but are what this project cares about
ONNX_OPSET = 17


class TrajectoryModel(nn.Module):
    """[N, 10, 4] of (x, y, vx, vy) -> [N, 30, 2] of (x, y), all in the vehicle's own frame."""

    def __init__(self, hidden=128, layers=2):
        super().__init__()
        self.register_buffer("scale", torch.tensor([POSITION_SCALE, POSITION_SCALE, SPEED_SCALE, SPEED_SCALE]))
        times = STEP_SECONDS * torch.arange(1, FUTURE + 1, dtype=torch.float32)
        self.register_buffer("times", times.view(1, FUTURE, 1))
        self.gru = nn.GRU(4, hidden, layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, FUTURE * 2))
        nn.init.zeros_(self.head[-1].weight)  # start exactly at constant velocity
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, history):
        constant_velocity = history[:, -1:, 2:4] * self.times
        encoded, _ = self.gru(history / self.scale)
        correction = self.head(encoded[:, -1]).view(-1, FUTURE, 2)
        return constant_velocity + correction


def mirror(inputs, targets):
    """A left turn seen in a mirror is a right turn, so flipping y doubles the turning data."""
    sign_in = inputs.new_tensor([1.0, -1.0, 1.0, -1.0])
    sign_out = targets.new_tensor([1.0, -1.0])
    return inputs * sign_in, targets * sign_out


def errors(predicted, target):
    """(ADE, FDE) per sample: mean and final position error, in metres."""
    distance = torch.linalg.vector_norm(predicted - target, dim=-1)
    return distance.mean(dim=1), distance[:, -1]


def evaluate(forward, inputs, targets, turns, batch=8192):
    ade, fde = [], []
    with torch.no_grad():
        for start in range(0, len(inputs), batch):
            a, f = errors(forward(inputs[start:start + batch]), targets[start:start + batch])
            ade.append(a)
            fde.append(f)
    ade, fde = torch.cat(ade), torch.cat(fde)
    report = {"all": (float(ade.mean()), float(fde.mean()), len(ade))}
    for name, value in (("straight", STRAIGHT), ("left", LEFT), ("right", RIGHT)):
        mask = turns == value
        if mask.any():
            report[name] = (float(ade[mask].mean()), float(fde[mask].mean()), int(mask.sum()))
    return report


def constant_velocity_baseline(inputs):
    times = STEP_SECONDS * torch.arange(1, FUTURE + 1, device=inputs.device, dtype=inputs.dtype)
    return inputs[:, -1:, 2:4] * times.view(1, FUTURE, 1)


def load(path, device):
    data = np.load(path)
    if data["inputs"].shape[1:] != (HISTORY, 4) or data["targets"].shape[1:] != (FUTURE, 2):
        raise ValueError(f"{path} has unexpected shapes. Re-run convert_waymo.py.")
    return (torch.from_numpy(data["inputs"]).to(device), torch.from_numpy(data["targets"]).to(device),
            torch.from_numpy(data["turns"]).to(device))


def print_report(title, report):
    print(f"[TRAIN] {title}")
    for name, (ade, fde, count) in report.items():
        print(f"          {name:<9} ADE {ade:6.3f} m   FDE {fde:6.3f} m   ({count} samples)")


def export(model, path):
    """Exports to ONNX and checks onnxruntime gives the same answer as PyTorch."""
    import onnxruntime

    model = model.to("cpu").eval()
    sample = torch.randn(4, HISTORY, 4) * model.scale
    path.parent.mkdir(exist_ok=True)
    torch.onnx.export(model, (sample,), str(path), input_names=["history"], output_names=["future"],
                      dynamic_axes={"history": {0: "N"}, "future": {0: "N"}}, opset_version=ONNX_OPSET,
                      dynamo=False)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    other_batch = (torch.randn(7, HISTORY, 4) * model.scale).numpy()
    from_onnx = session.run(None, {"history": other_batch})[0]
    with torch.no_grad():
        from_torch = model(torch.from_numpy(other_batch)).numpy()
    worst = float(np.abs(from_onnx - from_torch).max())
    if worst > 1e-3:
        raise RuntimeError(f"ONNX output differs from PyTorch by up to {worst} m.")
    print(f"[TRAIN] Exported {path} ({path.stat().st_size // 1024} KB); matches PyTorch within {worst:.1e} m.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default=str(TRAIN_DATA))
    parser.add_argument("--validation", default=str(VALIDATION_DATA))
    parser.add_argument("--output", default=str(MODEL_OUT))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TRAIN] Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    inputs, targets, turns = load(args.train, device)
    val_inputs, val_targets, val_turns = load(args.validation, device)
    print(f"[TRAIN] {len(inputs)} training samples, {len(val_inputs)} validation samples.")

    baseline = evaluate(constant_velocity_baseline, val_inputs, val_targets, val_turns)
    print_report("Constant velocity baseline, validation:", baseline)

    model = TrajectoryModel().to(device)
    weights = torch.ones(len(inputs), device=device)
    weights[turns != STRAIGHT] = TURN_WEIGHT
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    steps_per_epoch = max(1, len(inputs) // args.batch)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * steps_per_epoch)
    loss_function = nn.SmoothL1Loss()

    best, best_state, best_epoch = float("inf"), None, 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(steps_per_epoch):
            picks = torch.multinomial(weights, args.batch, replacement=True)
            batch_in, batch_out = inputs[picks], targets[picks]
            flip = torch.rand(len(picks), device=device) < 0.5
            mirrored_in, mirrored_out = mirror(batch_in, batch_out)
            batch_in = torch.where(flip.view(-1, 1, 1), mirrored_in, batch_in)
            batch_out = torch.where(flip.view(-1, 1, 1), mirrored_out, batch_out)
            loss = loss_function(model(batch_in), batch_out)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            schedule.step()
            total += float(loss)

        model.eval()
        report = evaluate(model, val_inputs, val_targets, val_turns)
        if report["all"][0] < best:
            best, best_epoch = report["all"][0], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[TRAIN] epoch {epoch:>3}/{args.epochs}  loss {total / steps_per_epoch:.4f}  val ADE {report['all'][0]:.3f} m  "
              f"FDE {report['all'][1]:.3f} m  ({time.time() - started:.0f} s)", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    final = evaluate(model.to(device), val_inputs, val_targets, val_turns)
    print_report(f"Trained model, validation (best epoch {best_epoch}):", final)

    if not args.no_export:
        output = Path(args.output)
        export(model, output)
        summary = {"baseline": baseline, "model": final, "best_epoch": best_epoch, "epochs": args.epochs,
                   "training_samples": len(inputs), "columns": ["ADE m", "FDE m", "samples"]}
        output.with_suffix(".metrics.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
