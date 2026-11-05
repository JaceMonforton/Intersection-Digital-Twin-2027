"""
Trains the roadside detector on sweeps from training/record_detection.py, and exports it to ONNX.

It saves rsu_detector.candidate.onnx, which nothing loads by itself. Seeds 1 to 3 train it and seed 4 picks the best
epoch. Seed 5 isn't read here, so that training/eval_detection.py can give a fair final figure on it.

Usage (in .venv, which has torch):
    python training/train_detection.py
    python training/train_detection.py --epochs 1 --limit 40 --output data/detection/smoke.onnx
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from detection_features import (
    CLASSES,
    GRID_SIZE,
    OFFSET,
    OUTPUT_CHANNELS,
    SIZE,
    YAW,
    encode_labels,
    load_sweep,
    rasterize,
)

RAW_DIR = ROOT / "data" / "detection" / "raw"
MODEL_OUT = ROOT / "models" / "rsu_detector.candidate.onnx"
TRAIN_SEEDS = (1, 2, 3)
VALIDATION_SEED = 4
ONNX_OPSET = 17
FOCAL_ALPHA, FOCAL_BETA = 2.0, 4.0
SIZE_WEIGHT, YAW_WEIGHT = 1.0, 0.5
PEDESTRIAN_WEIGHT = 2.0  # heatmap loss; they're small, and there are few on the road, where they matter most


class Block(nn.Sequential):
    def __init__(self, inputs, outputs, stride=1):
        super().__init__(
            nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True))


class Detector(nn.Module):
    """A small U-Net from the grid to the output in scripts/detection_features.py, at half the grid's resolution."""

    def __init__(self, width=24):
        super().__init__()
        self.down2 = Block(2, width, stride=2)
        self.down4 = Block(width, 2 * width, stride=2)
        self.down8 = Block(2 * width, 4 * width, stride=2)
        self.down16 = Block(4 * width, 6 * width, stride=2)
        self.up8 = Block(10 * width, 4 * width)
        self.up4 = Block(6 * width, 2 * width)
        self.up2 = Block(3 * width, width)
        self.head = nn.Conv2d(width, OUTPUT_CHANNELS, 1)
        self.grow = nn.Upsample(scale_factor=2, mode="nearest")
        nn.init.constant_(self.head.bias[:CLASSES], -4.6)  # starts by finding next to nothing, or the loss swamps it

    def forward(self, grid):
        at2 = self.down2(grid)
        at4 = self.down4(at2)
        at8 = self.down8(at4)
        at16 = self.down16(at8)
        x = self.up8(torch.cat((self.grow(at16), at8), dim=1))
        x = self.up4(torch.cat((self.grow(x), at4), dim=1))
        x = self.up2(torch.cat((self.grow(x), at2), dim=1))
        return self.head(x)


class ForExport(nn.Module):
    """The trained model with its heatmaps as scores from 0 to 1, as the output is defined."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, grid):
        output = self.model(grid)
        return torch.cat((torch.sigmoid(output[:, :CLASSES]), output[:, CLASSES:]), dim=1)


def heat_loss(logits, target, weight):
    """CenterNet's focal loss, with the loss near a hidden actor weighted out, per seen actor."""
    score = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    centre = target >= 0.999
    loss = torch.where(centre, -((1 - score) ** FOCAL_ALPHA) * torch.log(score),
                       -((1 - target) ** FOCAL_BETA) * (score ** FOCAL_ALPHA) * torch.log(1 - score))
    per_class = (loss * weight).sum(dim=(0, 2, 3)) / centre.sum(dim=(0, 2, 3)).clamp(min=1)
    return per_class[0] + PEDESTRIAN_WEIGHT * per_class[1]


def regression_loss(output, target, centres):
    """L1 at the actor centres only, as (offset, size, yaw)."""
    count = centres.sum().clamp(min=1)
    centres = centres[:, None]

    def part(start, channels):
        return (torch.abs(output[:, start:start + channels] - target[:, start:start + channels]) * centres).sum() / count

    return part(OFFSET, 2), part(SIZE, 3), part(YAW, 2)


def total_loss(output, target, weight, centres):
    offset, size, yaw = regression_loss(output, target, centres)
    return heat_loss(output[:, :CLASSES], target[:, :CLASSES], weight) + offset + SIZE_WEIGHT * size + YAW_WEIGHT * yaw


def load_seeds(seeds, limit=None):
    """Every sweep's grid as uint8, which takes a quarter of the memory, and its labels."""
    grids, labels = [], []
    for seed in seeds:
        files = sorted((RAW_DIR / f"seed_{seed}").glob("*.npz"), key=lambda path: int(path.stem))
        if not files:
            sys.exit(f"Nothing recorded for seed {seed}. See training/record_detection.py.")
        for path in files[:limit]:
            points, rows = load_sweep(path)
            grids.append(np.round(rasterize(points) * 255).astype(np.uint8))
            labels.append(rows)
    return grids, labels


def batches(grids, labels, batch, device, order):
    for start in range(0, len(order), batch):
        chosen = order[start:start + batch]
        grid = torch.from_numpy(np.stack([grids[i] for i in chosen])).to(device).float() / 255.0
        encoded = [encode_labels(labels[i]) for i in chosen]
        yield (grid, *(torch.from_numpy(np.stack([e[part] for e in encoded])).to(device) for part in range(3)))


def run_epoch(model, grids, labels, batch, device, optimizer=None, scaler=None, schedule=None, rng=None):
    training = optimizer is not None
    model.train(training)
    order = rng.permutation(len(grids)) if training else np.arange(len(grids))
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for grid, target, weight, centres in batches(grids, labels, batch, device, order):
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = total_loss(model(grid), target, weight, centres)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                schedule.step()
            total += float(loss) * len(grid)
            count += len(grid)
    return total / max(count, 1)


def export(model, path):
    """Exports to ONNX and checks onnxruntime gives the same answer as PyTorch."""
    import onnxruntime

    wrapped = ForExport(model).to("cpu").eval()
    sample = torch.rand(1, 2, GRID_SIZE, GRID_SIZE)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapped, (sample,), str(path), input_names=["grid"], output_names=["output"], opset_version=ONNX_OPSET, dynamo=False)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    from_onnx = session.run(None, {"grid": sample.numpy()})[0]
    with torch.no_grad():
        from_torch = wrapped(sample).numpy()
    worst = float(np.abs(from_onnx - from_torch).max())
    if worst > 1e-3:
        raise RuntimeError(f"ONNX output differs from PyTorch by up to {worst}.")
    print(f"[TRAIN] Exported {path} ({path.stat().st_size // 1024} KB); matches PyTorch within {worst:.1e}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(MODEL_OUT))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--limit", type=int, help="sweeps per seed, for a quick check")
    args = parser.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] On {device}. Loading...", flush=True)
    train_grids, train_labels = load_seeds(TRAIN_SEEDS, args.limit)
    check_grids, check_labels = load_seeds((VALIDATION_SEED,), args.limit)
    print(f"[TRAIN] {len(train_grids)} sweeps from seeds {TRAIN_SEEDS}; {len(check_grids)} from seed {VALIDATION_SEED} to choose the best epoch.", flush=True)

    model = Detector().to(device)
    print(f"[TRAIN] {sum(p.numel() for p in model.parameters()):,} parameters.", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    steps = args.epochs * ((len(train_grids) + args.batch - 1) // args.batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.learning_rate, total_steps=steps)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best, best_state, history = float("inf"), None, []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss = run_epoch(model, train_grids, train_labels, args.batch, device, optimizer, scaler, schedule, rng)
        check_loss = run_epoch(model, check_grids, check_labels, args.batch, device)
        history.append({"epoch": epoch, "train": train_loss, "validation": check_loss})
        better = check_loss < best
        if better:
            best, best_state = check_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"[TRAIN] Epoch {epoch}/{args.epochs}: train {train_loss:.3f}, validation {check_loss:.3f}"
              f"{' (best)' if better else ''}, {time.time() - started:.0f} s", flush=True)

    model.load_state_dict(best_state)
    path = Path(args.output)
    export(model, path)
    path.with_suffix(".history.json").write_text(json.dumps(history, indent=1))
    print(f"[TRAIN] Done. Best validation loss {best:.3f}. Now measure it with training/eval_detection.py.", flush=True)


if __name__ == "__main__":
    main()
