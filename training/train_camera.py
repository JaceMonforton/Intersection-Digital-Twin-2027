"""
Trains the roadside camera detector on images from training/record_camera.py, and exports it to ONNX.

It saves models/rsu_camera_detector.candidate.onnx, which only the camera_and_rsu config loads. The seeds are split
as for the LiDAR detector: TRAIN_SEEDS train it, VALIDATION_SEED picks the best epoch (and, in eval_camera.py, the
score thresholds), and seed 29 isn't read here, so that eval_camera.py can give a fair final figure on it.

Usage (in .venv, which has torch):
    python training/train_camera.py
    python training/train_camera.py --epochs 1 --limit 40 --output data/camera/smoke.onnx
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
from camera_features import (
    CLASSES,
    IMAGE_WIDTH,
    INPUT_HEIGHT,
    OFFSET,
    OUTPUT_CHANNELS,
    SIZE,
    YAW,
    encode_labels,
)
from train_detection import ONNX_OPSET, Block, heat_loss

RAW_DIR = ROOT / "data" / "camera" / "raw"
MODEL_OUT = ROOT / "models" / "rsu_camera_detector.candidate.onnx"
TRAIN_SEEDS = (20, 21, 22, 23, 25, 26)
VALIDATION_SEED = 28
SIZE_WEIGHT, YAW_WEIGHT = 0.5, 0.5
BRIGHTNESS = 0.3        # images are scaled by up to this either way in training, as a real camera's exposure would vary


class CameraDetector(nn.Module):
    """A small encoder-decoder from the image to the output in scripts/camera_features.py, at a quarter of its
    resolution. It runs on the processor's CPU for two cameras every tick, so it's kept narrow."""

    def __init__(self, width=16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.down4 = Block(width, 2 * width, stride=2)
        self.down8 = Block(2 * width, 4 * width, stride=2)
        self.down16 = nn.Sequential(Block(4 * width, 8 * width, stride=2), Block(8 * width, 8 * width))
        self.up8 = Block(12 * width, 4 * width)
        self.up4 = Block(6 * width, 2 * width)
        self.head = nn.Sequential(nn.Conv2d(2 * width, 2 * width, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(2 * width, OUTPUT_CHANNELS, 1))
        self.grow = nn.Upsample(scale_factor=2, mode="nearest")
        nn.init.constant_(self.head[-1].bias[:CLASSES], -4.6)  # starts by finding next to nothing, or the loss swamps it

    def forward(self, image):
        at4 = self.down4(self.stem(image))
        at8 = self.down8(at4)
        at16 = self.down16(at8)
        x = self.up8(torch.cat((self.grow(at16), at8), dim=1))
        x = self.up4(torch.cat((self.grow(x), at4), dim=1))
        return self.head(x)


class ForExport(nn.Module):
    """The trained model with its heatmaps as scores from 0 to 1, as the output is defined."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        output = self.model(image)
        return torch.cat((torch.sigmoid(output[:, :CLASSES]), output[:, CLASSES:]), dim=1)


def total_loss(output, target, weight, centres):
    count = centres.sum().clamp(min=1)
    centres = centres[:, None]

    def part(start, channels):
        return (torch.abs(output[:, start:start + channels] - target[:, start:start + channels]) * centres).sum() / count

    return (heat_loss(output[:, :CLASSES], target[:, :CLASSES], weight) + part(OFFSET, 2)
            + YAW_WEIGHT * part(YAW, 2) + SIZE_WEIGHT * part(SIZE, 3))


def load_seeds(seeds, limit=None):
    """Every image as uint8, and its labels."""
    images, labels = [], []
    for seed in seeds:
        files = sorted((RAW_DIR / f"seed_{seed}").glob("*.npz"))
        if not files:
            sys.exit(f"Nothing recorded for seed {seed}. See training/record_camera.py.")
        for path in files[:limit]:
            with np.load(path) as data:
                images.append(data["image"])
                labels.append(data["labels"])
    return images, labels


def batches(images, labels, batch, device, order, rng=None):
    for start in range(0, len(order), batch):
        chosen = order[start:start + batch]
        image = torch.zeros((len(chosen), 3, INPUT_HEIGHT, IMAGE_WIDTH), device=device)
        raw = torch.from_numpy(np.stack([images[i] for i in chosen])).to(device).permute(0, 3, 1, 2).float() / 255.0
        if rng is not None:
            scale = torch.from_numpy(rng.uniform(1 - BRIGHTNESS, 1 + BRIGHTNESS, (len(chosen), 1, 1, 1))).to(device).float()
            raw = (raw * scale).clamp(0.0, 1.0)
        image[:, :, :raw.shape[2]] = raw
        encoded = [encode_labels(labels[i]) for i in chosen]
        yield (image, *(torch.from_numpy(np.stack([e[part] for e in encoded])).to(device) for part in range(3)))


def run_epoch(model, images, labels, batch, device, optimizer=None, scaler=None, schedule=None, rng=None):
    training = optimizer is not None
    model.train(training)
    order = rng.permutation(len(images)) if training else np.arange(len(images))
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for image, target, weight, centres in batches(images, labels, batch, device, order, rng if training else None):
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = total_loss(model(image), target, weight, centres)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                schedule.step()
            total += float(loss) * len(image)
            count += len(image)
    return total / max(count, 1)


def export(model, path):
    """Exports to ONNX, with the batch free so the processor runs both cameras at once, and checks onnxruntime gives
    the same answer as PyTorch."""
    import onnxruntime

    wrapped = ForExport(model).to("cpu").eval()
    sample = torch.rand(2, 3, INPUT_HEIGHT, IMAGE_WIDTH)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapped, (sample,), str(path), input_names=["image"], output_names=["output"], opset_version=ONNX_OPSET,
                      dynamic_axes={"image": {0: "batch"}, "output": {0: "batch"}}, dynamo=False)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    from_onnx = session.run(None, {"image": sample.numpy()})[0]
    with torch.no_grad():
        from_torch = wrapped(sample).numpy()
    worst = float(np.abs(from_onnx - from_torch).max())
    if worst > 1e-3:
        raise RuntimeError(f"ONNX output differs from PyTorch by up to {worst}.")
    print(f"[TRAIN] Exported {path} ({path.stat().st_size // 1024} KB); matches PyTorch within {worst:.1e}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(MODEL_OUT))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--limit", type=int, help="images per seed, for a quick check")
    args = parser.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] On {device}. Loading...", flush=True)
    train_images, train_labels = load_seeds(TRAIN_SEEDS, args.limit)
    check_images, check_labels = load_seeds((VALIDATION_SEED,), args.limit)
    print(f"[TRAIN] {len(train_images)} images from seeds {TRAIN_SEEDS}; {len(check_images)} from seed {VALIDATION_SEED} to choose the best epoch.", flush=True)

    model = CameraDetector().to(device)
    print(f"[TRAIN] {sum(p.numel() for p in model.parameters()):,} parameters.", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    steps = args.epochs * ((len(train_images) + args.batch - 1) // args.batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.learning_rate, total_steps=steps)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best, best_state, history = float("inf"), None, []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss = run_epoch(model, train_images, train_labels, args.batch, device, optimizer, scaler, schedule, rng)
        check_loss = run_epoch(model, check_images, check_labels, args.batch, device)
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
    print(f"[TRAIN] Done. Best validation loss {best:.3f}. Now measure it with training/eval_camera.py.", flush=True)


if __name__ == "__main__":
    main()
