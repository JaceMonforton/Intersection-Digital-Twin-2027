"""
Teaches the driving model left turns across traffic, without touching how it drives anywhere else.

Retraining the whole model on these lessons made its other driving worse. So the general driver from
train_driving.py stays frozen, and a small second network, trained only on left-route samples, adds a correction
to its output. It only acts when the route says left, so on any other route the output is exactly the general
driver's. Both parts are learned, and they export as one model that takes the inputs in models/README.md.

Usage (in .venv, which has torch):
    python training/train_left_turns.py
"""
import argparse
import sys
import time
from pathlib import Path

import onnx
import torch
from onnx import numpy_helper
from torch import nn
from train_driving import (
    DATA_DIR,
    MODEL_OUT,
    SPEED_LOSS_SCALE,
    correction_mask,
    export,
    load,
    loss_function,
    predict,
    print_report,
    report,
    sample_weights,
    split_files,
    with_correction_share,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from driving_features import INPUT_SCALE, INPUT_SIZE, OUTPUT_SIZE
from waymo_predictor_node import LEFT

GENERAL_MODEL = MODEL_OUT.with_name("driving_general.onnx")
ROUTE_LEFT = 1          # where the route's left flag sits in the input
YIELD_FILES = "yield_"  # the lessons from record_left_turns.py
YIELD_SHARE = 0.5       # of what's drawn


def general_layers(path):
    """The general model's Linear layers from its ONNX file, in order."""
    weights = {i.name: numpy_helper.to_array(i).copy() for i in onnx.load(str(path)).graph.initializer}
    layers = []
    for index in sorted({int(name.split(".")[1]) for name in weights if name.startswith("net.")}):
        weight, bias = torch.from_numpy(weights[f"net.{index}.weight"]), torch.from_numpy(weights[f"net.{index}.bias"])
        layer = nn.Linear(weight.shape[1], weight.shape[0])
        layer.weight.data, layer.bias.data = weight, bias
        layers.append(layer)
    return layers


class LeftTurnModel(nn.Module):
    """[N, INPUT_SIZE] -> [N, 2] of (steer, target speed in m/s): the frozen general driver, plus a correction on a left route."""

    def __init__(self, layers, hidden=128):
        super().__init__()
        self.register_buffer("scale", torch.from_numpy(INPUT_SCALE.copy()))
        self.general_inputs = layers[0].in_features  # it may predate the left-turn inputs, which come last
        parts = []
        for layer in layers:
            parts += [layer, nn.ReLU()]
        self.general = nn.Sequential(*parts[:-1])
        for parameter in self.general.parameters():
            parameter.requires_grad = False
        self.correction = nn.Sequential(nn.Linear(INPUT_SIZE, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, OUTPUT_SIZE))
        nn.init.zeros_(self.correction[-1].weight)  # starts as exactly the general driver
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, inputs):
        scaled = inputs / self.scale
        output = self.general(scaled[:, :self.general_inputs]) + self.correction(scaled) * scaled[:, ROUTE_LEFT:ROUTE_LEFT + 1]
        return torch.stack((output[:, 0], output[:, 1] * SPEED_LOSS_SCALE), dim=1)


def left_samples(paths):
    inputs, targets, routes = load(paths)
    keep = routes == LEFT
    return inputs[keep], targets[keep], routes[keep], correction_mask(paths, YIELD_FILES)[keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--general", default=str(GENERAL_MODEL))
    parser.add_argument("--output", default=str(MODEL_OUT))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_files, validation_files = split_files(Path(args.data).glob("*.npz"))
    inputs, targets, routes, is_lesson = left_samples(train_files)
    val_inputs, val_targets, val_routes, val_is_lesson = left_samples(validation_files)
    print(f"[TRAIN] {len(inputs)} left-route samples, {int(is_lesson.sum())} of them lessons; {len(val_inputs)} held out, {int(val_is_lesson.sum())} of them lessons.")

    def weights_for(x, y, r, lesson):
        return torch.from_numpy(with_correction_share(sample_weights(x, y, r), lesson, YIELD_SHARE)).to(device)

    train_in, train_out, draw = torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device), weights_for(inputs, targets, routes, is_lesson)
    val_in, val_out, val_draw = torch.from_numpy(val_inputs).to(device), torch.from_numpy(val_targets).to(device), weights_for(val_inputs, val_targets, val_routes, val_is_lesson)

    model = LeftTurnModel(general_layers(args.general)).to(device)
    optimizer = torch.optim.Adam(model.correction.parameters(), lr=args.learning_rate)
    steps = max(1, len(inputs) // args.batch)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * steps)
    best, best_state, best_epoch = float("inf"), None, 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(steps):
            picks = torch.multinomial(draw, args.batch, replacement=True)
            loss = loss_function(model(train_in[picks]), train_out[picks]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            total += float(loss)
        model.eval()
        with torch.no_grad():
            val_loss = float((loss_function(model(val_in), val_out) * val_draw).sum() / val_draw.sum())
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 25 == 0:
            print(f"[TRAIN] epoch {epoch:>3}/{args.epochs}  loss {total / steps:.4f}  val {val_loss:.4f}  best {best:.4f} @ {best_epoch}  ({time.time() - started:.0f} s)", flush=True)

    model.load_state_dict(best_state)
    print_report(f"Held-out left routes (best epoch {best_epoch}):", report(predict(model, val_inputs, device), val_inputs, val_targets, val_routes))
    export(model, Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
