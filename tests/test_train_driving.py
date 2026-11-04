"""Training the driving model. Needs torch, so it is skipped outside .venv. No CARLA needed."""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from driving_features import INPUT_SCALE, INPUT_SIZE, OUTPUT_SIZE
from train_driving import (
    CAR_BRAKE_WEIGHT,
    HOLD_WEIGHT,
    LANE_END_Y,
    LEFT_WEIGHT,
    RED_BRAKE_WEIGHT,
    RIGHT_WEIGHT,
    WALKER_BRAKE_WEIGHT,
    DrivingModel,
    export,
    report,
    sample_weights,
    situations,
    split_files,
    with_correction_share,
)
from waymo_predictor_node import LEFT, RIGHT, STRAIGHT


def samples(count=4, speed=5.0, target=5.0):
    inputs = np.zeros((count, INPUT_SIZE), dtype=np.float32)
    inputs[:, 0] = speed
    targets = np.tile(np.array([0.0, target], dtype=np.float32), (count, 1))
    return inputs, targets, np.full(count, STRAIGHT)


def test_whole_files_are_held_out_and_new_ones_move_nothing():
    names = [Path(f"{i:02d}.npz") for i in range(13)]
    train, validation = split_files(reversed(names))
    assert not set(train) & set(validation)
    assert len(train) + len(validation) == 13
    assert validation == [Path("04.npz"), Path("10.npz")]
    assert split_files([*names, Path("99.npz")])[1] == validation


def test_turns_are_drawn_more_and_standing_still_less():
    inputs, targets, routes = samples(4)
    routes[0], routes[1] = LEFT, RIGHT
    inputs[3, 0] = 0.0
    targets[3, 1] = 0.0
    assert sample_weights(inputs, targets, routes) == pytest.approx([LEFT_WEIGHT, RIGHT_WEIGHT, 1.0, HOLD_WEIGHT])


def test_the_road_curving_counts_as_a_turn_without_a_route_choice():
    inputs, targets, routes = samples(2)
    inputs[0, LANE_END_Y] = 8.0    # y points left in the ego frame
    inputs[1, LANE_END_Y] = -8.0
    assert sample_weights(inputs, targets, routes) == pytest.approx([LEFT_WEIGHT, RIGHT_WEIGHT])


def test_pulling_away_from_a_stop_keeps_its_full_weight():
    inputs, targets, routes = samples(1, speed=0.0, target=2.0)
    assert sample_weights(inputs, targets, routes) == pytest.approx([1.0])
    assert situations(inputs, targets, routes)["pulling away"].all()


def test_braking_on_the_way_to_a_red_line_is_drawn_more():
    inputs, targets, routes = samples(2, target=2.0)
    inputs[0, 4], inputs[0, 5] = 1.0, 12.0
    assert sample_weights(inputs, targets, routes) == pytest.approx([RED_BRAKE_WEIGHT, 1.0])


def test_braking_behind_a_car_or_for_a_pedestrian_is_drawn_more():
    inputs, targets, routes = samples(4, target=2.0)
    inputs[0, 28:33] = (1.0, 10.0, 0.0, 0.0, 0.0)   # a car 10 m ahead
    inputs[1, 58:63] = (1.0, 6.0, 1.0, 0.0, 0.0)    # a pedestrian 6 m ahead
    inputs[2, 28:33] = (1.0, 10.0, 5.0, 0.0, 0.0)   # a car ahead, but in the next lane
    assert sample_weights(inputs, targets, routes) == pytest.approx([CAR_BRAKE_WEIGHT, WALKER_BRAKE_WEIGHT, 1.0, 1.0])


def test_driving_on_with_a_car_ahead_keeps_its_normal_weight():
    inputs, targets, routes = samples(1, target=5.0)
    inputs[0, 28:33] = (1.0, 10.0, 0.0, 0.0, 0.0)
    assert sample_weights(inputs, targets, routes) == pytest.approx([1.0])


def test_takeovers_make_up_their_share_of_what_is_drawn():
    weights = np.array([1.0, 1.0, 2.0, 1.0], dtype=np.float32)
    mine = np.array([False, False, False, True])
    scaled = with_correction_share(weights, mine, share=0.2)
    assert scaled[mine].sum() / scaled.sum() == pytest.approx(0.2)
    assert scaled[~mine] == pytest.approx(weights[~mine])
    assert with_correction_share(weights, np.zeros(4, bool)) is weights


def test_report_compares_with_doing_nothing():
    inputs, targets, routes = samples(2, speed=5.0, target=3.0)
    targets[:, 0] = 0.2
    predicted = np.array([[0.1, 3.0], [0.3, 7.0]], dtype=np.float32)
    row = report(predicted, inputs, targets, routes)["all"]
    assert row["steer_error"] == pytest.approx(0.1)
    assert row["steer_error_baseline"] == pytest.approx(0.2)
    assert row["speed_error"] == pytest.approx(2.0)
    assert row["speed_error_baseline"] == pytest.approx(2.0)  # keeping 5 m/s when it went to 3
    assert row["slowing_right"] == pytest.approx(0.5)


def test_scaling_is_inside_the_model():
    model = DrivingModel()
    assert model.scale.numpy() == pytest.approx(INPUT_SCALE)
    assert model(torch.zeros(3, INPUT_SIZE)).shape == (3, OUTPUT_SIZE)


def test_target_speed_comes_out_in_metres_per_second():
    model = DrivingModel()
    with torch.no_grad():
        for layer in model.net:
            if hasattr(layer, "weight"):
                layer.weight.zero_()
                layer.bias.zero_()
        model.net[-1].bias.copy_(torch.tensor([0.1, 1.0]))
        output = model(torch.zeros(1, INPUT_SIZE))
    assert output[0, 0].item() == pytest.approx(0.1)
    assert output[0, 1].item() == pytest.approx(7.0)


def test_export_runs_any_batch_size_the_same_as_torch(tmp_path):
    pytest.importorskip("onnxruntime")
    import onnxruntime

    path = tmp_path / "policy.onnx"
    export(DrivingModel(), path)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert session.run(None, {"inputs": np.zeros((1, INPUT_SIZE), dtype=np.float32)})[0].shape == (1, OUTPUT_SIZE)
