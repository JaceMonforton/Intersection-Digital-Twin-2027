"""Unit tests for measuring and training the roadside detector. Training itself needs torch, in .venv."""
import numpy as np
import pytest
from detection_features import (
    GRID_SIZE,
    OUTPUT_CHANNELS,
    OUTPUT_SIZE,
    PEDESTRIAN,
    VEHICLE,
    label_row,
)
from eval_detection import (
    average_precision,
    choose_threshold,
    match,
    on_road,
    road_mask,
)


def actor(x, y, kind=VEHICLE, points=40):
    row = label_row((x, -y, 0.7), (2.0, 1.0, 0.7), 0.0, (0.0, 0.0), kind)
    row[8] = points
    return row


def box(x, y, score, kind=VEHICLE):
    return [x, y, -1.1, 4.0, 2.0, 1.4, 0.0, score, kind]


def arrays(rows, width):
    return np.array(rows, np.float32).reshape(-1, width)


def test_a_box_near_an_actor_is_true_and_a_box_elsewhere_is_false():
    _, is_true, seen, found, errors = match(arrays([box(1.0, 0.0, 0.9), box(30.0, 0.0, 0.8)], 9), arrays([actor(0.0, 0.0)], 11), VEHICLE)
    assert is_true.tolist() == [True, False]
    assert found.tolist() == pytest.approx([0.9])
    assert errors.tolist() == pytest.approx([1.0])
    assert len(seen) == 1


def test_an_actor_can_only_be_found_once():
    _, is_true, _, found, _ = match(arrays([box(0.0, 0.0, 0.9), box(0.5, 0.0, 0.8)], 9), arrays([actor(0.0, 0.0)], 11), VEHICLE)
    assert is_true.tolist() == [True, False]
    assert found.tolist() == pytest.approx([0.9])


def test_a_pedestrian_has_to_be_found_closer_than_a_car():
    labels = arrays([actor(0.0, 0.0, PEDESTRIAN)], 11)
    assert match(arrays([box(1.5, 0.0, 0.9, PEDESTRIAN)], 9), labels, PEDESTRIAN)[1].tolist() == [False]
    assert match(arrays([box(0.5, 0.0, 0.9, PEDESTRIAN)], 9), labels, PEDESTRIAN)[1].tolist() == [True]


def test_a_box_on_a_hidden_actor_is_neither_right_nor_wrong():
    scores, _, seen, found, _ = match(arrays([box(0.0, 0.0, 0.9)], 9), arrays([actor(0.0, 0.0, points=1)], 11), VEHICLE)
    assert len(scores) == 0
    assert len(seen) == 0
    assert len(found) == 0


def test_classes_are_kept_apart():
    scores, _, seen, _, _ = match(arrays([box(0.0, 0.0, 0.9, PEDESTRIAN)], 9), arrays([actor(0.0, 0.0)], 11), VEHICLE)
    assert len(scores) == 0
    assert len(seen) == 1


def test_average_precision_of_perfect_and_of_half_right():
    assert average_precision(np.array([0.9, 0.8]), np.array([True, True]), 2) == pytest.approx(1.0)
    assert average_precision(np.array([0.9, 0.8]), np.array([True, False]), 2) == pytest.approx(0.5)
    assert average_precision(np.array([0.9, 0.8]), np.array([False, True]), 1) == pytest.approx(0.5)
    assert average_precision(np.zeros(0), np.zeros(0, bool), 2) == 0.0


def test_the_threshold_with_the_best_balance_is_chosen():
    scores = np.array([0.9, 0.8, 0.7, 0.2])
    assert choose_threshold(scores, np.array([True, True, True, False]), 3) == pytest.approx(0.7)


def test_road_is_near_where_vehicles_have_been():
    mask = road_mask([arrays([actor(10.0, 10.0), actor(-40.0, 5.0, PEDESTRIAN)], 11)])
    points = np.array([[10.0, 10.0], [11.5, 10.0], [20.0, 10.0], [-40.0, 5.0]], np.float32)
    assert on_road(mask, points).tolist() == [True, True, False, False]


def test_the_model_makes_the_output_the_features_expect_and_can_learn():
    torch = pytest.importorskip("torch")
    from detection_features import encode_labels
    from train_detection import Detector, ForExport, total_loss

    torch.manual_seed(0)
    model = Detector(width=4)
    grid = torch.rand(2, 2, GRID_SIZE, GRID_SIZE)
    target, weight, centres = (torch.from_numpy(np.stack([part, part])) for part in encode_labels(arrays([actor(5.0, 5.0)], 11)))
    output = model(grid)
    assert output.shape == (2, OUTPUT_CHANNELS, OUTPUT_SIZE, OUTPUT_SIZE)
    scores = ForExport(model).eval()(grid)
    assert 0.0 <= float(scores[:, :2].min()) and float(scores[:, :2].max()) <= 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    first = float(total_loss(model(grid), target, weight, centres))
    for _ in range(5):
        optimizer.zero_grad()
        loss = total_loss(model(grid), target, weight, centres)
        loss.backward()
        optimizer.step()
    assert float(total_loss(model(grid), target, weight, centres)) < first
