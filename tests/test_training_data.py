"""The Waymo training pipeline. Needs no dataset; the model tests need PyTorch."""
import math

import numpy as np
import pytest
from convert_waymo import (
    LEFT,
    RIGHT,
    SIGNALS,
    STRAIGHT,
    classify_turn,
    rotate_to_agent_frame,
    scene_samples,
    to_agent_frame,
)
from waymo_predictor_node import history_to_ego


def test_agent_frame_is_x_forward_y_left():
    # Facing world +y in Waymo's right-handed frame, world -x is to the vehicle's left
    ahead_and_left = np.array([[0.0, 5.0], [-3.0, 0.0]])
    assert to_agent_frame(ahead_and_left, 0, 0, math.pi / 2) == pytest.approx(np.array([[5.0, 0.0], [0.0, 3.0]]))


def test_waymo_and_carla_conversions_agree_on_the_same_motion():
    """The model trains on one and runs on the other, so a mismatch here would mirror every turn.
    The same car in each world: CARLA's frame is Waymo's with y negated, and yaw negated with it."""
    waymo_yaw = math.radians(30.0)
    waymo = np.array([[float(i), 0.5 * i, 2.0, 1.0] for i in range(10)])
    carla = waymo * np.array([1.0, -1.0, 1.0, -1.0])

    from_waymo = np.concatenate((to_agent_frame(waymo[:, :2], waymo[-1, 0], waymo[-1, 1], waymo_yaw),
                                 rotate_to_agent_frame(waymo[:, 2:], waymo_yaw)), axis=1)
    from_carla = history_to_ego([tuple(row) for row in carla], -math.degrees(waymo_yaw))
    assert from_waymo == pytest.approx(from_carla, abs=1e-4)


def arc(degrees, steps=30, length=30.0):
    """A target curving smoothly through a heading change; positive is left."""
    points, x, y = [], 0.0, 0.0
    for i in range(1, steps + 1):
        heading = math.radians(degrees * i / steps)
        x += length / steps * math.cos(heading)
        y += length / steps * math.sin(heading)
        points.append((x, y))
    return np.array(points)


@pytest.mark.parametrize("degrees,expected", [(90, LEFT), (-90, RIGHT), (0, STRAIGHT), (10, STRAIGHT)])
def test_classify_turn(degrees, expected):
    assert classify_turn(arc(degrees)) == expected


def test_a_short_shuffle_is_not_a_turn():
    assert classify_turn(arc(90, length=2.0)) == STRAIGHT


def empty_scene(agents=3):
    def section(frames):
        return {k: np.zeros((agents, frames), dtype=np.float32) for k in SIGNALS}

    past, current, future = section(10), section(1), section(80)
    valid = {"past": np.ones((agents, 10), bool), "current": np.ones((agents, 1), bool),
             "future": np.ones((agents, 80), bool)}
    return past, current, future, valid, np.ones(agents, dtype=np.float32)


def drive(past, current, future, agent, speed=10.0):
    """Puts an agent on a straight drive along world +x, passing the origin on the current frame."""
    past["x"][agent] = speed * 0.1 * np.arange(-10, 0)
    past["velocity_x"][agent] = speed
    current["velocity_x"][agent] = speed
    future["x"][agent] = speed * 0.1 * np.arange(1, 81)
    future["velocity_x"][agent] = speed


def samples_from(past, current, future, valid, types, seed=0):
    return scene_samples(past, current, future, valid["past"], valid["current"], valid["future"], types,
                         np.random.default_rng(seed))


def test_scene_samples_shape_and_content():
    past, current, future, valid, types = empty_scene(1)
    drive(past, current, future, 0)
    inputs, targets, turns = samples_from(past, current, future, valid, types)
    assert np.asarray(inputs).shape == (1, 10, 4)
    assert np.asarray(targets).shape == (1, 30, 2)
    assert inputs[0][-1] == pytest.approx(np.array([0.0, 0.0, 10.0, 0.0]))  # current frame at the origin
    assert inputs[0][0, 0] == pytest.approx(-9.0)                           # 0.9 s ago
    assert targets[0][0] == pytest.approx(np.array([1.0, 0.0]))
    assert targets[0][-1] == pytest.approx(np.array([30.0, 0.0]))
    assert turns == [STRAIGHT]


def test_scene_samples_skip_non_vehicles_and_gaps():
    past, current, future, valid, types = empty_scene(3)
    for agent in range(3):
        drive(past, current, future, agent)
    types[1] = 2.0                   # a pedestrian
    valid["future"][2, 10] = False   # a gap inside the 3 s we need
    inputs, _, _ = samples_from(past, current, future, valid, types)
    assert len(inputs) == 1


def test_a_gap_beyond_the_three_seconds_is_fine():
    past, current, future, valid, types = empty_scene(1)
    drive(past, current, future, 0)
    valid["future"][0, 50] = False
    assert len(samples_from(past, current, future, valid, types)[0]) == 1


def test_most_parked_cars_are_dropped_but_some_stay():
    past, current, future, valid, types = empty_scene(128)
    kept = len(samples_from(past, current, future, valid, types)[0])
    assert 0 < kept < 30


# --- Model ---
def test_untrained_model_is_exactly_constant_velocity():
    torch = pytest.importorskip("torch")
    from train_trajectory import TrajectoryModel, constant_velocity_baseline

    history = torch.randn(5, 10, 4) * 10
    with torch.no_grad():
        assert torch.allclose(TrajectoryModel()(history), constant_velocity_baseline(history))


def test_mirroring_flips_left_and_right():
    torch = pytest.importorskip("torch")
    from train_trajectory import mirror

    inputs, targets = mirror(torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]), torch.tensor([[[5.0, 6.0]]]))
    assert inputs.tolist() == [[[1.0, -2.0, 3.0, -4.0]]]
    assert targets.tolist() == [[[5.0, -6.0]]]


def test_a_small_model_can_learn_to_turn():
    """If the frames or the correction were wired wrongly, this couldn't get below constant velocity."""
    torch = pytest.importorskip("torch")
    from train_trajectory import TrajectoryModel, constant_velocity_baseline, errors

    torch.manual_seed(0)
    turn_rate = torch.linspace(-0.4, 0.4, 64)
    times = 0.1 * torch.arange(-9, 31, dtype=torch.float32)
    heading = turn_rate[:, None] * times[None]
    velocity = torch.stack((10 * torch.cos(heading), 10 * torch.sin(heading)), dim=-1)
    positions = torch.cumsum(velocity * 0.1, dim=1)
    positions = positions - positions[:, 9:10]
    inputs = torch.cat((positions[:, :10], velocity[:, :10]), dim=-1)
    targets = positions[:, 10:]

    model = TrajectoryModel(hidden=32, layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(300):
        loss = torch.nn.functional.smooth_l1_loss(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        model_error = errors(model(inputs), targets)[0].mean()
    baseline_error = errors(constant_velocity_baseline(inputs), targets)[0].mean()
    assert model_error < 0.5 * baseline_error
