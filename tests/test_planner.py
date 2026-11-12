"""The planner's choices, in CARLA's frame (y right). No CARLA needed."""
import math

import numpy as np
import pytest
from planner import (
    BLIND_APPROACH,
    Planner,
    along_route,
    hypotheses,
    predicted_path,
    roll_out,
    score,
    times,
)

JUNCTION = (0.0, 0.0)
ROUTE_EAST = [(float(x), 0.0) for x in range(60)]  # straight on along +x


def far_planner(**kw):
    return Planner((500.0, 500.0), **kw)  # a junction far away, so nothing junction-specific applies


def test_an_empty_road_keeps_the_models_speed():
    plan = far_planner().plan((0.0, 0.0, 5.0, 0.0, 0.0), ROUTE_EAST, [], [], 5.5)
    assert plan.option == "model" and not plan.changed and plan.speed == 5.5


def test_a_person_walking_onto_the_route_ahead_makes_it_slow_or_stop():
    person = (14.0, 4.0, 0.0, -1.5)  # 14 m ahead, coming from the right side onto the road
    plan = far_planner().plan((0.0, 0.0, 5.0, 0.0, 0.0), ROUTE_EAST, [], [person], 5.5)
    assert plan.changed and plan.speed < 4.0 and plan.option in ("ease", "stop", "hard_stop")


def test_a_car_stopped_in_the_lane_ahead_is_stopped_for():
    parked = (12.0, 0.0, 0.0, 0.0, 0.0)
    plan = far_planner().plan((0.0, 0.0, 5.0, 0.0, 0.0), ROUTE_EAST, [parked], [], 5.5)
    assert plan.speed < 3.0


def test_in_the_junction_it_clears_rather_than_stop_in_a_crossing_cars_path():
    crossing = (0.0, -12.0, 0.0, 8.0, 90.0)  # reaches where the ML car is in 1.5 s
    plan = Planner(JUNCTION).plan((0.0, 0.0, 3.0, 0.0, 0.0), ROUTE_EAST, [crossing], [], 0.5)  # the model wants to stop
    assert plan.option == "clear" and plan.speed > 3.0


def test_in_the_junction_it_doesnt_override_the_model_waiting_to_give_way():
    far_off = (0.0, -60.0, 0.0, 8.0, 90.0)  # coming, but nowhere near within the horizon
    plan = Planner(JUNCTION).plan((0.0, 0.0, 0.5, 0.0, 0.0), ROUTE_EAST, [far_off], [], 0.0)
    assert plan.option != "clear" and plan.speed <= 0.5


def test_outside_the_junction_clearing_isnt_an_option():
    crossing = (40.0, -12.0, 0.0, 8.0, 90.0)
    plan = Planner(JUNCTION).plan((40.0, 0.0, 3.0, 0.0, 0.0), [(40.0 + x, 0.0) for x in range(60)], [crossing], [], 0.5)
    assert plan.option != "clear"


def test_with_no_roadside_unit_it_comes_into_the_junction_slowly():
    route = [(-25.0 + x, 0.0) for x in range(60)]
    blind = Planner(JUNCTION, has_roadside=False).plan((-25.0, 0.0, 5.5, 0.0, 0.0), route, [], [], 6.0)
    seen = Planner(JUNCTION, has_roadside=True).plan((-25.0, 0.0, 5.5, 0.0, 0.0), route, [], [], 6.0)
    assert blind.speed == BLIND_APPROACH and blind.option == "blind_approach"
    assert seen.speed == 6.0


def test_a_left_turn_in_carlas_frame_bends_to_negative_y_when_facing_plus_x():
    path, yaws = predicted_path(0.0, 0.0, 5.0, 0.0, -0.5, np.array([1.0, 2.0]))
    assert (path[:, 1] < 0).all() and yaws[-1] < 0
    straight, _ = predicted_path(0.0, 0.0, 5.0, 0.0, 0.0, np.array([2.0]))
    assert straight[0] == pytest.approx((10.0, 0.0))


def test_cars_near_the_junction_may_turn_and_people_by_the_road_may_step_out():
    route = np.array(ROUTE_EAST)
    near = hypotheses([(10.0, 5.0, -6.0, 0.0, 180.0)], [(20.0, 3.0, 0.0, 0.0)], JUNCTION, route)
    kinds = sorted(h.kind for h in near)
    assert kinds == ["standing", "step_out", "straight", "turn_left", "turn_right"]
    far = hypotheses([(300.0, 5.0, -6.0, 0.0, 180.0)], [], JUNCTION, route)
    assert [h.kind for h in far] == ["straight"]
    assert all(len(h.path) == len(times()) for h in near)
    step = next(h for h in near if h.kind == "step_out")
    assert math.hypot(*(step.path[5] - step.path[0])) > 0.5  # it really walks towards the road


def test_it_keeps_room_for_the_car_ahead_braking_hard():
    lead = (12.0, 0.0, 5.0, 0.0, 0.0)  # 12 m ahead at the ML car's own speed
    plan = far_planner().plan((0.0, 0.0, 5.0, 0.0, 0.0), ROUTE_EAST, [lead], [], 5.0)
    assert plan.changed and plan.speed < 5.0
    route = np.array(ROUTE_EAST)
    kinds = [h.kind for h in hypotheses([lead, (30.0, 3.5, -5.0, 0.0, 180.0), (-12.0, 0.0, 5.0, 0.0, 0.0)], [], (500.0, 500.0), route)]
    assert kinds.count("braking") == 1  # not the oncoming car, nor the one behind
    xy, _ = along_route(route, roll_out(5.0, 5.0)[1])
    assert xy[-1][0] == pytest.approx(20.0)
    assert score(xy, np.zeros(len(xy)), [h for h in hypotheses([lead], [], (500.0, 500.0), route) if h.kind == "braking"])[2] is not None

