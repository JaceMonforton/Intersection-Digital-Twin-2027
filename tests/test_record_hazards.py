"""The rule that teaches giving way to hazards on the route. No CARLA needed."""
import math

import pytest

pytest.importorskip("paho")

import numpy as np
from record_hazards import (
    BRAKING,
    REACH,
    STOP_SHORT,
    conflict_distance,
    hazard_target,
    predicted_path,
)
from waymo_predictor_node import LEFT, STRAIGHT

LANE = [(2.0 * i, 0.0) for i in range(11)]  # straight ahead along x, as the model is given it


def seen(cars=(), walkers=(), speed=5.0, route=STRAIGHT):
    return {"lane": LANE, "speed": speed, "route": route, "cars": list(cars), "walkers": list(walkers),
            "crossing_ttc": None, "turn_gap": 40.0, "stop_gap": None}


def test_a_person_stepping_onto_the_route_ahead_is_a_conflict_where_they_cross():
    person = (8.0, 3.0, 0.0, -1.5)  # 8 m ahead, 3 m to the left, walking across
    where = conflict_distance(seen(walkers=[person]))
    assert 8.0 - REACH["walker"] <= where <= 8.0  # a body covers the route points within its reach
    room = where - STOP_SHORT["walker"]
    assert hazard_target(seen(walkers=[person]), 5.5) == pytest.approx(math.sqrt(2.0 * BRAKING * room))


def test_a_car_ahead_going_the_same_way_or_passing_the_other_way_is_left_to_the_general_driver():
    assert conflict_distance(seen(cars=[(10.0, 0.0, 5.0, 0.0)])) == math.inf
    assert conflict_distance(seen(cars=[(15.0, 3.5, -8.0, 0.0)])) == math.inf  # oncoming, in its own lane
    assert hazard_target(seen(cars=[(10.0, 0.0, 5.0, 0.0)]), 5.5) == 5.5


def test_a_car_turning_across_the_route_makes_it_stop_short_and_one_already_gone_doesnt():
    turner = (16.0, 6.0, 0.0, -6.0)  # will be on the route 16 m ahead in 1 s, just before the ML car
    assert 16.0 - REACH["car"] <= conflict_distance(seen(cars=[turner])) <= 16.0
    gone = (16.0, 1.0, 0.0, -12.0)  # across and away long before the ML car is there
    assert conflict_distance(seen(cars=[gone], speed=2.0)) == math.inf


def test_it_asks_to_stop_now_when_a_conflict_is_too_close_to_stop_short_of():
    assert hazard_target(seen(walkers=[(4.0, 0.5, 0.0, 0.0)]), 5.5) == 0.0


def test_on_a_left_turn_oncoming_cars_are_the_left_turn_rules_but_people_still_count():
    car = (16.0, 6.0, 0.0, -6.0)
    assert hazard_target(seen(cars=[car], route=LEFT), 5.5) == hazard_target(seen(route=LEFT), 5.5)
    assert hazard_target(seen(walkers=[(6.0, 2.0, 0.0, -1.5)], route=LEFT), 5.5) < hazard_target(seen(route=LEFT), 5.5)


def test_a_turning_car_follows_its_turn_and_is_seen_crossing_before_it_points_across():
    path = predicted_path(0.0, 0.0, 5.0, 0.0, math.pi / 2, np.array([0.0, 1.0]))  # a quarter circle to the left in 1 s
    radius = 5.0 / (math.pi / 2)
    assert path[1] == pytest.approx((radius, radius))
    oncoming = (30.0, 3.5, -8.0, 0.0)  # in its own lane, coming towards the ML car
    assert conflict_distance(seen(cars=[oncoming])) == math.inf
    assert conflict_distance(seen(cars=[(*oncoming, 0.6)])) < math.inf  # the same car, turning left across the route
