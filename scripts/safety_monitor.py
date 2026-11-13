"""
The ML car's own safety monitor: a few fixed rules after the model and the planner, the same in every config, that can
only slow the car down. It isn't AI and isn't the roadside safety gate; it answers for what the car itself knows.

    stale     what the car perceives isn't fresh: hold it stopped, and let it move off only after RESUME_TICKS fresh
              ticks in a row, so a flickering sensor can't restart it every other tick
    link      the roadside link is down, or came back less than LINK_HOLD ticks ago: no faster than DEGRADED_SPEED, as
              the car can no longer see round the corners it was counting on the roadside unit for. Not once the car is
              inside the junction: there its own sensors cover it, and crawling only keeps it in cross traffic's path
    space     going on at the target collides within SPACE_HORIZON with someone keeping their velocity, and a hard stop
              now wouldn't: stop. Only when stopping helps, so not for a car coming from behind, nor once the car is
              committed and stopping would leave it in the way

No carla imports, so it stays testable. The space check uses the planner's geometry but not its futures: everyone
keeps their velocity, the simplest prediction there is, so the two can't share a mistake about where people go.
"""
import math
from collections import Counter

import numpy as np
from planner import (
    CAR_CIRCLES,
    CAR_RADIUS,
    HARD_BRAKE,
    WALKER_RADIUS,
    Hypothesis,
    along_route,
    predicted_path,
    roll_out,
    score,
    times,
)

RESUME_TICKS = 5        # fresh ticks in a row before a car held for stale perception may move off: 0.5 s at 0.1 s a tick
DEGRADED_SPEED = 4.0    # m/s while the roadside link is down; the planner's speed into a junction with no roadside unit
LINK_HOLD = 10          # ticks the cap stays after the link comes back, as one message may be a fluke
SPACE_HORIZON = 3.0     # s ahead that the space check looks for a collision


def stopping_helps(ego, route, cars, walkers, target):
    """Would going on at target hit someone within SPACE_HORIZON, everyone keeping their velocity, when a hard stop now
    wouldn't? ego is (x, y, vx, vy, yaw); route the (x, y) points ahead; cars (x, y, vx, vy, yaw in degrees) and
    walkers (x, y, vx, vy)."""
    route = np.asarray(route, float)
    if len(route) < 2 or not (cars or walkers):
        return False
    at, futures = times(), []
    for x, y, vx, vy, *rest in cars:
        path, yaws = predicted_path(x, y, vx, vy, 0.0, at)
        if math.hypot(vx, vy) < 0.5 and rest:
            yaws = np.full(len(at), math.radians(rest[0]))  # a standing car faces its own way, not along a zero velocity
        futures.append(Hypothesis("straight", 1.0, path, yaws, CAR_RADIUS, CAR_CIRCLES))
    for x, y, vx, vy in walkers:
        futures.append(Hypothesis("straight", 1.0, *predicted_path(x, y, vx, vy, 0.0, at), WALKER_RADIUS, (0.0,)))
    speed = math.hypot(ego[2], ego[3])
    go = score(*along_route(route, roll_out(speed, target)[1]), futures)[2]
    if go is None or go > SPACE_HORIZON:
        return False
    return score(*along_route(route, roll_out(speed, 0.0, HARD_BRAKE)[1]), futures)[2] is None


class SafetyMonitor:
    def __init__(self):
        self.fresh_ticks = RESUME_TICKS
        self.link_ticks = LINK_HOLD
        self.ticks = Counter()   # ticks each rule acted on, for the bench
        self.times = Counter()   # times each rule started acting
        self.last_rule = None

    def check(self, fresh, link_up, target, inside=False, conflict=False):
        """The target speed to drive at, or None to hold the car stopped. link_up is None for a config with no roadside
        unit, which has no link to lose. target is the model's or the planner's speed. inside is whether the car is in the
        junction already, and conflict whether stopping_helps."""
        self.fresh_ticks = self.fresh_ticks + 1 if fresh else 0
        if link_up is not None:
            self.link_ticks = self.link_ticks + 1 if link_up else 0
        if self.fresh_ticks < RESUME_TICKS:
            speed, rule = None, "stale"
        elif conflict and target > 0.0:
            speed, rule = 0.0, "space"
        elif link_up is not None and not inside and self.link_ticks < LINK_HOLD and target > DEGRADED_SPEED:
            speed, rule = DEGRADED_SPEED, "link"
        else:
            speed, rule = target, None
        if rule is not None:
            self.ticks[rule] += 1
            if rule != self.last_rule:
                self.times[rule] += 1
        self.last_rule = rule
        return speed
