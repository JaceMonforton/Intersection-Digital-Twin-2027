"""
The ML car's own safety monitor: a few fixed rules after the model and the planner, the same in every config, that can
only slow the car down. It isn't AI and isn't the roadside safety gate; it answers for what the car itself knows.

    stale     what the car perceives isn't fresh: hold it stopped, and let it move off only after RESUME_TICKS fresh
              ticks in a row, so a flickering sensor can't restart it every other tick
    link      the roadside link is down, or came back less than LINK_HOLD ticks ago: no faster than DEGRADED_SPEED, as
              the car can no longer see round the corners it was counting on the roadside unit for

No carla or numpy imports, so it stays testable.
"""
from collections import Counter

RESUME_TICKS = 5        # fresh ticks in a row before a car held for stale perception may move off: 0.5 s at 0.1 s a tick
DEGRADED_SPEED = 4.0    # m/s while the roadside link is down; the planner's speed into a junction with no roadside unit
LINK_HOLD = 10          # ticks the cap stays after the link comes back, as one message may be a fluke


class SafetyMonitor:
    def __init__(self):
        self.fresh_ticks = RESUME_TICKS
        self.link_ticks = LINK_HOLD
        self.ticks = Counter()   # ticks each rule acted on, for the bench
        self.times = Counter()   # times each rule started acting
        self.last_rule = None

    def check(self, fresh, link_up, target):
        """The target speed to drive at, or None to hold the car stopped. link_up is None for a config with no roadside
        unit, which has no link to lose. target is the model's or the planner's speed."""
        self.fresh_ticks = self.fresh_ticks + 1 if fresh else 0
        if link_up is not None:
            self.link_ticks = self.link_ticks + 1 if link_up else 0
        if self.fresh_ticks < RESUME_TICKS:
            speed, rule = None, "stale"
        elif link_up is not None and self.link_ticks < LINK_HOLD and target > DEGRADED_SPEED:
            speed, rule = DEGRADED_SPEED, "link"
        else:
            speed, rule = target, None
        if rule is not None:
            self.ticks[rule] += 1
            if rule != self.last_rule:
                self.times[rule] += 1
        self.last_rule = rule
        return speed
