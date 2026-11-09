"""
What changes from one run of a scenario to the next, drawn from its seed, so that the car is trained and judged on
many situations rather than one. The same scenario and seed always give the same variation, so any run can be
repeated exactly, and the seeds are split so that nothing is tuned on the runs that judge it:

    train   seeds the driving model and planner may learn from
    tune    seeds for choosing thresholds and comparing versions while working
    test    seeds for the final measurement only; looking at them while working makes them worthless

Standard library only, so the launcher and tests can use it without CARLA.

shortcut: weather and time of day don't change, since both detectors were only trained in clear daylight; vary them
once the detectors have been trained on varied weather.
"""
import random
from dataclasses import asdict, dataclass

SPLITS = {"train": range(1000, 5000), "tune": range(5000, 6000), "test": range(9000, 10000)}
SCENARIOS = ("left_turn", "oncoming_turner", "hidden_pedestrian")


@dataclass(frozen=True)
class Variation:
    scenario: str
    seed: int
    traffic: int = 0            # background autopilot cars
    approach: int = 0           # which of the approaches that suit the scenario, counted round
    lead: float = 0.0           # s the hazard reaches the ML car's route ahead of it; a big lead or lag is easy, near 0 is hard
    hidden: bool = True         # whether the containers hide the hazard
    walk_speed: float = 1.6     # m/s, the hidden pedestrian
    turner_speed: float = 8.0   # m/s, the oncoming car in oncoming_turner
    turner_turns: bool = True   # False: it carries straight on, so the right answer is not to stop for it
    oncoming_speed: float = 12.0  # m/s, the oncoming traffic in left_turn
    first_wave: float = 30.0    # s of steady oncoming traffic at the start of left_turn
    headway: tuple = (2.0, 6.0)  # s between oncoming cars in a wave
    gap: tuple = (11.0, 14.0)   # s between waves, the gaps to turn through

    def as_row(self):
        return {f"var_{k}": (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items() if k not in ("scenario", "seed")}


def split_of(seed):
    return next((name for name, seeds in SPLITS.items() if seed in seeds), None)


def draw(scenario, seed, traffic=None):
    """The variation for this scenario and seed. traffic fixes the background cars instead of drawing them."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario}")
    rng = random.Random(f"{scenario}:{seed}")
    common = {"scenario": scenario, "seed": seed, "approach": rng.randrange(4),
              "traffic": rng.randint(0, 20) if traffic is None else traffic}
    if scenario == "hidden_pedestrian":
        return Variation(**common, lead=rng.uniform(-1.5, 1.5), hidden=rng.random() < 0.8, walk_speed=rng.uniform(1.0, 2.2))
    if scenario == "oncoming_turner":
        return Variation(**common, lead=rng.uniform(-1.5, 1.5), hidden=rng.random() < 0.8, turner_speed=rng.uniform(5.0, 10.0),
                         turner_turns=rng.random() < 0.8)
    return Variation(**common, oncoming_speed=rng.uniform(9.0, 14.0), first_wave=rng.uniform(0.0, 30.0),
                     headway=(2.0, rng.uniform(4.0, 10.0)), gap=(rng.uniform(5.0, 11.0), 14.0))


def seeds(split, count, start=0):
    """count seeds from a split, in order, so that a bench of N runs is always the same N."""
    chosen = SPLITS[split]
    if start + count > len(chosen):
        raise ValueError(f"The {split} split has only {len(chosen)} seeds")
    return list(chosen[start:start + count])
