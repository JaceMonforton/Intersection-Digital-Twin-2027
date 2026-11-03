"""
Safe vs unsafe unprotected left turn, driven purely inside CARLA.

vehicle_ml waits at the stop line to turn left across human_vehicle, which comes straight through
without slowing. Both runs use the same paths and speeds and differ only in when the turner goes:
    unsafe: it pulls out just ahead of the oncoming car
    safe:   it waits for the oncoming car to clear, then goes

The running central_gpu_processor.py decides whether to brake, from straight-line TTC and from the
vehicles' predicted paths, and the turner obeys its SafetyGate.
Each run is compared with the same plan replayed with no brake, to show what the brake changed.

Normally started by run_simulation.py. To run it alone, start the processor first:
    python scripts/left_turn_scenario.py --case both
"""
import argparse
import csv
import itertools
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from central_gpu_processor import (
    COLLISION_RADIUS,
    DITTO_API,
    FIXED_DELTA,
    TTC_BRAKE,
    Track,
    deceleration_to_avoid_crash,
    post_encroachment_time,
    time_to_collision,
)

# --- Configuration ---
CARLA_HOST = "localhost"
CARLA_PORT = 2000
GATE_URL = f"{DITTO_API}/things/org.example:vehicle_ml/features/SafetyGate/properties"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

ONCOMING_SPEED = 10.0     # m/s, about 36 km/h
TURN_SPEED = 6.0          # m/s, the turner's speed once rolling
TURN_ACCEL = 2.5          # m/s^2, pulling away from the stop line
BRAKE_DECEL = 6.0         # m/s^2, firm braking on dry road
ONCOMING_ARRIVAL = 7.0    # s after the run starts that the oncoming car reaches the conflict point
UNSAFE_LEAD = 0.8         # s the unsafe turner arrives ahead of the oncoming car
SAFE_GAP = 3.0            # s the safe turner arrives after the oncoming car
STOP_LINE_SETBACK = 3.0   # m before the junction entry
UPSTREAM = 80.0           # m of road modelled before the junction entry
DOWNSTREAM = 30.0         # m of road modelled after the junction exit
ZONE_RADIUS = COLLISION_RADIUS  # m around the conflict point counted as "in the conflict zone"
WAYPOINT_STEP = 1.0       # m
RUN_LIMIT = 25.0          # s of simulation before a run is cut off
SETTLE_AFTER = 1.5        # s kept running after the run's outcome is decided
Z_CLEARANCE = 0.3         # m above the road
VEHICLE_LENGTH = 4.8      # m, Tesla Model 3
VEHICLE_WIDTH = 2.1       # m
DRAC_MIN_GAP = 0.5        # m; below this the required deceleration runs off to meaningless values
UNSAFE_COLOR = (160, 0, 0)  # debug text and markers; dimmed, because full brightness blooms
SAFE_COLOR = (0, 140, 0)
SIGNAL_OFF_AFTER = 15.0   # m past the conflict point that the turn signal is cancelled
MAX_OFF_VIEW = 30.0       # m the conflict point may be from directly under the spectator

PET_UNSAFE = 1.5          # s; below this, a crossing is commonly treated as a serious conflict
DRAC_UNSAFE = 3.35        # m/s^2; a widely used threshold for a conflict needing hard braking


# ---------------------------------------------------------------------------
# Path geometry (no CARLA needed)
# ---------------------------------------------------------------------------
class Path2D:
    """A polyline that can be sampled by distance travelled along it."""

    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)
        segments = np.diff(self.points[:, :2], axis=0)
        self.lengths = np.concatenate(([0.0], np.cumsum(np.hypot(segments[:, 0], segments[:, 1]))))
        self.length = float(self.lengths[-1])

    def position(self, s):
        s = min(max(s, 0.0), self.length)
        return (float(np.interp(s, self.lengths, self.points[:, 0])),
                float(np.interp(s, self.lengths, self.points[:, 1])))

    def height(self, s):
        return float(np.interp(s, self.lengths, self.points[:, 2])) if self.points.shape[1] > 2 else 0.0

    def yaw(self, s):
        """Direction of travel at s, in degrees."""
        ahead, behind = self.position(s + 0.5), self.position(s - 0.5)
        return math.degrees(math.atan2(ahead[1] - behind[1], ahead[0] - behind[0]))

    def nearest_s(self, x, y):
        distance = np.hypot(self.points[:, 0] - x, self.points[:, 1] - y)
        return float(self.lengths[int(np.argmin(distance))])


def conflict_point(a, b):
    """(x, y, s along a, s along b) where two paths come closest, and that distance."""
    delta = a.points[:, None, :2] - b.points[None, :, :2]
    distance = np.hypot(delta[..., 0], delta[..., 1])
    i, j = np.unravel_index(int(np.argmin(distance)), distance.shape)
    x = float((a.points[i, 0] + b.points[j, 0]) / 2)
    y = float((a.points[i, 1] + b.points[j, 1]) / 2)
    return x, y, float(a.lengths[i]), float(b.lengths[j]), float(distance[i, j])


def yaw_difference(a, b):
    """b - a in degrees, wrapped to (-180, 180]. Negative is a left turn in CARLA's frame."""
    return (b - a + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Motion plan (no CARLA needed)
# ---------------------------------------------------------------------------
def time_to_cover(distance, cruise, accel):
    """Seconds to cover distance from rest, accelerating to cruise and holding it."""
    ramp_distance = cruise * cruise / (2 * accel)
    if distance <= ramp_distance:
        return math.sqrt(2 * distance / accel)
    return cruise / accel + (distance - ramp_distance) / cruise


def departure_time(case, distance_to_conflict):
    """When the turner should leave the stop line for this case."""
    arrival = ONCOMING_ARRIVAL - UNSAFE_LEAD if case == "unsafe" else ONCOMING_ARRIVAL + SAFE_GAP
    return arrival - time_to_cover(distance_to_conflict, TURN_SPEED, TURN_ACCEL)


class Turner:
    """Waits, pulls away, and brakes hard if told to. Stateful so a brake can interrupt the plan."""

    def __init__(self, depart_at, start_s=0.0):
        self.depart_at = depart_at
        self.s = start_s
        self.v = 0.0

    def step(self, t, dt, braked):
        if braked:
            self.v = max(0.0, self.v - BRAKE_DECEL * dt)
        elif t >= self.depart_at:
            self.v = min(TURN_SPEED, self.v + TURN_ACCEL * dt)
        self.s += self.v * dt
        return self.s, self.v


# ---------------------------------------------------------------------------
# Analysis (no CARLA needed)
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    t: float
    turner_x: float
    turner_y: float
    turner_vx: float
    turner_vy: float
    oncoming_x: float
    oncoming_y: float
    oncoming_vx: float
    oncoming_vy: float
    braked: bool = False
    turner_yaw: float = 0.0
    oncoming_yaw: float = 0.0


@dataclass
class Result:
    case: str
    verdict: str = ""
    reasons: list = field(default_factory=list)
    contact: bool = False
    min_separation: float = math.inf
    min_ttc: float = math.inf
    min_predicted_pet: float = math.inf
    max_drac: float = 0.0
    pet: float = None          # measured; None if the turner never entered the conflict zone
    turner_entered_zone: bool = False
    brake_fired_at: float = None
    ttc_when_braked: float = None
    separation_when_braked: float = None
    turner_stopped_short_by: float = None  # m before the conflict point, if it never reached it


def outline(x, y, yaw_deg):
    """The four corners of a vehicle seen from above."""
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    half_l, half_w = VEHICLE_LENGTH / 2, VEHICLE_WIDTH / 2
    return [(x + c * lx - s * ly, y + s * lx + c * ly)
            for lx, ly in ((half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w))]


def rectangles_overlap(a, b):
    """Separating axis test: two rectangles are apart if some edge's normal separates them."""
    for rect in (a, b):
        for (x1, y1), (x2, y2) in itertools.pairwise(rect):
            nx, ny = y2 - y1, x1 - x2
            pa = [nx * x + ny * y for x, y in a]
            pb = [nx * x + ny * y for x, y in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


def outlines_overlap(s):
    return rectangles_overlap(outline(s.turner_x, s.turner_y, s.turner_yaw),
                              outline(s.oncoming_x, s.oncoming_y, s.oncoming_yaw))


def zone_times(samples, x_attr, y_attr, center):
    """(enter, exit) of the conflict zone for one vehicle, or None if it never entered."""
    inside = [s.t for s in samples
              if math.hypot(getattr(s, x_attr) - center[0], getattr(s, y_attr) - center[1]) <= ZONE_RADIUS]
    return (inside[0], inside[-1]) if inside else None


def measured_pet(turner_zone, oncoming_zone):
    """Seconds between the first vehicle leaving the zone and the second entering it. A negative
    value means both were in it together. None if they didn't both pass through."""
    if turner_zone is None or oncoming_zone is None:
        return None
    first, second = sorted((turner_zone, oncoming_zone))
    return second[0] - first[1]


def analyse(case, samples, center):
    result = Result(case)
    for s in samples:
        a = Track("vehicle_ml", s.turner_x, s.turner_y, s.turner_vx, s.turner_vy)
        b = Track("human_vehicle", s.oncoming_x, s.oncoming_y, s.oncoming_vx, s.oncoming_vy)
        ttc = time_to_collision(a, b)
        separation = math.hypot(a.x - b.x, a.y - b.y)
        result.min_separation = min(result.min_separation, separation)
        result.min_ttc = min(result.min_ttc, ttc)
        result.min_predicted_pet = min(result.min_predicted_pet, post_encroachment_time(a, b))
        # Only while heading for contact: a car passing in its own lane is "closing" but no conflict
        if ttc < math.inf and separation - COLLISION_RADIUS >= DRAC_MIN_GAP:
            result.max_drac = max(result.max_drac, deceleration_to_avoid_crash(a, b))
        result.contact = result.contact or outlines_overlap(s)
        if s.braked and result.brake_fired_at is None:
            result.brake_fired_at = s.t
            result.ttc_when_braked = ttc
            result.separation_when_braked = separation

    turner_zone = zone_times(samples, "turner_x", "turner_y", center)
    result.turner_entered_zone = turner_zone is not None
    result.pet = measured_pet(turner_zone, zone_times(samples, "oncoming_x", "oncoming_y", center))
    if turner_zone is None and samples:
        last = samples[-1]
        result.turner_stopped_short_by = math.hypot(last.turner_x - center[0], last.turner_y - center[1])
    result.verdict, result.reasons = classify(result)
    return result


def classify(r):
    """UNSAFE if anything about the run crossed a conflict threshold."""
    reasons = []
    if r.contact:
        reasons.append(f"contact: vehicle outlines overlapped (centres {r.min_separation:.1f} m apart)")
    if r.min_ttc < TTC_BRAKE:
        reasons.append(f"min TTC {r.min_ttc:.2f} s (under {TTC_BRAKE} s)")
    if r.pet is not None and r.pet < PET_UNSAFE:
        reasons.append(f"PET {r.pet:.2f} s (under {PET_UNSAFE} s)")
    if r.max_drac > DRAC_UNSAFE:
        reasons.append(f"max DRAC {r.max_drac:.1f} m/s2 (over {DRAC_UNSAFE} m/s2)")
    if r.brake_fired_at is not None:
        reasons.append(f"safety brake fired at t={r.brake_fired_at:.1f} s")
    return ("UNSAFE" if reasons else "SAFE"), reasons


def replay_without_brake(case, turner_path, oncoming_path, turner_start, conflict, dt=FIXED_DELTA):
    """What the plan does if nothing intervenes."""
    cx, cy, turner_s, oncoming_s, _ = conflict
    turner = Turner(departure_time(case, turner_s - turner_start), turner_start)
    oncoming_start = oncoming_s - ONCOMING_SPEED * ONCOMING_ARRIVAL
    samples = []
    t = 0.0
    while t < RUN_LIMIT and turner.s < turner_path.length:
        s, v = turner.step(t, dt, False)
        samples.append(make_sample(t, turner_path, s, v, oncoming_path,
                                   oncoming_start + ONCOMING_SPEED * t, False))
        t += dt
    return analyse(case, samples, (cx, cy))


def make_sample(t, turner_path, turner_s, turner_v, oncoming_path, oncoming_s, braked):
    tx, ty = turner_path.position(turner_s)
    ox, oy = oncoming_path.position(oncoming_s)
    tyaw_deg, oyaw_deg = turner_path.yaw(turner_s), oncoming_path.yaw(oncoming_s)
    tyaw, oyaw = math.radians(tyaw_deg), math.radians(oyaw_deg)
    oncoming_v = ONCOMING_SPEED if oncoming_s < oncoming_path.length else 0.0
    return Sample(t, tx, ty, turner_v * math.cos(tyaw), turner_v * math.sin(tyaw),
                  ox, oy, oncoming_v * math.cos(oyaw), oncoming_v * math.sin(oyaw), braked, tyaw_deg, oyaw_deg)


def finite(v):
    """JSON has no Infinity, so 'never' is written as null."""
    if isinstance(v, float):
        return round(v, 3) if math.isfinite(v) else None
    return v


def format_seconds(v, unit=" s"):
    return "never" if v is None or not math.isfinite(v) else f"{v:.2f}{unit}"


def print_comparison(runs):
    rows = [
        ("Verdict", lambda r: r.verdict),
        ("Min TTC", lambda r: format_seconds(r.min_ttc)),
        ("Measured PET", lambda r: format_seconds(r.pet)),
        ("Max DRAC", lambda r: f"{r.max_drac:.2f} m/s2"),
        ("Closest centres", lambda r: f"{r.min_separation:.2f} m"),
        ("Contact", lambda r: "YES" if r.contact else "no"),
        ("Brake fired", lambda r: format_seconds(r.brake_fired_at).replace("never", "no")),
    ]
    columns = []
    for case, (actual, planned) in runs.items():
        columns.append((f"{case} (no brake)", planned))
        columns.append((f"{case} (actual)", actual))
    width = 18
    print("\n" + "=" * (18 + width * len(columns)))
    print("LEFT TURN ANALYSIS")
    print("=" * (18 + width * len(columns)))
    print(f"{'':<18}" + "".join(f"{title:<{width}}" for title, _ in columns))
    for label, read in rows:
        print(f"{label:<18}" + "".join(f"{read(r):<{width}}" for _, r in columns))
    for case, (actual, _) in runs.items():
        print(f"\n{case.upper()} run: {actual.verdict}")
        for reason in actual.reasons:
            print(f"  - {reason}")


# ---------------------------------------------------------------------------
# CARLA
# ---------------------------------------------------------------------------
def extended_path(carla, entry, exit_wp):
    """A lane through the junction, extended up and down the road on either side."""
    upstream = []
    wp = entry
    for _ in range(int(UPSTREAM / WAYPOINT_STEP)):
        previous = wp.previous(WAYPOINT_STEP)
        if not previous:
            break
        wp = previous[0]
        upstream.append(wp)
    middle = [entry, *entry.next_until_lane_end(WAYPOINT_STEP)]
    downstream = []
    wp = exit_wp
    for _ in range(int(DOWNSTREAM / WAYPOINT_STEP)):
        following = wp.next(WAYPOINT_STEP)
        if not following:
            break
        wp = following[0]
        downstream.append(wp)
    all_points = [*reversed(upstream), *middle, *downstream]
    path = Path2D([(w.transform.location.x, w.transform.location.y, w.transform.location.z) for w in all_points])
    return path, float(path.lengths[len(upstream)])


def find_left_turn(carla, world):
    """The left-turn lane and the opposing straight lane that crosses it, at the junction the
    processor has aimed the spectator at."""
    view = world.get_spectator().get_transform().location
    junctions = {}
    for wp, _ in world.get_map().get_topology():
        if wp.is_junction:
            junction = wp.get_junction()
            junctions[junction.id] = junction
    junction = min(junctions.values(), key=lambda j: j.bounding_box.location.distance(view))

    lanes = junction.get_waypoints(carla.LaneType.Driving)
    turning, straight = [], []
    for entry, exit_wp in lanes:
        change = yaw_difference(entry.transform.rotation.yaw, exit_wp.transform.rotation.yaw)
        if -135 < change < -45:
            turning.append((entry, exit_wp))
        elif abs(change) < 20:
            straight.append((entry, exit_wp))

    for entry, exit_wp in turning:
        for other_entry, other_exit in straight:
            opposing = abs(yaw_difference(entry.transform.rotation.yaw, other_entry.transform.rotation.yaw)) > 150
            if not opposing:
                continue
            turner_path, turner_entry_s = extended_path(carla, entry, exit_wp)
            oncoming_path, _ = extended_path(carla, other_entry, other_exit)
            if conflict_point(turner_path, oncoming_path)[4] < 2.0:
                return turner_path, turner_entry_s, oncoming_path
    raise RuntimeError("No left turn across an opposing lane at this junction.")


def gate_engaged(session):
    try:
        return session.get(GATE_URL, timeout=1.0).json().get("emergency_brake") is True
    except (requests.RequestException, ValueError):
        return False  # the processor retries; a blip here only delays this run's reaction


def release_gate(session):
    body = {"emergency_brake": False, "reason": None, "ttc": None,
            "updated_at": datetime.now().astimezone().isoformat()}
    session.put(GATE_URL, json=body, timeout=3.0).raise_for_status()


def destroy_twins(world):
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.attributes.get("role_name") in ("vehicle_ml", "human_vehicle"):
            actor.destroy()
    world.wait_for_tick(5.0)  # the actor list is a per-tick snapshot, and still shows them until then


def spawn_twin(carla, world, role, color, path, s):
    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    bp.set_attribute("role_name", role)
    bp.set_attribute("color", color)
    actor = world.spawn_actor(bp, twin_transform(carla, path, s))
    actor.set_simulate_physics(False)  # the plan owns the pose
    return actor


def twin_transform(carla, path, s):
    x, y = path.position(s)
    return carla.Transform(carla.Location(x=x, y=y, z=path.height(s) + Z_CLEARANCE),
                           carla.Rotation(yaw=path.yaw(s)))


def run_case(carla, world, session, case, turner_path, oncoming_path, turner_start, conflict):
    cx, cy, turner_conflict_s, oncoming_conflict_s, _ = conflict
    label_at = carla.Location(x=cx, y=cy, z=4.0)

    destroy_twins(world)
    release_gate(session)
    time.sleep(0.5)

    turner = Turner(departure_time(case, turner_conflict_s - turner_start), turner_start)
    oncoming_start = oncoming_conflict_s - ONCOMING_SPEED * ONCOMING_ARRIVAL
    ml = spawn_twin(carla, world, "vehicle_ml", "255,120,0", turner_path, turner.s)
    human = spawn_twin(carla, world, "human_vehicle", "0,120,255", oncoming_path, oncoming_start)
    ml.set_light_state(carla.VehicleLightState.LeftBlinker)  # tells the predictor where it's headed
    signalling = True
    print(f"\n[{case.upper()}] Oncoming reaches the conflict point at t={ONCOMING_ARRIVAL:.1f} s; "
          f"turner leaves the stop line at t={turner.depart_at:.1f} s.")

    samples = []
    start = None
    braked = False
    settle_until = None
    while True:
        snapshot = world.wait_for_tick(5.0)
        now = snapshot.timestamp.elapsed_seconds
        start = now if start is None else start
        t = now - start
        dt = snapshot.timestamp.delta_seconds

        if not braked and gate_engaged(session):
            braked = True
            print(f"[{case.upper()}] t={t:.1f} s: brake received, turner stopping.")

        s, v = turner.step(t, dt, braked)
        oncoming_s = oncoming_start + ONCOMING_SPEED * t
        ml.set_transform(twin_transform(carla, turner_path, s))
        human.set_transform(twin_transform(carla, oncoming_path, oncoming_s))
        samples.append(make_sample(t, turner_path, s, v, oncoming_path, oncoming_s, braked))
        if signalling and s > turner_conflict_s + SIGNAL_OFF_AFTER:
            ml.set_light_state(carla.VehicleLightState.NONE)
            signalling = False

        color = carla.Color(*(UNSAFE_COLOR if case == "unsafe" else SAFE_COLOR))
        world.debug.draw_string(label_at, f"{case.upper()} LEFT TURN  t={t:.1f}s", color=color, life_time=0.11)
        world.debug.draw_point(carla.Location(x=cx, y=cy, z=0.5), size=0.2, color=color, life_time=0.11)

        turner_done = s >= turner_path.length or (braked and v == 0.0)
        oncoming_done = oncoming_s >= oncoming_path.length
        if settle_until is None and turner_done and oncoming_done:
            settle_until = t + SETTLE_AFTER
        if (settle_until is not None and t >= settle_until) or t >= RUN_LIMIT:
            break

    result = analyse(case, samples, (cx, cy))
    verdict_color = carla.Color(*(UNSAFE_COLOR if result.verdict == "UNSAFE" else SAFE_COLOR))
    world.debug.draw_string(label_at, f"{case.upper()} LEFT TURN: {result.verdict}",
                            color=verdict_color, life_time=4.0)
    time.sleep(4.0)
    destroy_twins(world)
    release_gate(session)
    return result, samples


def write_outputs(runs, samples_by_case, conflict):
    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for case, samples in samples_by_case.items():
        with open(OUTPUT_DIR / f"left_turn_{stamp}_{case}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(samples[0])))
            writer.writeheader()
            for s in samples:
                writer.writerow(asdict(s))
    report = {
        "conflict_point": {"x": conflict[0], "y": conflict[1]},
        "thresholds": {"ttc": TTC_BRAKE, "pet": PET_UNSAFE, "drac": DRAC_UNSAFE, "contact_m": COLLISION_RADIUS},
        "runs": {
            case: {
                "actual": {k: finite(v) for k, v in asdict(actual).items()},
                "without_brake": {k: finite(v) for k, v in asdict(planned).items()},
            } for case, (actual, planned) in runs.items()
        },
    }
    path = OUTPUT_DIR / f"left_turn_{stamp}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {path}\nPer-tick data: {OUTPUT_DIR / f'left_turn_{stamp}_<case>.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=("safe", "unsafe", "both"), default="both")
    args = parser.parse_args()

    import carla

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(30.0)
    world = client.get_world()
    if not world.get_settings().synchronous_mode:
        print("[SCENARIO] CARLA isn't synchronous. Start central_gpu_processor.py first.")
        return 1

    world.wait_for_tick(10.0)  # until the first tick, a new client reads the spectator as (0, 0)
    turner_path, entry_s, oncoming_path = find_left_turn(carla, world)
    turner_start = max(0.0, entry_s - STOP_LINE_SETBACK)
    conflict = conflict_point(turner_path, oncoming_path)
    view = world.get_spectator().get_transform().location
    if math.hypot(conflict[0] - view.x, conflict[1] - view.y) > MAX_OFF_VIEW:
        print(f"[SCENARIO] The conflict point ({conflict[0]:.0f}, {conflict[1]:.0f}) isn't under the "
              f"spectator ({view.x:.0f}, {view.y:.0f}), so the run would be off-screen.")
        return 1
    print(f"[SCENARIO] Conflict point at x={conflict[0]:.1f} y={conflict[1]:.1f}; the turner's stop line is "
          f"{conflict[2] - turner_start:.1f} m before it.")
    if conflict[3] < ONCOMING_SPEED * ONCOMING_ARRIVAL:
        print(f"[SCENARIO] Only {conflict[3]:.0f} m of road upstream, so the oncoming car starts mid-road.")

    session = requests.Session()
    runs, samples_by_case = {}, {}
    # Unsafe first: the point of the comparison is what changes when the turner waits
    for case in ("unsafe", "safe") if args.case == "both" else (args.case,):
        actual, samples = run_case(carla, world, session, case, turner_path, oncoming_path, turner_start, conflict)
        planned = replay_without_brake(case, turner_path, oncoming_path, turner_start, conflict)
        runs[case] = (actual, planned)
        samples_by_case[case] = samples

    print_comparison(runs)
    write_outputs(runs, samples_by_case, conflict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
