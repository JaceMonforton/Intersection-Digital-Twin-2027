"""
Does the roadside unit help? Runs the same hidden hazards with three versions of the ML car, and compares them.

    truth   told about every actor by CARLA: the best the roadside unit could possibly do
    rsu     told what the roadside unit's detector found and what its own roof LiDAR hits, with the safety gate on the detector
    sight   knows only what a LiDAR on its own roof hits, with no roadside unit and so no safety gate

The hazards, each hidden from the car by two stacked shipping containers until late (LiDAR sees legs under a lorry):

    runner      a speeding car from behind the containers runs its red light (the suite's test 3)
    pedestrian  a person steps out from behind the containers onto the marked crossing
    turner      an oncoming car turns left across the ML car, hidden by the containers
    clear       the same containers but no hazard, to catch stops for nothing

Needs CARLA and the Docker stack running, with the entities provisioned. It starts and stops the processor itself,
so nothing else may be using CARLA. Results go to output/rsu_comparison.json and .md, and each version's tick by tick
telemetry to output/rsu_telemetry_<version>.csv.
    python scripts/run_rsu_comparison.py --model models/rsu_detector.candidate.onnx --repeat 5
"""
import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from roadside_link import MODES
from run_professor_test_suite import (
    APPROACH,
    CARLA_HOST,
    CARLA_PORT,
    OUTPUT_DIR,
    T3_LEAD,
    Console,
    Policy,
    Suite,
    along_and_across,
    follow_path,
    launch_now,
    locations_array,
    outline,
    paths_cross,
    rectangles_gap,
    release_gate,
    yaw_class,
)
from waymo_predictor_node import LEFT, STRAIGHT

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ("runner", "pedestrian", "turner", "clear")
PROCESSOR_TIMEOUT = 180.0
# Shipping containers, solid to the ground, as a lorry or a bus stands high on its wheels and a LiDAR sees a person's legs underneath
HIDER = "static.prop.container"
HIDER_BACKS = (3.0, 10.0)  # m from the stop line to the centre of each one
CROSSING_AHEAD = 5.0    # m past the stop line that the person crosses, on the crossing, just in front of the containers
HIDER_HALF_WIDTH = 1.3  # m
HIDER_REACH = 7.0       # m from a container's centre to a stopped car that is really that container
CRUISE = 6.0            # m/s; what the car settles to here, for timing a hazard, since it may slow before one sets off
WALK_SPEED = 1.6        # m/s
TURNER_SPEED = 8.0      # m/s
TURNER_BACK = 30.0      # m before its stop line that the oncoming car starts
LEADS = (-0.4, 0.0, 0.4, -0.2, 0.2)  # s the hazard reaches the ML car's route ahead of it, varied between repeats
SCENARIO_LIMIT = 45.0   # s
STOPPED = 0.3           # m/s
HALF_LENGTH, HALF_WIDTH = 2.4, 1.0  # m; the ML car, from its centre


def gap_to_person(ego, x, y):
    """Metres from the outside of the car at ego to a person at (x, y)."""
    ahead, side = along_and_across(ego, (x, y))
    return max(abs(ahead) - HALF_LENGTH, abs(side) - HALF_WIDTH, 0.0)


def summarise(runs):
    """Totals for one version on one scenario, from its runs."""
    gaps = [r["closest_m"] for r in runs if r["closest_m"] is not None]
    return {"runs": len(runs), "ran": sum(r["ran"] for r in runs), "collisions": sum(bool(r["collided"]) for r in runs),
            "closest_mean_m": round(float(np.mean(gaps)), 2) if gaps else None,
            "closest_worst_m": round(float(np.min(gaps)), 2) if gaps else None,
            "gate_brakes": sum(r["gate_brakes"] for r in runs), "stopped": sum(bool(r["stopped"]) for r in runs),
            "seconds_mean": round(float(np.mean([r["seconds"] for r in runs])), 1) if runs else None}


def report_rows(results):
    """Markdown table rows, a row for each scenario and version."""
    rows = ["| Hazard | Version | Ran | Collisions | Closest (mean) | Closest (worst) | Safety brakes | Stopped |",
            "| ------ | ------- | --- | ---------- | -------------- | --------------- | ------------- | ------- |"]
    for scenario, versions in results.items():
        for mode, runs in versions.items():
            s = summarise(runs)
            closest = "n/a" if s["closest_mean_m"] is None else f"{s['closest_mean_m']:.2f} m"
            worst = "n/a" if s["closest_worst_m"] is None else f"{s['closest_worst_m']:.2f} m"
            rows.append(f"| {scenario} | {mode} | {s['ran']} of {s['runs']} | {s['collisions']} | {closest} | {worst} | {s['gate_brakes']} | {s['stopped']} |")
    return rows


class Comparison(Suite):
    def believed(self, cars):
        """The detector reports a container as a parked car, which would slow only that version, so it's left out of what the
        driver is told, as a fixed roadside unit would learn its static background.
        shortcut: told where the containers are; a real unit needs background learning, and the safety gate still gets them."""
        places = [a.get_location() for a in self.world.get_actors().filter(HIDER)]
        return [c for c in cars if math.hypot(c[2], c[3]) > STOPPED or all(math.hypot(c[0] - p.x, c[1] - p.y) > HIDER_REACH for p in places)]

    def clear(self):
        """Suite.clear only removes cars and people. Any containers still in the world have to go too, or they stay in the road."""
        self.client.apply_batch_sync([self.carla.command.DestroyActor(a) for a in self.world.get_actors().filter(HIDER)])
        super().clear()

    def _beside_truck(self, truck_side):
        """(stop line of a lane going straight with a lane beside it on this side, that lane's stop waypoint)."""
        for stop, _ in sorted(self.controlled, key=lambda c: (round(c[0].transform.rotation.yaw / 10.0), c[0].lane_id)):
            beside = stop.get_right_lane() if truck_side == "right" else stop.get_left_lane()
            behind = stop.previous(APPROACH)
            if (beside is None or beside.lane_type != self.carla.LaneType.Driving or beside.lane_id * stop.lane_id <= 0
                    or not behind or behind[0].is_junction):
                continue
            if abs(locations_array(self.straight_path(stop, metres=40, every=5))[-1] - (stop.transform.location.x, stop.transform.location.y)).sum() > 15.0:
                return stop, beside
        raise RuntimeError(f"No lane at junction 189 goes straight with a lane on its {truck_side}.")

    def _park_truck(self, waypoint):
        bp = self.world.get_blueprint_library().find(HIDER)
        for back in HIDER_BACKS:
            place = waypoint.previous(back)[0].transform
            place.location.z += 0.1
            place.rotation.yaw += 90.0  # the container's long side is its y
            hider = self.world.try_spawn_actor(bp, place)
            if hider is None:
                raise RuntimeError("Couldn't place the container.")
            hider.set_simulate_physics(False)

    def _green_for(self, stop):
        red, green = self.carla.TrafficLightState.Red, self.carla.TrafficLightState.Green
        self.set_lights(stop.transform.rotation.yaw, green, green, red)

    def run_pedestrian(self, lead, launch=True):
        stop, beside = self._beside_truck("right")
        self._green_for(stop)
        my_path = locations_array(self.straight_path(stop.previous(APPROACH)[0], every=1))
        forward, right = stop.transform.get_forward_vector(), stop.transform.get_right_vector()
        line = stop.transform.location
        # The person waits right in front of the truck, at its far side, so that it hides them from every distance
        crossing = (line.x + forward.x * CROSSING_AHEAD, line.y + forward.y * CROSSING_AHEAD)
        walk = stop.lane_width + HIDER_HALF_WIDTH
        start = self.carla.Location(x=crossing[0] + right.x * walk, y=crossing[1] + right.y * walk, z=line.z + 1.0)
        my_index = int(np.argmin(np.hypot(my_path[:, 0] - crossing[0], my_path[:, 1] - crossing[1])))
        state = {"person": None, "launched_at": None, "closest": math.inf, "stopped": False, "speed_at_closest": None}

        def build(ml):
            self._park_truck(beside)
            if launch:
                bp = self.world.get_blueprint_library().find("walker.pedestrian.0001")
                if bp.has_attribute("is_invincible"):
                    bp.set_attribute("is_invincible", "false")
                person = self.world.try_spawn_actor(bp, self.carla.Transform(start))
                if person is None:
                    raise RuntimeError("Couldn't place the pedestrian.")
                state["person"] = person

        def on_tick(tick, result, driver):
            nearest = int(np.argmin(np.hypot(my_path[:, 0] - tick.ego[0], my_path[:, 1] - tick.ego[1])))
            person = state["person"]
            if person is not None:
                if state["launched_at"] is None and launch_now(my_index - nearest, CRUISE, walk, WALK_SPEED, lead):
                    state["launched_at"] = tick.t
                    person.apply_control(self.carla.WalkerControl(direction=self.carla.Vector3D(x=-right.x, y=-right.y), speed=WALK_SPEED))
                    self.console.line("SCENARIO", f"Person steps out, ML car {my_index - nearest} m away, t={tick.t:.1f} s")
                if person.id in tick.walkers:
                    gap = gap_to_person(tick.ego, *tick.walkers[person.id][:2])
                    if gap < state["closest"]:
                        state["closest"], state["speed_at_closest"] = gap, tick.speed
            passed = nearest > my_index + 10
            state["stopped"] = state["stopped"] or (tick.speed < STOPPED and tick.t > 5.0 and nearest < my_index)
            if tick.gate_on and tick.held_for > 3.0:
                release_gate(self.session)
            if state["launched_at"] is not None and tick.t - state["launched_at"] > 12.0:
                return True
            return passed

        result = self.run_stage(21, "Hidden pedestrian" if launch else "Clear road", "[HIDDEN PEDESTRIAN]" if launch else "[CLEAR ROAD]",
                                stop, STRAIGHT, SCENARIO_LIMIT, on_tick, build)
        if state["person"] is not None:
            state["person"].destroy()
        return result, state["closest"], state["stopped"], state["launched_at"] is not None or not launch

    def run_turner(self, lead):
        stop, beside = self._beside_truck("left")
        self._green_for(stop)
        my_path = locations_array(self.straight_path(stop.previous(APPROACH)[0], every=1))
        found = None
        for candidate, _ in self.controlled:
            if yaw_class(candidate.transform.rotation.yaw, stop.transform.rotation.yaw) != "oncoming" or not candidate.previous(TURNER_BACK):
                continue
            begin = candidate.previous(TURNER_BACK)[0]
            their_points = self.straight_path(begin, every=1, turn=LEFT)
            crossing = paths_cross(my_path, locations_array(their_points))
            if crossing is not None:
                found = (begin, their_points, crossing)
                break
        if found is None:
            raise RuntimeError("No oncoming left turn crosses the ML car's route.")
        begin, their_points, (my_index, their_index) = found
        route = [(p.x, p.y) for p in their_points]
        state = {"car": None, "launched_at": None, "closest": math.inf, "stopped": False}

        def build(ml):
            self._park_truck(beside)
            state["car"] = self.spawn_car("vehicle.audi.tt", begin)
            state["car"].set_light_state(self.carla.VehicleLightState.LeftBlinker)  # as a real driver would

        def on_tick(tick, result, driver):
            nearest = int(np.argmin(np.hypot(my_path[:, 0] - tick.ego[0], my_path[:, 1] - tick.ego[1])))
            car = state["car"]
            if car.id not in tick.others:
                return False
            other = tick.others[car.id]
            if state["launched_at"] is None and launch_now(my_index - nearest, max(tick.speed, 4.0), their_index, TURNER_SPEED, lead):
                state["launched_at"] = tick.t
                self.console.line("SCENARIO", f"Oncoming car sets off, ML car {my_index - nearest} m from the crossing, t={tick.t:.1f} s")
            if state["launched_at"] is None:
                car.apply_control(self.carla.VehicleControl(hand_brake=True))
            else:
                steer, throttle, brake = follow_path(other, route, TURNER_SPEED)
                car.apply_control(self.carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
                state["closest"] = min(state["closest"], rectangles_gap(outline(tick.ego[0], tick.ego[1], tick.ego[4]), outline(other[0], other[1], other[4])))
            state["stopped"] = state["stopped"] or (tick.speed < STOPPED and tick.t > 5.0 and nearest < my_index)
            if tick.gate_on and tick.held_for > 3.0:
                release_gate(self.session)
            if state["launched_at"] is not None and tick.t - state["launched_at"] > 12.0:
                return True
            return nearest > my_index + 10

        result = self.run_stage(22, "Hidden oncoming turner", "[HIDDEN ONCOMING TURNER]", stop, STRAIGHT, SCENARIO_LIMIT, on_tick, build)
        return result, state["closest"], state["stopped"], state["launched_at"] is not None

    def run_runner(self, lead):
        result = self.test_blind_corner(T3_LEAD + lead)
        gap = result.hazard_gap if result.hazard_gap is not None else math.inf
        return result, gap, result.brake_at is not None or bool(result.stopping_margin), gap != math.inf

    def scenario(self, name, lead):
        try:
            if name == "runner":
                result, closest, stopped, ran = self.run_runner(lead)
            elif name == "pedestrian":
                result, closest, stopped, ran = self.run_pedestrian(lead)
            elif name == "turner":
                result, closest, stopped, ran = self.run_turner(lead)
            else:
                result, closest, stopped, ran = self.run_pedestrian(lead, launch=False)
        except RuntimeError as error:
            self.console.line("ALERT", f"{name} couldn't run: {error}")
            return {"ran": False, "collided": 0, "closest_m": None, "gate_brakes": 0, "stopped": False, "seconds": 0.0, "error": str(error)}
        return {"ran": bool(ran), "lead": lead, "collided": len(result.collisions), "closest_m": None if math.isinf(closest) else round(float(closest), 2),
                "gate_brakes": result.brake_events, "stopped": bool(stopped), "seconds": result.duration, "collisions": result.collisions}


def start_processor(mode, model, log_path):
    detector = ["--detector", "onnx", "--model", model] if mode == "rsu" else ["--detector", "groundtruth"]
    log = open(log_path, "w", encoding="utf-8", errors="replace")  # noqa: SIM115
    process = subprocess.Popen([sys.executable, "-u", "scripts/central_gpu_processor.py", *detector], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
    deadline = time.time() + PROCESSOR_TIMEOUT
    while time.time() < deadline and process.poll() is None:
        if "Master tick" in Path(log_path).read_text(encoding="utf-8", errors="replace"):
            return process, log
        time.sleep(1.0)
    raise RuntimeError(f"The processor didn't start. See {log_path}.")


def stop_processor(process, log):
    if process.poll() is None:
        process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
    log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=str(ROOT / "models" / "rsu_detector.candidate.onnx"))
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    results_path = OUTPUT_DIR / "rsu_comparison.json"
    if len(args.modes) > 1:
        # Each version gets its own process, as a Traffic Manager doesn't survive being reused after its processor changes
        results_path.unlink(missing_ok=True)
        rest = [a for a in sys.argv[1:] if a not in args.modes and a != "--modes"]
        return max(subprocess.call([sys.executable, "-u", __file__, *rest, "--modes", mode], cwd=ROOT) for mode in args.modes)

    import carla

    console = Console(args.quiet)
    earlier = json.loads(results_path.read_text(encoding="utf-8"))["runs"] if results_path.exists() else {}
    results = {name: earlier.get(name, {}) for name in args.scenarios}  # the other versions' runs, measured separately
    for mode in args.modes:
        console.line("SCENARIO", f"=== Version: {mode} ===")
        process, log = start_processor(mode, args.model, OUTPUT_DIR / f"rsu_processor_{mode}.log")
        try:
            client = carla.Client(CARLA_HOST, CARLA_PORT)
            client.set_timeout(30.0)
            suite = Comparison(carla, client, Policy(), console, seed=1, perception=mode)
            try:
                for name in args.scenarios:
                    runs = []
                    for repeat in range(args.repeat):
                        run = suite.scenario(name, LEADS[repeat % len(LEADS)])
                        console.line("SCENARIO", f"{mode} {name} {repeat + 1}/{args.repeat}: {run}")
                        runs.append(run)
                    results[name][mode] = runs
            finally:
                suite.close()
            shutil.copy(OUTPUT_DIR / "live_telemetry.csv", OUTPUT_DIR / f"rsu_telemetry_{mode}.csv")
        finally:
            stop_processor(process, log)
        time.sleep(3.0)
        results_path.write_text(json.dumps({"generated": datetime.now(timezone.utc).isoformat(), "model": args.model, "repeat": args.repeat,
                                                                    "summary": {n: {m: summarise(r) for m, r in v.items()} for n, v in results.items()},
                                                                    "runs": results}, indent=2), encoding="utf-8")
    rows = report_rows(results)
    (OUTPUT_DIR / "rsu_comparison.md").write_text("\n".join(["# Roadside unit comparison", "", f"{args.repeat} runs of each. Every figure is measured.", "", *rows, ""]), encoding="utf-8")
    console.plain("\n" + "\n".join(rows) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
