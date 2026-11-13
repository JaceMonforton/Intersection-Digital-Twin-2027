"""
The hidden hazards, each hidden from the ML car by two stacked shipping containers until late (a LiDAR sees legs and
wheels under a lorry, so a lorry doesn't hide anything):

    run_runner      a speeding car from behind a truck runs its red light (the suite's test_blind_corner)
    run_pedestrian  a person steps out from behind the containers onto the marked crossing
    run_turner      an oncoming car turns left across the ML car, hidden by the containers (for training/record_hazards.py)

HiddenHazards extends the suite with these, and with what the car is told about the containers themselves.
start_processor and stop_processor run central_gpu_processor.py for one config. run_bench.py uses all of this.
"""
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scenario_suite import (
    APPROACH,
    T3_LEAD,
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
from study_names import processor_args
from waymo_predictor_node import LEFT, STRAIGHT, yaw_difference

ROOT = Path(__file__).resolve().parent.parent
PROCESSOR_TIMEOUT = 180.0
# Shipping containers, solid to the ground, as a lorry or a bus stands high on its wheels and a LiDAR sees a person's legs underneath
HIDER = "static.prop.container"
HIDER_BACKS = (3.0, 10.0)  # m from the stop line to the centre of each one
CROSSING_AHEAD = 5.0    # m past the stop line that the person crosses, on the crossing, just in front of the containers
HIDER_HALF_WIDTH = 1.3  # m
HIDER_REACH = 7.0       # m from a container's centre to a stopped car that is really that container
HIDER_MARGIN = 0.5      # m round a container's footprint that a track is still that container, whatever its speed
CRUISE = 6.0            # m/s; what the car settles to here, for timing a hazard, since it may slow before one sets off
WALK_SPEED = 1.6        # m/s
TURNER_SPEED = 8.0      # m/s
TURNER_BACK = 30.0      # m before its stop line that the oncoming car starts
LEADS = (-0.4, 0.0, 0.4, -0.2, 0.2)  # s the hazard reaches the ML car's route ahead of it, varied between repeats
SCENARIO_LIMIT = 45.0   # s
STOPPED = 0.3           # m/s
HALF_LENGTH, HALF_WIDTH = 2.4, 1.0  # m; the ML car, from its centre
CROSSING_SLACK = 5.0    # m short of the stop line that a crossing still counts as in the junction
RUN_UP_REACH = 3.0      # m from the ML car's run-up that another junction's stop line is on it


def gap_to_person(ego, x, y):
    """Metres from the outside of the car at ego to a person at (x, y)."""
    ahead, side = along_and_across(ego, (x, y))
    return max(abs(ahead) - HALF_LENGTH, abs(side) - HALF_WIDTH, 0.0)


def in_footprint(x, y, cx, cy, yaw, half_x, half_y, margin=HIDER_MARGIN):
    """Is (x, y) within margin of a box centred on (cx, cy), yaw degrees, half_x by half_y along its own axes?"""
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    dx, dy = x - cx, y - cy
    return abs(c * dx + s * dy) <= half_x + margin and abs(-s * dx + c * dy) <= half_y + margin


class HiddenHazards(Suite):
    def believed(self, cars):
        """The detector reports a container as a parked car, which would slow only that version, so it's left out of what the
        driver is told, as a fixed roadside unit would learn its static background.
        The centroid tracker gives a container a speed that jumps about, up to 19 m/s as more of it comes into view, so a
        track inside a container's footprint is dropped whatever its speed: no real car can be there.
        shortcut: told where the containers are; a real unit needs background learning, and the safety gate still gets them."""
        hiders = self.world.get_actors().filter(HIDER)
        places = [a.get_location() for a in hiders]
        footprints = [(a.get_transform(), a.bounding_box.extent) for a in hiders]
        return [c for c in cars
                if (math.hypot(c[2], c[3]) > STOPPED or all(math.hypot(c[0] - p.x, c[1] - p.y) > HIDER_REACH for p in places))
                and not any(in_footprint(c[0], c[1], tf.location.x, tf.location.y, tf.rotation.yaw, e.x, e.y) for tf, e in footprints)]

    def clear(self):
        """Suite.clear only removes cars and people. Any containers still in the world have to go too, or they stay in the road."""
        self.client.apply_batch_sync([self.carla.command.DestroyActor(a) for a in self.world.get_actors().filter(HIDER)])
        super().clear()

    def _beside_truck(self, truck_side):
        """(stop line of a lane going straight with a lane beside it on this side, that lane's stop waypoint): the
        variation's approach of those, counted round, or the first."""
        found = self._beside_truck_options(truck_side)
        return found[(self.variation.approach if self.variation else 0) % len(found)]

    def _beside_truck_options(self, truck_side):
        """Every (stop line, lane beside it) that suits a hidden hazard, in a fixed order."""
        found = []
        for stop, _ in sorted(self.controlled, key=lambda c: (round(c[0].transform.rotation.yaw / 10.0), c[0].lane_id)):
            beside = stop.get_right_lane() if truck_side == "right" else stop.get_left_lane()
            behind = stop.previous(APPROACH)
            if (beside is None or beside.lane_type != self.carla.LaneType.Driving or beside.lane_id * stop.lane_id <= 0
                    or not behind or behind[0].is_junction):
                continue
            if abs(locations_array(self.straight_path(stop, metres=40, every=5))[-1] - (stop.transform.location.x, stop.transform.location.y)).sum() > 15.0:
                found.append((stop, beside))
        if not found:
            raise RuntimeError(f"No lane at junction 189 goes straight with a lane on its {truck_side}.")
        return found

    def _park_truck(self, waypoint):
        if self.variation is not None and not self.variation.hidden:
            return  # nothing hides the hazard this time
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
        # Freezing one light freezes every light in the map, so one on the way here would stay as it was, red or not,
        # and the car would wait at it for the whole run. Any on the run-up are made green.
        run_up = [w.transform for back in range(0, int(APPROACH) + 1, 2) for w in stop.previous(float(back) + 0.5)[:1]]
        controlled = {light.id for _, light in self.controlled}
        for light in self.world.get_actors().filter("traffic.traffic_light"):
            if light.id in controlled:
                continue
            # Heading is compared with the run-up where it passes, as the run-up may turn through that junction
            if any(w.transform.location.distance(place.location) < RUN_UP_REACH and abs(yaw_difference(w.transform.rotation.yaw, place.rotation.yaw)) < 45.0
                   for w in light.get_stop_waypoints() for place in run_up):
                light.set_state(green)

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
        walk_speed = self.variation.walk_speed if self.variation else WALK_SPEED

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
                if state["launched_at"] is None and launch_now(my_index - nearest, CRUISE, walk, walk_speed, lead):
                    state["launched_at"] = tick.t
                    person.apply_control(self.carla.WalkerControl(direction=self.carla.Vector3D(x=-right.x, y=-right.y), speed=walk_speed))
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

    def _turner_options(self, stop, my_path):
        """Every (start, path, crossing) of an oncoming car whose left turn crosses the ML car's route in the junction."""
        found = []
        for candidate, _ in self.controlled:
            if yaw_class(candidate.transform.rotation.yaw, stop.transform.rotation.yaw) != "oncoming" or not candidate.previous(TURNER_BACK):
                continue
            begin = candidate.previous(TURNER_BACK)[0]
            their_points = self.straight_path(begin, every=1, turn=LEFT)
            crossing = paths_cross(my_path, locations_array(their_points))
            # Only where it crosses in the junction: a path can also meet the ML car's route somewhere on the way there
            if crossing is not None and crossing[0] >= APPROACH - CROSSING_SLACK:
                found.append((begin, their_points, crossing))
        return found

    def run_turner(self, lead):
        v = self.variation
        approaches = self._beside_truck_options("left")
        first = (v.approach if v else 0) % len(approaches)
        for stop, beside in approaches[first:] + approaches[:first]:  # the variation's approach, or the next that has a turner
            my_path = locations_array(self.straight_path(stop.previous(APPROACH)[0], every=1))
            found = self._turner_options(stop, my_path)
            if found:
                break
        else:
            raise RuntimeError("No oncoming left turn crosses the ML car's route in the junction.")
        self._green_for(stop)
        begin, their_points, (my_index, their_index) = found[(v.approach if v else 0) % len(found)]
        turner_speed = v.turner_speed if v else TURNER_SPEED
        # Timed for its left turn either way; one that carries straight on is a car the ML car mustn't stop for
        route = [(p.x, p.y) for p in (their_points if v is None or v.turner_turns else self.straight_path(begin, every=1))]
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
            if state["launched_at"] is None and launch_now(my_index - nearest, max(tick.speed, 4.0), their_index, turner_speed, lead):
                state["launched_at"] = tick.t
                self.console.line("SCENARIO", f"Oncoming car sets off, ML car {my_index - nearest} m from the crossing, t={tick.t:.1f} s")
            if state["launched_at"] is None:
                car.apply_control(self.carla.VehicleControl(hand_brake=True))
            else:
                steer, throttle, brake = follow_path(other, route, turner_speed)
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


def start_processor(mode, model, log_path, view=False, extra=()):
    log = open(log_path, "w", encoding="utf-8", errors="replace")  # noqa: SIM115
    process = subprocess.Popen([sys.executable, "-u", "scripts/central_gpu_processor.py", *processor_args(mode, model, view), *extra], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
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


