"""
How the ML car learns about the actors around it, whichever way it's getting them.

    truth   CARLA's list of every actor, which a real car could never have
    rsu     tracks from the roadside unit's detector, sent by the processor, plus what the car's own roof LiDAR hits,
            as the roadside unit only covers 60 m around the junction and a car alone past that is blind
    sight   only what a LiDAR on the car's own roof hits: the baseline the roadside unit has to beat
    camera  as rsu, but the roadside tracks come from the roadside camera detector (scripts/camera_features.py),
            from a processor run with --detector camera. The car can't tell the difference, which is the point

Each view gives the driver cars as (x, y, vx, vy, yaw) and pedestrians as (x, y, vx, vy), in CARLA's frame, plus
whether its information is fresh. A driver whose information isn't fresh must brake, because an empty road and a
dead link look the same. No carla or numpy imports here, so that it stays testable and light.
"""
import json
import socket
import struct
import time
from collections import Counter

# A plain UDP socket, not Ditto, so the Windows localhost rule doesn't apply. It costs no tick of delay on one machine.
# shortcut: one machine only, with no loss handling; a real roadside unit needs a V2X radio link and signed messages.
ROADSIDE_ADDRESS = ("127.0.0.1", 47805)
STALE_AFTER = 0.5        # simulated seconds; older roadside tracks don't count
SIGHT_MEMORY = 1.0       # seconds a car remembers an actor its LiDAR has stopped hitting
SIGHT_MIN_POINTS = 3     # hits in a sweep to count as seen; the same as MIN_POINTS in detection_features.py
# Roof LiDAR for the line-of-sight baseline, like a typical car-mounted unit
SIGHT_LIDAR = {"channels": "32", "points_per_second": "300000", "range": "70", "upper_fov": "10", "lower_fov": "-30"}
SIGHT_HEIGHT = 1.9       # m above the car's origin
SEMANTIC_POINT = struct.Struct("<ffffII")  # x, y, z, cos of the angle, id of the actor hit, semantic tag
MODES = ("truth", "rsu", "sight", "camera", "onboard", "fused", "fused_camera")
ROADSIDE_MODES = ("rsu", "camera", "fused", "fused_camera")  # the ones that listen to the roadside link
# The ones that perceive with the car's own LiDAR detector rather than CARLA's truth, in onboard_perception.py.
# fused_camera fuses with the roadside camera detector's tracks instead of the roadside LiDAR's.
ONBOARD_MODES = ("onboard", "fused", "fused_camera")
SAME_CAR, SAME_WALKER = 3.0, 1.5   # m between a roadside track and what the car sees for them to be the same actor


class RoadsideSender:
    """Used by the processor. Telling a car must never get in the way of the clock, so send errors are ignored."""

    def __init__(self, address=ROADSIDE_ADDRESS):
        self.address = address
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, now, cars, walkers):
        message = {"t": now, "cars": [[round(v, 2) for v in car] for car in cars],
                   "walkers": [[round(v, 2) for v in walker] for walker in walkers]}
        try:
            self.socket.sendto(json.dumps(message).encode(), self.address)
        except OSError:
            pass


class RoadsideReceiver:
    def __init__(self, address=ROADSIDE_ADDRESS):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(address)
        self.socket.setblocking(False)
        self.latest = None

    def poll(self):
        """The newest message that's arrived, or the last one if nothing new has."""
        while True:
            try:
                self.latest = json.loads(self.socket.recv(65535))
            except BlockingIOError:
                return self.latest
            except (OSError, ValueError):  # Windows reports a closed peer here; a bad packet is just skipped
                return self.latest

    def close(self):
        self.socket.close()


LOCKSTEP_ADDRESS = ("127.0.0.1", 47806)
LOCKSTEP_WAIT = 10.0  # real s the processor waits for the driver before ticking anyway, and then stops waiting until it hears again;
# the driver never waits on the processor for anything but the tick itself, so a long wait can't deadlock
RELEASE = "off"  # what the driver sends when it stops acking, so the processor doesn't wait out a stage's end


class TickAck:
    """Used by the driver: tells the processor it has acted on a frame. Nobody listening is fine."""

    def __init__(self, address=LOCKSTEP_ADDRESS):
        self.address = address
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def release(self):
        """Says the driver has stopped acking, at a stage's end: the clock runs free until it acks again."""
        self.send(RELEASE)

    def send(self, frame):
        try:
            self.socket.sendto(str(frame).encode(), self.address)
        except OSError:
            pass

    def close(self):
        self.socket.close()


class TickGate:
    """Used by the processor with --lockstep: holds the next tick until the driver has acted on this one, so that a
    slow driver loop can't skip ticks. It only waits while a driver is acking, so setup and a missing driver can't
    stall the clock, and never longer than wait."""

    def __init__(self, address=LOCKSTEP_ADDRESS, wait=LOCKSTEP_WAIT):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(address)
        self.wait_for = wait
        self.acked = -1
        self.engaged = False
        self.timeouts = 0

    def _receive(self, timeout):
        """Reads one ack; False on a timeout, Windows reporting a closed peer, or a bad packet."""
        self.socket.settimeout(timeout) if timeout > 0 else self.socket.setblocking(False)
        try:
            data = self.socket.recv(64)
            self.engaged = data != RELEASE.encode()
            if self.engaged:
                self.acked = max(self.acked, int(data))
            return True
        except (OSError, ValueError):
            return False

    def wait(self, frame):
        """Returns once the driver has acted on frame, or at once if no driver is acking."""
        while self._receive(0.0):  # whatever has arrived meanwhile
            pass
        deadline = time.monotonic() + self.wait_for
        while self.engaged and self.acked < frame:
            left = deadline - time.monotonic()
            if left <= 0:
                self.engaged = False
                self.timeouts += 1
                print(f"[LOCKSTEP] No ack for frame {frame} within {self.wait_for:.1f} s; running free until one comes.", flush=True)
            else:
                self._receive(left)

    def close(self):
        self.socket.close()


class TruthView:
    name = "truth"

    def see(self, ego, cars, walkers, now):
        return list(cars.values()), list(walkers.values()), True

    def close(self):
        pass


class RoadsideView:
    """What the roadside unit reports. The processor leaves the ML car itself out, using the car's own telemetry."""
    name = "rsu"

    def __init__(self, receiver=None):
        self.owns_receiver = receiver is None  # a shared one outlives this view, for the next car
        self.receiver = receiver or RoadsideReceiver()

    def see(self, ego, cars, walkers, now):
        message = self.receiver.poll()
        if message is None or abs(now - message["t"]) > STALE_AFTER:
            return [], [], False
        return [tuple(car) for car in message["cars"]], [tuple(walker) for walker in message["walkers"]], True

    def close(self):
        if self.owns_receiver:
            self.receiver.close()


def visible_actors(raw_data, min_points=SIGHT_MIN_POINTS):
    """The ids of the actors with at least min_points in a semantic LiDAR sweep: the same bar as "seen" for the roadside
    unit, as a stray ray or two under a truck can't be told from noise. Id 0 means the point hit the map, not an actor."""
    hits = Counter(hit[4] for hit in SEMANTIC_POINT.iter_unpack(raw_data[:len(raw_data) - len(raw_data) % SEMANTIC_POINT.size]))
    return {actor_id for actor_id, count in hits.items() if actor_id != 0 and count >= min_points}


class LineOfSightView:
    """True positions, but only of actors a LiDAR on the car's roof has really hit. That's generous, since a real car
    would also have to detect them and measure their speed, so a win over this is a win over any on-board sensing."""
    name = "sight"

    def __init__(self, carla, world, car, sweep_seconds):
        bp = world.get_blueprint_library().find("sensor.lidar.ray_cast_semantic")
        for key, value in SIGHT_LIDAR.items():
            bp.set_attribute(key, value)
        bp.set_attribute("rotation_frequency", str(1.0 / sweep_seconds))
        place = carla.Transform(carla.Location(z=SIGHT_HEIGHT))
        self.sensor = world.spawn_actor(bp, place, attach_to=car)
        self.last_seen = {}
        self.fresh_ids = set()
        self.last_sweep = None  # (frame, sensor transform, raw bytes), kept for a window of the roof LiDAR
        self.sensor.listen(self._on_sweep)

    def _on_sweep(self, measurement):
        raw = bytes(measurement.raw_data)
        self.fresh_ids = visible_actors(raw)
        self.last_sweep = (measurement.frame, measurement.transform, raw)

    def see(self, ego, cars, walkers, now):
        for actor_id in self.fresh_ids:
            self.last_seen[actor_id] = now
        known = {actor_id for actor_id, seen in self.last_seen.items() if now - seen <= SIGHT_MEMORY}
        return [car for actor_id, car in cars.items() if actor_id in known], [walker for actor_id, walker in walkers.items() if actor_id in known], True

    def close(self):
        self.sensor.stop()
        self.sensor.destroy()


def add_unmatched(tracks, seen, reach):
    """tracks, plus anything in seen that no track is within reach of."""
    return tracks + [s for s in seen if all((s[0] - t[0]) ** 2 + (s[1] - t[1]) ** 2 > reach ** 2 for t in tracks)]


class RoadsideAndSightView:
    """The roadside unit adds to what the car sees for itself. Fresh only follows the roadside link, since that's the
    one that can die unnoticed."""
    name = "rsu"

    def __init__(self, roadside, own):
        self.roadside, self.own = roadside, own

    def see(self, ego, cars, walkers, now):
        road_cars, road_walkers, fresh = self.roadside.see(ego, cars, walkers, now)
        own_cars, own_walkers, _ = self.own.see(ego, cars, walkers, now)
        return add_unmatched(road_cars, own_cars, SAME_CAR), add_unmatched(road_walkers, own_walkers, SAME_WALKER), fresh

    def close(self):
        self.roadside.close()
        self.own.close()


def make_view(mode, carla, world, car, sweep_seconds, receiver=None):
    """The view for a car. Pass a receiver to reuse its socket from one car to the next."""
    if mode in ONBOARD_MODES:
        from onboard_perception import (
            make_onboard_view,  # numpy and the detector, which this module stays free of
        )

        return make_onboard_view(mode, carla, world, car, sweep_seconds, receiver=receiver)
    if mode in ROADSIDE_MODES:
        view = RoadsideAndSightView(RoadsideView(receiver), LineOfSightView(carla, world, car, sweep_seconds))
        view.name = mode
        return view
    if mode == "sight":
        return LineOfSightView(carla, world, car, sweep_seconds)
    return TruthView()
