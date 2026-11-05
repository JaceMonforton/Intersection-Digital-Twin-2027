"""
How the ML car learns about the actors around it, whichever way it's getting them.

    truth   CARLA's list of every actor, which a real car could never have
    rsu     tracks from the roadside unit's detector, sent by the processor
    sight   only what a LiDAR on the car's own roof hits: the baseline the roadside unit has to beat

Each view gives the driver cars as (x, y, vx, vy, yaw) and pedestrians as (x, y, vx, vy), in CARLA's frame, plus
whether its information is fresh. A driver whose information isn't fresh must brake, because an empty road and a
dead link look the same. No carla or numpy imports here, so that it stays testable and light.
"""
import json
import socket
import struct

# A plain UDP socket, not Ditto, so the Windows localhost rule doesn't apply. It costs no tick of delay on one machine.
# shortcut: one machine only, with no loss handling; a real roadside unit needs a V2X radio link and signed messages.
ROADSIDE_ADDRESS = ("127.0.0.1", 47805)
STALE_AFTER = 0.5        # simulated seconds; older roadside tracks don't count
SIGHT_MEMORY = 1.0       # seconds a car remembers an actor its LiDAR has stopped hitting
# Roof LiDAR for the line-of-sight baseline, like a typical car-mounted unit
SIGHT_LIDAR = {"channels": "32", "points_per_second": "300000", "range": "70", "upper_fov": "10", "lower_fov": "-30"}
SIGHT_HEIGHT = 1.9       # m above the car's origin
SEMANTIC_POINT = struct.Struct("<ffffII")  # x, y, z, cos of the angle, id of the actor hit, semantic tag
MODES = ("truth", "rsu", "sight")


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
        self.receiver = receiver or RoadsideReceiver()

    def see(self, ego, cars, walkers, now):
        message = self.receiver.poll()
        if message is None or abs(now - message["t"]) > STALE_AFTER:
            return [], [], False
        return [tuple(car) for car in message["cars"]], [tuple(walker) for walker in message["walkers"]], True

    def close(self):
        self.receiver.close()


def visible_actors(raw_data):
    """The ids of the actors in a semantic LiDAR sweep. Id 0 means the point hit the map, not an actor."""
    ids = {hit[4] for hit in SEMANTIC_POINT.iter_unpack(raw_data[:len(raw_data) - len(raw_data) % SEMANTIC_POINT.size])}
    ids.discard(0)
    return ids


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
        self.sensor.listen(self._on_sweep)

    def _on_sweep(self, measurement):
        self.fresh_ids = visible_actors(bytes(measurement.raw_data))

    def see(self, ego, cars, walkers, now):
        for actor_id in self.fresh_ids:
            self.last_seen[actor_id] = now
        known = {actor_id for actor_id, seen in self.last_seen.items() if now - seen <= SIGHT_MEMORY}
        return [car for actor_id, car in cars.items() if actor_id in known], [walker for actor_id, walker in walkers.items() if actor_id in known], True

    def close(self):
        self.sensor.stop()
        self.sensor.destroy()


def make_view(mode, carla, world, car, sweep_seconds, receiver=None):
    """The view for a car. Pass a receiver to reuse its socket from one car to the next."""
    if mode == "rsu":
        return RoadsideView(receiver)
    if mode == "sight":
        return LineOfSightView(carla, world, car, sweep_seconds)
    return TruthView()
