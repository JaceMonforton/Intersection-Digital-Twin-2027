"""
Where a car is, read one way and mapped one way, for everything that places a car: the processor, which brakes on
it, and the twin mirror (ditto_carla_agent.py), which shows it in CARLA. Two mappings would put the brake's car and
the visible car in different places. Standard library only.

Two kinds of telemetry come in:

    pose        features/pose/properties: x, y in lab metres, heading in degrees and speed in m/s, plus t and source.
                The lab floor is right-handed: x along a chosen axis, y to its left, heading counter-clockwise. This is
                what scripts/fake_lab_car.py sends, and what the overhead camera will.
    the rest    location (+ velocity, or orientation.yaw) and kinematics (speed in km/h, from hardware/kuksa_to_ditto.py),
                which are already in CARLA's frame, as the README has always said, and pass through unchanged.

A pose wins when a Thing carries both. A lab pose reaches CARLA through LabFrame, in this order: scale, rotate by
rotation_deg, flip y (CARLA is left-handed, y to the right, and its yaw turns clockwise), then offset in CARLA metres.
Velocity is scaled, rotated and flipped but never offset, and speed scales with distance, so a 1 m/s car at x10 is a
10 m/s twin and every time-to-collision is unchanged. The numbers come from config/lab_frame.json; a missing or
malformed file fails loudly rather than placing the brake's car somewhere it isn't.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

CALIBRATION = Path(__file__).resolve().parent.parent / "config" / "lab_frame.json"
KMH = 1 / 3.6  # kinematics speed is km/h, as hardware/kuksa_to_ditto.py forwards VSS Vehicle.Speed


@dataclass(frozen=True)
class LabFrame:
    scale: float
    rotation_deg: float
    flip_y: bool
    offset_x: float
    offset_y: float
    calibrated: bool = False

    def _turn(self, x, y):
        a = math.radians(self.rotation_deg)
        x, y = x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)
        return x, -y if self.flip_y else y

    def to_carla_point(self, x, y):
        cx, cy = self._turn(x * self.scale, y * self.scale)
        return cx + self.offset_x, cy + self.offset_y

    def to_carla_velocity(self, vx, vy):
        return self._turn(vx * self.scale, vy * self.scale)

    def to_carla_speed(self, speed):
        return speed * self.scale

    def to_carla_heading(self, heading):
        """A lab heading as CARLA yaw, in degrees from -180 to 180."""
        yaw = heading + self.rotation_deg
        if self.flip_y:
            yaw = -yaw
        return (yaw + 180.0) % 360.0 - 180.0

    def to_lab_point(self, cx, cy):
        x, y = cx - self.offset_x, cy - self.offset_y
        y = -y if self.flip_y else y
        a = math.radians(-self.rotation_deg)
        return ((x * math.cos(a) - y * math.sin(a)) / self.scale, (x * math.sin(a) + y * math.cos(a)) / self.scale)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load(path=CALIBRATION):
    """The LabFrame in a calibration file. Raises FileNotFoundError if it's missing and ValueError if it's malformed."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        fields = json.loads(text)
    except ValueError as e:
        raise ValueError(f"{path} isn't valid JSON: {e}") from e
    if not isinstance(fields, dict):
        raise ValueError(f"{path} must hold one JSON object")  # noqa: TRY004  # a malformed file, not a caller's mistake
    for key in ("scale", "rotation_deg", "offset_x", "offset_y"):
        if not _number(fields.get(key)):
            raise ValueError(f"{path}: {key} must be a finite number")
    for key in ("flip_y", "calibrated"):
        if not isinstance(fields.get(key), bool):
            raise ValueError(f"{path}: {key} must be true or false")  # noqa: TRY004  # likewise
    if fields["scale"] <= 0:
        raise ValueError(f"{path}: scale must be above 0")
    return LabFrame(float(fields["scale"]), float(fields["rotation_deg"]), fields["flip_y"], float(fields["offset_x"]),
                    float(fields["offset_y"]), fields["calibrated"])


def _signal(thing, group, field):
    """A value from a Ditto thing in any layout seen so far: features.<group>.properties, features.<group> or
    attributes.<group>."""
    for path in (("features", group, "properties", field), ("features", group, field), ("attributes", group, field)):
        node = thing
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if node is not None:
            return node
    return None


def read_pose(thing):
    """The pose a Ditto thing reports, in its own frame, or None without a usable one: x, y, z, heading (degrees),
    vx, vy, speed (m/s), t, source, and frame, which is "lab" or "carla"."""
    if all(_number(_signal(thing, "pose", k)) for k in ("x", "y", "heading", "speed")):
        heading, speed = float(_signal(thing, "pose", "heading")), float(_signal(thing, "pose", "speed"))
        t = _signal(thing, "pose", "t")
        return {"x": float(_signal(thing, "pose", "x")), "y": float(_signal(thing, "pose", "y")), "z": None, "heading": heading,
                "vx": speed * math.cos(math.radians(heading)), "vy": speed * math.sin(math.radians(heading)), "speed": speed,
                "t": float(t) if _number(t) else None, "source": _signal(thing, "pose", "source"), "frame": "lab"}
    if any(_signal(thing, "pose", k) is not None for k in ("x", "y")):
        return None  # a pose that's there but broken isn't quietly replaced by an older, staler format
    x, y = _signal(thing, "location", "x"), _signal(thing, "location", "y")
    if not (_number(x) and _number(y)):
        x, y = _signal(thing, "kinematics", "x"), _signal(thing, "kinematics", "y")
    if not (_number(x) and _number(y)):
        return None
    heading = next((float(v) for v in (_signal(thing, "orientation", "yaw"), _signal(thing, "location", "yaw"),
                                        _signal(thing, "kinematics", "heading")) if _number(v)), None)
    vx, vy = _signal(thing, "velocity", "vx"), _signal(thing, "velocity", "vy")
    if _number(vx) and _number(vy):
        vx, vy = float(vx), float(vy)
    else:
        speed = _signal(thing, "kinematics", "speed")
        speed = float(speed) * KMH if _number(speed) else 0.0
        vx, vy = speed * math.cos(math.radians(heading or 0.0)), speed * math.sin(math.radians(heading or 0.0))
    if heading is None and (vx or vy):
        heading = math.degrees(math.atan2(vy, vx))
    z = _signal(thing, "location", "z")
    return {"x": float(x), "y": float(y), "z": float(z) if _number(z) else None, "heading": heading, "vx": vx, "vy": vy,
            "speed": math.hypot(vx, vy), "t": None, "source": None, "frame": "carla"}


def carla_pose(thing, frame):
    """read_pose's pose in CARLA's frame, with heading as CARLA yaw, or None without a usable pose."""
    pose = read_pose(thing)
    if pose is None or pose["frame"] == "carla":
        if pose is not None:
            pose["yaw"] = pose["heading"]
        return pose
    x, y = frame.to_carla_point(pose["x"], pose["y"])
    vx, vy = frame.to_carla_velocity(pose["vx"], pose["vy"])
    return {**pose, "x": x, "y": y, "vx": vx, "vy": vy, "speed": frame.to_carla_speed(pose["speed"]),
            "yaw": frame.to_carla_heading(pose["heading"]), "frame": "carla"}
