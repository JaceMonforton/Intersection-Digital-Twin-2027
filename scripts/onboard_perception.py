"""
What the ML car perceives with its own sensors, and how that's fused with the roadside unit, the way a Waymo-style
car decides from what it measures rather than from the simulator's truth:

    onboard   the car's roof LiDAR through its own detector (models/roof_detector.candidate.onnx, trained by
              training/train_detection.py on training/record_roof.py's sweeps), tracked over time for velocity
    fused     those tracks fused with the roadside unit's, which arrive over the link with a delay and can stop:
              where both see an actor, each counts for more the closer its sensor is; what only one sees is kept;
              and with the link down the car drives on its own LiDAR rather than braking blind

Each view gives cars as (x, y, vx, vy, yaw) and pedestrians as (x, y, vx, vy) in CARLA's frame, plus whether its
information is fresh, like the views in roadside_link.py. carla and onnxruntime are imported lazily.
"""
import math
from pathlib import Path

import numpy as np
from central_gpu_processor import (
    CAR_CONFIRM,
    LIDAR_MOUNTS,
    WALKER_BETA,
    WALKER_CONFIRM,
    CentroidTracker,
)
from detection_features import PEDESTRIAN, decode, rasterize
from roadside_link import SAME_CAR, SAME_WALKER, STALE_AFTER, RoadsideView
from roof_features import ROOF_HEIGHT, ROOF_LIDAR, model_to_world, sweep_to_model

ROOF_MODEL = Path(__file__).resolve().parent.parent / "models" / "roof_detector.candidate.onnx"
ROOF_THRESHOLD = (0.506, 0.542)  # vehicle, pedestrian: the best F1 on seed 48 (training/eval_detection.py), kept apart from seed 49
PERSON_LENGTH = 1.2          # m; a box shorter than this is tracked as a person, whatever class scored higher
# How far out a position is, by range from the sensor that measured it: about the detectors' median error near and far
POSITION_ERROR = (0.15, 0.01)  # m at the sensor, and m per m of range
JUNCTION = (-47.8, 20.4)       # junction 189's centre, as the processor finds it; the roadside poles stand around it


def sensor_error(distance):
    return POSITION_ERROR[0] + POSITION_ERROR[1] * distance


def fuse_tracks(own, road, own_from, road_from, reach):
    """One list from the car's tracks and the roadside unit's. Each is (x, y, vx, vy, ...). own_from is where the car's
    sensor is, and road_from the roadside poles, as (x, y). Where one of each is within reach, they're one actor,
    averaged by inverse variance; everything else is kept as it is."""
    def weight(track, origins):
        return 1.0 / sensor_error(min(math.hypot(track[0] - ox, track[1] - oy) for ox, oy in origins)) ** 2

    pairs = sorted(((math.hypot(o[0] - r[0], o[1] - r[1]), i, j) for i, o in enumerate(own) for j, r in enumerate(road)))
    used_own, used_road, fused = set(), set(), []
    for gap, i, j in pairs:
        if gap > reach:
            break
        if i in used_own or j in used_road:
            continue
        used_own.add(i)
        used_road.add(j)
        o, r = own[i], road[j]
        wo, wr = weight(o, [own_from]), weight(r, road_from)
        mixed = tuple((wo * a + wr * b) / (wo + wr) for a, b in zip(o[:4], r[:4]))
        fused.append(mixed + tuple(o[4:] or r[4:]))  # the heading from the car's own track, which it measures every sweep
    fused += [o for i, o in enumerate(own) if i not in used_own]
    fused += [r for j, r in enumerate(road) if j not in used_road]
    return fused


def detections_to_world(rows, sensor):
    """Decoded rows (x, y, z, l, w, h, yaw, score, class) to (cars, people), each (x, y, yaw in degrees) in CARLA's frame."""
    cars, people = [], []
    for row in rows:
        x, y = model_to_world(row, sensor)
        yaw = sensor[3] - math.degrees(float(row[6]))
        (people if int(row[8]) == PEDESTRIAN or row[3] < PERSON_LENGTH else cars).append((x, y, yaw))
    return cars, people


class OnboardView:
    """The car's own roof LiDAR, detector and tracker."""
    name = "onboard"

    def __init__(self, carla, world, car, sweep_seconds, model=ROOF_MODEL, threshold=ROOF_THRESHOLD):
        import onnxruntime

        if not Path(model).exists():
            raise FileNotFoundError(f"{model} not found: train it with training/train_detection.py --data data/detection/roof.")
        self.session = onnxruntime.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = threshold
        bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
        for key, value in {**ROOF_LIDAR, "rotation_frequency": str(1.0 / sweep_seconds)}.items():
            bp.set_attribute(key, value)
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(z=ROOF_HEIGHT)), attach_to=car)
        self.latest, self.done_frame = None, None
        self.cars, self.walkers = [], []
        self.car_tracker = CentroidTracker(confirm=CAR_CONFIRM, name="own")
        self.walker_tracker = CentroidTracker(confirm=WALKER_CONFIRM, name="own_walker", beta=WALKER_BETA)
        self.last_sweep = None  # (frame, sensor transform, raw bytes), for a window of the roof LiDAR
        self.sensor.listen(self._on_sweep)

    def _on_sweep(self, measurement):
        self.latest = measurement

    @property
    def origin(self):
        m = self.latest
        return (m.transform.location.x, m.transform.location.y) if m is not None else (0.0, 0.0)

    def see(self, ego, cars, walkers, now):
        m = self.latest
        if m is None or now - m.timestamp > STALE_AFTER:
            return [], [], False
        if m.frame != self.done_frame:
            self.done_frame = m.frame
            raw = bytes(m.raw_data)
            self.last_sweep = (m.frame, m.transform, raw)
            tf = m.transform
            sensor = (tf.location.x, tf.location.y, tf.location.z, tf.rotation.yaw)
            grid = rasterize(sweep_to_model(np.frombuffer(raw, dtype=np.float32)))
            output = self.session.run(None, {self.input_name: grid[None]})[0][0]
            found_cars, found_people = detections_to_world(decode(output, self.threshold), sensor)
            tracked = self.car_tracker.update(found_cars, m.timestamp)
            self.cars = [(t.x, t.y, t.vx, t.vy, t.yaw if t.yaw is not None else math.degrees(math.atan2(t.vy, t.vx))) for t in tracked]
            self.walkers = [(t.x, t.y, t.vx, t.vy) for t in self.walker_tracker.update([p[:2] for p in found_people], m.timestamp)]
        return list(self.cars), list(self.walkers), True

    def close(self):
        self.sensor.stop()
        self.sensor.destroy()


class FusedView:
    """The roadside unit's tracks fused with the car's own. Fresh follows the car's own LiDAR: the roadside link may
    drop, and then the car carries on with what it sees itself."""
    name = "fused"

    def __init__(self, roadside, own, junction):
        self.roadside, self.own = roadside, own
        self.poles = [(junction[0] + x, junction[1] + y) for x, y, _ in LIDAR_MOUNTS]
        self.link_up = False

    def see(self, ego, cars, walkers, now):
        own_cars, own_walkers, fresh = self.own.see(ego, cars, walkers, now)
        road_cars, road_walkers, self.link_up = self.roadside.see(ego, cars, walkers, now)
        origin = self.own.origin
        return (fuse_tracks(own_cars, road_cars, origin, self.poles, SAME_CAR),
                fuse_tracks(own_walkers, road_walkers, origin, self.poles, SAME_WALKER), fresh)

    @property
    def last_sweep(self):
        return self.own.last_sweep

    def close(self):
        self.roadside.close()
        self.own.close()


def make_onboard_view(mode, carla, world, car, sweep_seconds, junction=JUNCTION, receiver=None):
    own = OnboardView(carla, world, car, sweep_seconds)
    if mode == "onboard":
        return own
    view = FusedView(RoadsideView(receiver), own, junction)
    view.name = mode
    return view
