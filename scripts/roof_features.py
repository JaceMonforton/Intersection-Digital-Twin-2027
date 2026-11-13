"""
The ML car's own roof LiDAR, as a sensor with its own detector, the way a Waymo-style car perceives: a sweep from the
roof is turned into the detector's frame centred on the car, boxes come out of a model trained on roof sweeps, and
those are tracked and fused with the roadside unit's tracks. This file holds what recording, training and the car
must agree on. numpy only, so that it's testable without CARLA.

The model's frame is detection_features'. Here it's centred on the roof sensor rather than the junction: x is the
car's forward, y its LEFT, and the road under the car at MODEL_GROUND_Z, so the same rasterize, encode_labels and
decode, and the same network, work on it.
"""
import math

import numpy as np
from detection_features import GRID_HALF, MODEL_GROUND_Z, PEDESTRIAN, VEHICLE, label_row

# A typical car-mounted spinning LiDAR. One sweep per 0.1 s tick, with no points silently dropped.
ROOF_LIDAR = {"channels": "64", "points_per_second": "600000", "range": "70", "upper_fov": "10", "lower_fov": "-30",
              "rotation_frequency": "10", "dropoff_general_rate": "0.0"}
ROOF_HEIGHT = 1.9       # m above the car's origin, which sits on the road
OWN_BODY = (2.6, 1.2)   # m: half the length and width of the ML car around the sensor, whose own points are cut out
ROOF_RAW_DIR_NAME = "roof"  # under data/detection


def sweep_to_model(raw_points, height=ROOF_HEIGHT):
    """A roof sweep as (x, y, z) float32 in the model's frame. raw_points are the sensor's (x, y, z, intensity) in
    CARLA's frame, y right; the car's own body is cut out, since it's always there and isn't anything to find."""
    points = np.asarray(raw_points, np.float32).reshape(-1, 4)[:, :3]
    out = np.c_[points[:, 0], -points[:, 1], points[:, 2] + height + MODEL_GROUND_Z].astype(np.float32)
    own = (np.abs(out[:, 0]) < OWN_BODY[0]) & (np.abs(out[:, 1]) < OWN_BODY[1])
    return out[~own]


def world_to_sensor(dx, dy, yaw_degrees):
    """A world offset (CARLA's frame, y right) turned into the sensor's axes, still y right."""
    yaw = math.radians(yaw_degrees)
    cos, sin = math.cos(yaw), math.sin(yaw)
    return dx * cos + dy * sin, -dx * sin + dy * cos


def actor_label(sensor, actor, height=ROOF_HEIGHT):
    """A label row for an actor around the roof sensor, or None if it's off the grid. sensor is (x, y, z, yaw) of
    the sensor in the world. actor is (x, y, z of its box centre, half length, half width, half height, yaw, vx, vy,
    is_walker), all in CARLA's world frame."""
    sx, sy, sz, syaw = sensor
    x, y, z, hl, hw, hh, yaw, vx, vy, walker = actor
    ax, ay = world_to_sensor(x - sx, y - sy, syaw)
    if max(abs(ax), abs(ay)) >= GRID_HALF:
        return None
    road = sz - height  # the road under the car
    velocity = world_to_sensor(vx, vy, syaw)
    return label_row((ax, ay, z - road), (hl, hw, hh), yaw - syaw, velocity, PEDESTRIAN if walker else VEHICLE)


def model_to_world(row, sensor):
    """(x, y) of a detection in the model's frame back in CARLA's world frame, from the sensor's (x, y, z, yaw)."""
    sx, sy, _, syaw = sensor
    yaw = math.radians(syaw)
    cos, sin = math.cos(yaw), math.sin(yaw)
    lx, ly = float(row[0]), -float(row[1])  # back to y right
    return sx + lx * cos - ly * sin, sy + lx * sin + ly * cos
