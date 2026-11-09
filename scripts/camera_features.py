"""
The roadside camera detector's inputs, outputs and geometry, shared by training/record_camera.py, train_camera.py,
eval_camera.py and the processor. numpy only, so that it's testable without CARLA or onnxruntime.

A camera at each roadside mount (the LiDARs' poles) looks at the junction centre. The model finds where each actor
stands in the image: the point on the ground under its box centre. A fixed camera knows its height and pose, so that
pixel's ray meets the road at one place, which is the actor's position. That's how roadside monocular detectors
place things, and why they get worse with range: one pixel covers more road the further it is.

The model's output: [OUTPUT_CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH], a quarter of the image's resolution. Two heatmaps
of those ground points, from 0 to 1, then for the cell at a point: where in the cell it is (2, in cells), sin and cos
of the actor's yaw in the junction frame (2: a camera can tell a car's front from its back), and the log of its
length, width and height in metres (3).
"""
import math

import numpy as np
from detection_features import (
    GRID_HALF,
    MODEL_GROUND_Z,
    PEDESTRIAN,
    VEHICLE,
    _local_peaks,
)

IMAGE_WIDTH, IMAGE_HEIGHT, FOV = 640, 360, 90.0
INPUT_HEIGHT = 368      # the image padded at the bottom to a multiple of 16, for the model's four halvings
STRIDE = 4
OUTPUT_HEIGHT, OUTPUT_WIDTH = INPUT_HEIGHT // STRIDE, IMAGE_WIDTH // STRIDE
CLASSES = 2
OFFSET, YAW, SIZE = CLASSES, CLASSES + 2, CLASSES + 4
OUTPUT_CHANNELS = CLASSES + 7
CENTRE_SIGMA = {VEHICLE: 1.5, PEDESTRIAN: 1.0}  # output cells
MIN_PIXELS = 20         # of an actor in the image for it to count as seen; with fewer it's neither right nor wrong to find it
MAX_DETECTIONS = 100
MERGE_REACH = {VEHICLE: 2.5, PEDESTRIAN: 1.0}  # m: two cameras' detections closer than this are the same actor
DEPTH_PIXELS = 2.0      # pixels a ground point can be out, for depth_error: at 76 m from a pole the ML car was placed 2-5 m ahead

# One row per actor per camera image, as saved by training/record_camera.py. (u, v) is the pixel of the ground point
# under the box centre, the rest is the LiDAR labels' (x, y, z, length, width, height, yaw, class) in the junction
# frame, and pixels is how many of the image's pixels are that actor, from an instance segmentation camera.
LABEL_COLUMNS = ("u", "v", "x", "y", "z", "length", "width", "height", "yaw", "class", "pixels")
LABEL_SIZE = len(LABEL_COLUMNS)


def focal(width=IMAGE_WIDTH, fov=FOV):
    return width / (2.0 * math.tan(math.radians(fov) / 2.0))


def aim_at_centre(mount):
    """(pitch, yaw) in degrees that point a camera at mount (x, y, z from the junction centre, CARLA's frame) at the centre."""
    x, y, z = mount
    return -math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(-y, -x))


def project(points, world_to_camera, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, fov=FOV):
    """World points (N x 3) to pixels (N x 2), NaN for any behind the camera. world_to_camera is the camera
    transform's inverse matrix; in its frame x is forward, y right and z up, as everywhere in CARLA."""
    local = (np.asarray(world_to_camera) @ np.c_[points, np.ones(len(points))].T)[:3].T
    f = focal(width, fov)
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = np.c_[f * local[:, 1] / local[:, 0] + width / 2.0, -f * local[:, 2] / local[:, 0] + height / 2.0]
    pixels[local[:, 0] < 0.1] = np.nan
    return pixels


def pixels_to_ground(pixels, camera_to_world, ground_z, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, fov=FOV):
    """Where each pixel's ray meets the plane z = ground_z, as world (x, y), NaN for pixels at or above the horizon.
    camera_to_world is the camera transform's matrix."""
    pixels = np.asarray(pixels, float).reshape(-1, 2)
    f = focal(width, fov)
    rays = np.c_[np.ones(len(pixels)), (pixels[:, 0] - width / 2.0) / f, -(pixels[:, 1] - height / 2.0) / f]
    matrix = np.asarray(camera_to_world, float)
    directions, origin = rays @ matrix[:3, :3].T, matrix[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        reach = (ground_z - origin[2]) / directions[:, 2]
    ground = origin[:2] + reach[:, None] * directions[:, :2]
    ground[~(reach > 0)] = np.nan
    return ground


def instance_ids(raw, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """The actor id of each pixel of an instance segmentation image (BGRA): CARLA puts the id's low byte in G and
    its high byte in B. 0 is the map."""
    image = np.frombuffer(raw, np.uint8).reshape(height, width, 4)
    return image[:, :, 1].astype(np.int32) | (image[:, :, 0].astype(np.int32) << 8)


def pad_image(image):
    """An H x W x 3 uint8 image as the model's 3 x INPUT_HEIGHT x W float input, from 0 to 1."""
    out = np.zeros((3, INPUT_HEIGHT, image.shape[1]), np.float32)
    out[:, :image.shape[0]] = np.transpose(image, (2, 0, 1)) / 255.0
    return out


def _gaussian(row, column, sigma):
    reach = math.ceil(3 * sigma)
    rows = np.arange(max(row - reach, 0), min(row + reach + 1, OUTPUT_HEIGHT))
    columns = np.arange(max(column - reach, 0), min(column + reach + 1, OUTPUT_WIDTH))
    values = np.exp(-((rows[:, None] - row) ** 2 + (columns[None, :] - column) ** 2) / (2 * sigma ** 2))
    return rows, columns, values.astype(np.float32)


def encode_labels(labels, min_pixels=MIN_PIXELS):
    """What the model should output for one image's labels, as (target, heat_weight, centres), as in
    detection_features.encode_labels: the heat weight is 0 around an actor that's in view but barely shows."""
    target = np.zeros((OUTPUT_CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH), np.float32)
    heat_weight = np.ones((CLASSES, OUTPUT_HEIGHT, OUTPUT_WIDTH), np.float32)
    centres = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH), np.float32)
    for label in np.asarray(labels, np.float32).reshape(-1, LABEL_SIZE):
        u, v, kind, pixels = label[0] / STRIDE, label[1] / STRIDE, int(label[9]), label[10]
        if not (0 <= u < OUTPUT_WIDTH and 0 <= v < IMAGE_HEIGHT / STRIDE) or pixels <= 0:
            continue  # out of the picture, or hidden behind something: not there to be found
        row, column = int(v), int(u)
        rows, columns, values = _gaussian(row, column, CENTRE_SIGMA[kind])
        if pixels < min_pixels:
            heat_weight[kind][np.ix_(rows, columns)] = 0.0
            continue
        patch = target[kind][np.ix_(rows, columns)]
        target[kind][np.ix_(rows, columns)] = np.maximum(patch, values)
        target[OFFSET:OFFSET + 2, row, column] = (u - column, v - row)
        target[YAW:YAW + 2, row, column] = (math.sin(label[8]), math.cos(label[8]))
        target[SIZE:SIZE + 3, row, column] = np.log(np.maximum(label[5:8], 0.1))
        centres[row, column] = 1.0
    return target, heat_weight, centres


def decode(output, threshold, camera_to_world, center, max_detections=MAX_DETECTIONS):
    """One camera's output as float32 rows of (x, y, z, length, width, height, yaw, score, class) in the junction
    frame, as the LiDAR detector gives them. center is the junction's world (x, y, z); the road is taken as flat at
    its height, and anything placed outside the LiDAR grid is dropped."""
    thresholds = np.broadcast_to(threshold, (CLASSES,))
    found = []
    for kind in range(CLASSES):
        rows, columns = np.nonzero(_local_peaks(output[kind]) & (output[kind] >= thresholds[kind]))
        pixels = np.c_[(columns + output[OFFSET, rows, columns]) * STRIDE, (rows + output[OFFSET + 1, rows, columns]) * STRIDE]
        ground = pixels_to_ground(pixels, camera_to_world, center[2])
        x, y = ground[:, 0] - center[0], -(ground[:, 1] - center[1])
        sizes = np.exp(output[SIZE:SIZE + 3, rows, columns])
        yaw = np.arctan2(output[YAW, rows, columns], output[YAW + 1, rows, columns])
        found.append(np.column_stack((x, y, MODEL_GROUND_Z + sizes[2] / 2, sizes[0], sizes[1], sizes[2], yaw,
                                      output[kind, rows, columns], np.full(len(rows), kind))))
    detections = np.vstack(found).astype(np.float32)
    inside = np.isfinite(detections[:, 0]) & (np.abs(detections[:, :2]) < GRID_HALF).all(axis=1)
    detections = detections[inside]
    return detections[np.argsort(-detections[:, 7])[:max_detections]]


def depth_error(distance, height, pixels=DEPTH_PIXELS):
    """How far along the ground a place found from a camera at height can be out, at a horizontal distance from its
    pole, for a ground point off by pixels up or down: one pixel covers about distance^2 / (height * focal) of road."""
    return pixels * distance ** 2 / (height * focal())


def merge_cameras(parts):
    """The cameras' detections as one set: where two place the same class closer than MERGE_REACH, the higher score's
    box is kept. Cars and people are merged apart, so that a person by a car stays."""
    rows = np.vstack([np.asarray(p, np.float32).reshape(-1, 9) for p in parts] or [np.zeros((0, 9), np.float32)])
    kept = []
    for row in rows[np.argsort(-rows[:, 7])]:
        reach = MERGE_REACH[int(row[8])]
        if all(k[8] != row[8] or math.hypot(k[0] - row[0], k[1] - row[1]) >= reach for k in kept):
            kept.append(row)
    return np.array(kept, np.float32).reshape(-1, 9)
