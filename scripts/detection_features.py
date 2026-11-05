"""
The roadside detector's inputs and labels, so that recording, training and detecting all agree. It only
needs numpy, because training runs in a Python environment without CARLA or the processor's dependencies.

Everything here is in the model's frame: centred on the junction, x forward, y LEFT, road at
MODEL_GROUND_Z. See models/README.md.
"""
import math

import numpy as np

MODEL_GROUND_Z = -1.8   # z of the road in the model's frame
GRID_HALF = 60.0        # m from the junction centre to the edge of the grid
GRID_CELL = 0.25        # m per cell
GRID_SIZE = round(2 * GRID_HALF / GRID_CELL)
MAX_HEIGHT = 5.0        # m above the road; anything higher can't be a car or a person, and is mostly buildings
ABOVE_ROAD = 0.3        # m; points lower than this are the road itself
COUNT_SCALE = math.log1p(32.0)  # a cell with this many points reads as 1
MIN_POINTS = 3          # LiDAR points on an actor for it to count as seen

VEHICLE, PEDESTRIAN = 0, 1
# One row per actor, as saved by training/record_detection.py
LABEL_COLUMNS = ("x", "y", "z", "length", "width", "height", "yaw", "class", "points", "vx", "vy")
LABEL_SIZE = len(LABEL_COLUMNS)
POINT_SCALE = 100.0     # points are saved as whole centimetres, which is a third of the size

# The model's output: [9, OUTPUT_SIZE, OUTPUT_SIZE], half the grid's resolution, with rows along x and columns
# along y. Two heatmaps of actor centres, from 0 to 1, then for the cell at a centre: where in the cell it
# is (2, in cells), the log of its length, width and height in metres (3), and sin and cos of TWICE its yaw (2).
# Twice, because from a few roof points the front of a car looks like its back: which way it faces comes
# from its track, not from here.
CLASSES = 2
OUTPUT_STRIDE = 2
OUTPUT_SIZE = GRID_SIZE // OUTPUT_STRIDE
OUTPUT_CELL = GRID_CELL * OUTPUT_STRIDE
OUTPUT_CHANNELS = CLASSES + 7
OFFSET, SIZE, YAW = CLASSES, CLASSES + 2, CLASSES + 5
CENTRE_SIGMA = {VEHICLE: 2.0, PEDESTRIAN: 1.0}  # output cells
MAX_DETECTIONS = 150


def crop_to_grid(points, ground_z=MODEL_GROUND_Z):
    """The points of a cloud that the grid can use, which is what's worth keeping when recording."""
    up = points[:, 2] - ground_z
    keep = (np.abs(points[:, 0]) < GRID_HALF) & (np.abs(points[:, 1]) < GRID_HALF) & (up >= 0.0) & (up <= MAX_HEIGHT)
    return points[keep]


def rasterize(points, ground_z=MODEL_GROUND_Z):
    """A cloud as the model's float32 [2, GRID_SIZE, GRID_SIZE] input, with rows along x and columns along y.
    Channel 0 is how many points landed in the cell, and channel 1 is the height of the tallest. Both are
    from 0 to 1."""
    points = crop_to_grid(points, ground_z)
    row = ((points[:, 0] + GRID_HALF) / GRID_CELL).astype(np.int64)
    column = ((points[:, 1] + GRID_HALF) / GRID_CELL).astype(np.int64)
    grid = np.zeros((2, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    np.add.at(grid[0], (row, column), 1.0)
    np.maximum.at(grid[1], (row, column), (points[:, 2] - ground_z) / MAX_HEIGHT)
    grid[0] = np.minimum(np.log1p(grid[0]) / COUNT_SCALE, 1.0)
    return grid


def points_in_box(points, x, y, yaw, half_length, half_width, height, ground_z=MODEL_GROUND_Z):
    """How many points of a cloud land on a box standing on the road. x, y and yaw (radians,
    counter-clockwise) are in the model's frame too. The road under the box doesn't count."""
    dx, dy = points[:, 0] - x, points[:, 1] - y
    cos, sin = math.cos(yaw), math.sin(yaw)
    up = points[:, 2] - ground_z
    inside = (np.abs(dx * cos + dy * sin) <= half_length) & (np.abs(-dx * sin + dy * cos) <= half_width)
    return int(np.sum(inside & (up > ABOVE_ROAD) & (up <= height)))


def label_row(offset, half_size, yaw_degrees, velocity, kind, ground_z=MODEL_GROUND_Z):
    """An actor's label without its point count. offset is its box centre (x, y, z) from the junction
    centre, and velocity is (vx, vy), both in CARLA's frame, where y is right and yaw runs the other way."""
    x, y, z = offset
    return [x, -y, z + ground_z, 2 * half_size[0], 2 * half_size[1], 2 * half_size[2], -math.radians(yaw_degrees),
            kind, 0, velocity[0], -velocity[1]]


def count_points(row, points, ground_z=MODEL_GROUND_Z):
    """How many points of a cloud land on a label row."""
    return points_in_box(points, row[0], row[1], row[6], row[3] / 2, row[4] / 2, row[5], ground_z)


def load_sweep(path):
    """A recorded sweep's points as float32 (x, y, z), and its labels as in LABEL_COLUMNS."""
    with np.load(path) as data:
        return data["points"].astype(np.float32) / POINT_SCALE, data["labels"]


def _gaussian(row, column, sigma):
    """A patch of a Gaussian that peaks at 1 in the cell (row, column), as (rows, columns, values), cut to fit."""
    reach = math.ceil(3 * sigma)
    rows = np.arange(max(row - reach, 0), min(row + reach + 1, OUTPUT_SIZE))
    columns = np.arange(max(column - reach, 0), min(column + reach + 1, OUTPUT_SIZE))
    values = np.exp(-((rows[:, None] - row) ** 2 + (columns[None, :] - column) ** 2) / (2 * sigma ** 2))
    return rows, columns, values.astype(np.float32)


def encode_labels(labels, min_points=MIN_POINTS):
    """What the model should output for these labels, as (target, heat_weight, centres).
    target has the output's shape. heat_weight is 0 around an actor with too few points on it to be seen, since
    finding it there is neither right nor wrong, and 1 elsewhere. centres is 1 where the regressions count."""
    target = np.zeros((OUTPUT_CHANNELS, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
    heat_weight = np.ones((CLASSES, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
    centres = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
    for label in labels:
        kind = int(label[7])
        place_x, place_y = (label[0] + GRID_HALF) / OUTPUT_CELL, (label[1] + GRID_HALF) / OUTPUT_CELL
        row, column = int(place_x), int(place_y)
        if not (0 <= row < OUTPUT_SIZE and 0 <= column < OUTPUT_SIZE):
            continue
        rows, columns, values = _gaussian(row, column, CENTRE_SIGMA[kind])
        window = np.ix_(rows, columns)
        if label[8] < min_points:
            heat_weight[kind][window] = 0.0
            continue
        target[kind][window] = np.maximum(target[kind][window], values)
        target[OFFSET:OFFSET + 2, row, column] = place_x - row, place_y - column
        target[SIZE:SIZE + 3, row, column] = np.log(np.maximum(label[3:6], 0.05))
        target[YAW:YAW + 2, row, column] = math.sin(2 * label[6]), math.cos(2 * label[6])
        centres[row, column] = 1.0
    # A seen actor beside an unseen one still has to be found
    for kind in range(CLASSES):
        heat_weight[kind][target[kind] > 0.5] = 1.0
    return target, heat_weight, centres


def _local_peaks(heat):
    """Where a heatmap is the highest of its 3 x 3 neighbours."""
    padded = np.pad(heat, 1, constant_values=-1.0)
    neighbours = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    return heat >= neighbours.max(axis=(2, 3))


def decode(output, threshold, ground_z=MODEL_GROUND_Z, max_detections=MAX_DETECTIONS):
    """The model's output as float32 rows of (x, y, z, length, width, height, yaw, score, class). threshold is
    a score, or a score for each class. The yaw can be half a turn out: see the note on the output above."""
    thresholds = np.broadcast_to(threshold, (CLASSES,))
    found = []
    for kind in range(CLASSES):
        rows, columns = np.nonzero(_local_peaks(output[kind]) & (output[kind] >= thresholds[kind]))
        scores = output[kind, rows, columns]
        x = (rows + output[OFFSET, rows, columns]) * OUTPUT_CELL - GRID_HALF
        y = (columns + output[OFFSET + 1, rows, columns]) * OUTPUT_CELL - GRID_HALF
        sizes = np.exp(output[SIZE:SIZE + 3, rows, columns])
        yaw = 0.5 * np.arctan2(output[YAW, rows, columns], output[YAW + 1, rows, columns])
        found.append(np.column_stack((x, y, ground_z + sizes[2] / 2, sizes[0], sizes[1], sizes[2], yaw, scores,
                                      np.full(len(rows), kind))))
    detections = np.vstack(found).astype(np.float32)
    return detections[np.argsort(-detections[:, 7])[:max_detections]]
