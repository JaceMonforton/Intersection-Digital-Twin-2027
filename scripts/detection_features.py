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
