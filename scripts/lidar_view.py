"""
A live top-down window of what the two roadside LiDARs are really producing, to show that they're working.

The processor hands over each LiDAR's own sweep, the detector's boxes and CARLA's true boxes through shared memory, so
this window can't hold up the clock or the safety gate, and needs neither carla nor the model. The true boxes are only
for comparing by eye: nothing else reads them from here. Everything shown is measured, and goes red or says so if it stops.

    python run_simulation.py --roadside --lidar-view
    python scripts/lidar_view.py         # alongside a processor started with --detector onnx --lidar-view

Press 1 or 2 to hide a LiDAR's points, which shows what the other one covers. It only changes this window.
AI-driven pedestrians (controller.ai.walker) stay as dim true boxes, because LiDAR rays pass straight through them.
"""
import math
import signal
import struct
import sys
import time
from collections import deque
from multiprocessing import shared_memory

import numpy as np
from detection_features import (
    ABOVE_ROAD,
    GRID_HALF,
    LABEL_SIZE,
    MIN_POINTS,
    MODEL_GROUND_Z,
    PEDESTRIAN,
    POINT_SCALE,
    VEHICLE,
    count_points,
    crop_to_grid,
)

FEED_NAME = "intersection_lidar_view"
LIDARS = 2
MAX_POINTS = 60000      # per LiDAR; each makes 30,000 a sweep
MAX_DETECTIONS = 300
MAX_TRUE = 500
DETECTION_SIZE = 9      # as detection_features.decode gives them
# sequence, frame, simulated time, then per LiDAR: points, the frame of its last sweep, sweeps so far; then box counts
HEADER = struct.Struct(f"<Qqd{LIDARS}I{LIDARS}q{LIDARS}QII")
BODY_START = 128        # leaves the header room, and the arrays aligned

STOPPED_AFTER = 2.0     # wall-clock seconds without a new sweep before the window says so
RATE_WINDOW = 2.0       # wall-clock seconds to measure sweeps per second over
TRUE_BOX_GROWTH = 0.35  # m; so a true box shows round a detected one on top of it
FPS = 20
PANEL_WIDTH = 300
MAX_WINDOW = 880
COLOURS = ((70, 150, 240), (240, 130, 60))
TRUE_SEEN, TRUE_UNSEEN = (235, 235, 235), (95, 95, 100)
DETECTED_CAR, DETECTED_PERSON = (70, 230, 120), (240, 90, 220)
BAD, TEXT, TITLE = (240, 90, 80), (205, 205, 205), (240, 240, 240)
BACKGROUND, PANEL, RING = (16, 16, 18), (28, 28, 32), (45, 45, 50)


class SweepFeed:
    """A block of shared memory holding the newest sweeps. One processor writes it and a window reads it. The sequence
    number is odd during a write, and a read that sees it change is thrown away, so nothing ever waits on anything."""

    def __init__(self, memory):
        self.memory = memory
        offset = BODY_START
        self.mounts = np.ndarray((LIDARS, 2), np.float32, memory.buf, offset)
        offset += self.mounts.nbytes
        self.points = np.ndarray((LIDARS, MAX_POINTS, 3), np.int16, memory.buf, offset)
        offset += self.points.nbytes + self.points.nbytes % 4
        self.detections = np.ndarray((MAX_DETECTIONS, DETECTION_SIZE), np.float32, memory.buf, offset)
        offset += self.detections.nbytes
        self.true_boxes = np.ndarray((MAX_TRUE, LABEL_SIZE), np.float32, memory.buf, offset)
        self.seen = 0
        self.last_frames = [-1] * LIDARS
        self.totals = [0] * LIDARS

    @staticmethod
    def size():
        points = LIDARS * MAX_POINTS * 3 * 2
        return BODY_START + LIDARS * 2 * 4 + points + points % 4 + (MAX_DETECTIONS * DETECTION_SIZE + MAX_TRUE * LABEL_SIZE) * 4

    @classmethod
    def create(cls):
        try:
            memory = shared_memory.SharedMemory(FEED_NAME, create=True, size=cls.size())
        except FileExistsError:  # a window from the last run is still open, so carry on with its block
            memory = shared_memory.SharedMemory(FEED_NAME)
        return cls(memory)

    @classmethod
    def attach(cls):
        """None until a processor has made the feed."""
        try:
            return cls(shared_memory.SharedMemory(FEED_NAME))
        except FileNotFoundError:
            return None

    def _sequence(self):
        return struct.unpack_from("<Q", self.memory.buf, 0)[0]

    def write(self, frame, now, mounts, sweeps, detections, true_boxes):
        """sweeps has one entry per LiDAR in the model's frame, or None for one that didn't arrive this tick."""
        sequence = self._sequence() | 1
        struct.pack_into("<Q", self.memory.buf, 0, sequence)
        counts = []
        for index, sweep in enumerate(sweeps):
            if sweep is None:
                counts.append(0)
                continue
            kept = np.round(crop_to_grid(sweep)[:MAX_POINTS, :3] * POINT_SCALE).astype(np.int16)
            self.points[index, :len(kept)] = kept
            counts.append(len(kept))
            self.last_frames[index] = frame
            self.totals[index] += 1
        self.mounts[:] = mounts
        detections = np.asarray(detections, np.float32).reshape(-1, DETECTION_SIZE)[:MAX_DETECTIONS]
        true_boxes = np.asarray(true_boxes, np.float32).reshape(-1, LABEL_SIZE)[:MAX_TRUE]
        self.detections[:len(detections)] = detections
        self.true_boxes[:len(true_boxes)] = true_boxes
        HEADER.pack_into(self.memory.buf, 0, sequence, frame, now, *counts, *self.last_frames, *self.totals, len(detections), len(true_boxes))
        struct.pack_into("<Q", self.memory.buf, 0, sequence + 1)

    def read(self):
        """The newest state as a dict, or None if there's nothing new or it changed while being read."""
        sequence = self._sequence()
        if sequence == 0 or sequence % 2 or sequence == self.seen:
            return None
        fields = HEADER.unpack_from(self.memory.buf, 0)
        counts, last_frames, totals = fields[3:3 + LIDARS], fields[3 + LIDARS:3 + 2 * LIDARS], fields[3 + 2 * LIDARS:3 + 3 * LIDARS]
        found, true_count = fields[-2:]
        state = {"frame": fields[1], "time": fields[2], "last_frames": last_frames, "totals": totals, "mounts": self.mounts.copy(),
                 "sweeps": [self.points[i, :counts[i]].astype(np.float32) / POINT_SCALE for i in range(LIDARS)],
                 "detections": self.detections[:found].copy(), "true_boxes": self.true_boxes[:true_count].copy()}
        if self._sequence() != sequence:
            return None
        self.seen = sequence
        return state

    def close(self):
        self.mounts = self.points = self.detections = self.true_boxes = None  # or close refuses while they point at it
        self.memory.close()

    def destroy(self):
        self.close()
        self.memory.unlink()


def to_screen(x, y, size):
    """Model-frame metres (x forward, y left) to pixels, with x up the window and y to the left of it."""
    scale = size / (2 * GRID_HALF)
    return size / 2 - np.asarray(y) * scale, size / 2 - np.asarray(x) * scale


def box_corners(row, grow=0.0):
    """The four corners of a box row, in the model's frame. Labels and detections share their first seven columns."""
    x, y, half_length, half_width, yaw = float(row[0]), float(row[1]), float(row[3]) / 2 + grow, float(row[4]) / 2 + grow, float(row[6])
    cos, sin = math.cos(yaw), math.sin(yaw)
    return [(x + a * cos - b * sin, y + a * sin + b * cos)
            for a, b in ((half_length, half_width), (half_length, -half_width), (-half_length, -half_width), (-half_length, half_width))]


def points_on_boxes(true_boxes, sweeps):
    """How many points of the sweeps land on each true box. Road points can't, so they're left out, which is far quicker."""
    merged = np.vstack(sweeps)
    raised = merged[merged[:, 2] - MODEL_GROUND_Z > ABOVE_ROAD]
    return np.array([count_points(row, raised) for row in true_boxes], dtype=np.int64)


class SweepRate:
    """Sweeps per second of wall-clock time, from a LiDAR's running total."""

    def __init__(self, window=RATE_WINDOW):
        self.window = window
        self.samples = deque()

    def add(self, now, total):
        self.samples.append((now, total))
        while now - self.samples[0][0] > self.window:
            self.samples.popleft()

    def per_second(self):
        if len(self.samples) < 2 or self.samples[-1][0] <= self.samples[0][0]:
            return None
        return (self.samples[-1][1] - self.samples[0][1]) / (self.samples[-1][0] - self.samples[0][0])


class Window:
    def __init__(self, pygame):
        self.pygame = pygame
        pygame.init()
        self.size = min(MAX_WINDOW, pygame.display.Info().current_h - 120)
        self.screen = pygame.display.set_mode((self.size + PANEL_WIDTH, self.size))
        pygame.display.set_caption("Roadside LiDARs")
        self.small, self.medium, self.large = (pygame.font.SysFont("segoeui", size) for size in (14, 17, 22))
        self.clock = pygame.time.Clock()
        self.feed = None
        self.state = None
        self.seen_counts = None
        self.updated = None
        self.hidden = [False] * LIDARS
        self.rates = [SweepRate() for _ in range(LIDARS)]

    def poll(self, now):
        self.feed = self.feed or SweepFeed.attach()
        state = self.feed.read() if self.feed else None
        if state is not None:
            self.state, self.updated = state, now
            self.seen_counts = points_on_boxes(state["true_boxes"], state["sweeps"])
            for rate, total in zip(self.rates, state["totals"]):
                rate.add(now, total)

    def draw(self, now):
        self.screen.fill(BACKGROUND)
        self.pygame.draw.rect(self.screen, PANEL, (self.size, 0, PANEL_WIDTH, self.size))
        self.cursor = 22
        self._line("Roadside LiDARs", self.large, TITLE, 36)
        if self.state is None:
            self._line("Waiting for the processor.", colour=BAD)
            self._line("It needs --detector onnx")
            self._line("and --lidar-view.")
        else:
            self._draw_scene()
            self._draw_panel(now)
        self.pygame.display.flip()

    def _place(self, x, y):
        across, down = to_screen(x, y, self.size)
        return float(across), float(down)

    def _draw_scene(self):
        for metres in (30, GRID_HALF):
            self.pygame.draw.circle(self.screen, RING, self._place(0, 0), metres * self.size / (2 * GRID_HALF), 1)
        pixels = self.pygame.surfarray.pixels3d(self.screen)
        for index, sweep in enumerate(self.state["sweeps"]):
            if self.hidden[index] or not len(sweep):
                continue
            across, down = to_screen(sweep[:, 0], sweep[:, 1], self.size)
            columns, rows = np.clip(across.astype(int), 0, self.size - 1), np.clip(down.astype(int), 0, self.size - 1)
            raised = sweep[:, 2] - MODEL_GROUND_Z > ABOVE_ROAD
            pixels[columns[~raised], rows[~raised]] = tuple(c // 3 for c in COLOURS[index])
            pixels[columns[raised], rows[raised]] = COLOURS[index]
        del pixels  # or the surface stays locked
        for row, points in zip(self.state["true_boxes"], self.seen_counts):
            self._box(row, TRUE_SEEN if points >= MIN_POINTS else TRUE_UNSEEN, 1, TRUE_BOX_GROWTH)
        for row in self.state["detections"]:
            self._box(row, DETECTED_PERSON if int(row[8]) == PEDESTRIAN else DETECTED_CAR, 2)
        for index, mount in enumerate(self.state["mounts"]):
            self.pygame.draw.circle(self.screen, COLOURS[index], self._place(*mount), 7)
            self.pygame.draw.circle(self.screen, TITLE, self._place(*mount), 7, 2)

    def _box(self, row, colour, width, grow=0.0):
        self.pygame.draw.polygon(self.screen, colour, [self._place(*corner) for corner in box_corners(row, grow)], width)

    def _line(self, words, font=None, colour=TEXT, gap=24, indent=0):
        self.screen.blit((font or self.small).render(words, True, colour), (self.size + 24 + indent, self.cursor))
        self.cursor += gap

    def _draw_panel(self, now):
        state = self.state
        for index, name in enumerate(("A", "B")):
            self.pygame.draw.circle(self.screen, COLOURS[index], (self.size + 31, self.cursor + 12), 6)
            self._line(f"LiDAR {name}" + (" (hidden)" if self.hidden[index] else ""), self.medium, TITLE, 28, indent=22)
            self._line(f"Points in the grid: {len(state['sweeps'][index]):,}")
            rate = self.rates[index].per_second()
            self._line("Sweeps per second: " + ("measuring..." if rate is None else f"{rate:.1f}"))
            behind = state["frame"] - state["last_frames"][index]
            self._line(f"Last sweep: {'this tick' if behind == 0 else f'{behind} ticks ago'}", colour=TEXT if behind == 0 else BAD, gap=34)
        self._line("This sweep", self.medium, TITLE, 28)
        true_boxes, seen = state["true_boxes"], self.seen_counts >= MIN_POINTS
        for label, kind in (("Vehicles", VEHICLE), ("People", PEDESTRIAN)):
            of_kind = true_boxes[:, 7] == kind
            self._line(f"{label}: {int((state['detections'][:, 8] == kind).sum())} detected,", gap=20)
            self._line(f"{int((of_kind & seen).sum())} hit by LiDAR, {int(of_kind.sum())} in CARLA", gap=28, indent=12)
        self.cursor += 8
        self._line("Key", self.medium, TITLE, 28)
        for colour, words in ((TRUE_SEEN, "CARLA's true box"), (TRUE_UNSEEN, f"True box, under {MIN_POINTS} LiDAR points"),
                              (DETECTED_CAR, "Detected vehicle"), (DETECTED_PERSON, "Detected person")):
            self.pygame.draw.rect(self.screen, colour, (self.size + 24, self.cursor + 4, 16, 12), 2)
            self._line(words, indent=26)
        self.cursor += 6
        self._line("Dim points are the road, bright are above it.")
        self._line("1 or 2 hides a LiDAR. Esc closes.")
        self.cursor = self.size - 34
        waited = now - self.updated
        if waited > STOPPED_AFTER:
            self._line(f"NO NEW SWEEPS for {waited:.0f} s", self.medium, BAD)
        else:
            self._line(f"Frame {state['frame']}, {state['time']:.1f} s simulated")

    def run(self):
        print("[LIDAR VIEW] Close the window to exit.", flush=True)
        while True:
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT or (event.type == self.pygame.KEYDOWN and event.key == self.pygame.K_ESCAPE):
                    return
                if event.type == self.pygame.KEYDOWN and event.key in (self.pygame.K_1, self.pygame.K_2):
                    index = event.key - self.pygame.K_1
                    self.hidden[index] = not self.hidden[index]
            now = time.monotonic()
            self.poll(now)
            self.draw(now)
            self.clock.tick(FPS)

    def close(self):
        self.pygame.quit()
        if self.feed:
            self.feed.close()


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    import pygame

    for name in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _raise_keyboard_interrupt)
    window = Window(pygame)
    try:
        window.run()
    except KeyboardInterrupt:
        pass
    finally:
        window.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
