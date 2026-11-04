"""
Shows what the driving model sees and decides, for the ML car's window.

The driver sends the model's real input and output each tick over a local UDP socket, and this draws
exactly those values. It works nothing out for itself and adds no explanations, so if the messages stop
it says so. pygame is only needed for drawing, and the driver never imports it.
"""
import json
import math
import socket
import time
from itertools import pairwise

import numpy as np
from driving_features import decode_inputs
from waymo_predictor_node import LEFT, RIGHT

# --- Configuration ---
TELEMETRY_ADDRESS = ("127.0.0.1", 47801)  # a plain UDP socket, not Ditto, so the Windows localhost rule doesn't apply
SAFETY_ADDRESSES = (("127.0.0.1", 47802), ("127.0.0.1", 47803))  # the test suite, and the ML car's window
SUITE_ADDRESS = ("127.0.0.1", 47804)  # the test suite to the window: the title and the event log
STALE_SECONDS = 1.0      # with nothing newer than this, the panel says it has no data

PANEL_HEIGHT = 240
MARGIN = 16
MAP_SCALE = 4.0          # pixels per metre
MAP_BEHIND = 8.0         # metres of road shown behind the car
BAR_WIDTH = 380
BAR_HEIGHT = 22
TARGET_BAR_FULL = 10.0   # m/s at the end of the target speed bar

TEXT = (235, 235, 235)
DIM = (140, 140, 140)
PANEL = (24, 24, 28)
MAP_BACK = (12, 12, 14)
ML = (255, 0, 255)
LANE = (0, 220, 120)
CAR = (120, 170, 255)
WALKER = (255, 170, 40)
THROTTLE = (0, 220, 120)
BRAKE = (255, 60, 60)
STEER = (120, 170, 255)
TARGET = (255, 220, 90)
BAR_BACK = (55, 55, 60)
ROUTE_NAMES = {LEFT: "LEFT", RIGHT: "RIGHT"}


# ---------------------------------------------------------------------------
# Telemetry (no pygame needed)
# ---------------------------------------------------------------------------
class TelemetrySender:
    """Used by the driver. A display must never get in the way of driving, so send errors are ignored."""

    def __init__(self, address=TELEMETRY_ADDRESS):
        self.address = address
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, inputs, output, applied, model_driving):
        message = {"inputs": [round(float(v), 3) for v in inputs], "output": [float(v) for v in output],
                   "applied": [float(v) for v in applied], "model_driving": bool(model_driving)}
        try:
            self.socket.sendto(json.dumps(message).encode(), self.address)
        except OSError:
            pass


class SafetySender:
    """Used by the processor: what its safety checks say about the ML car this tick."""

    def __init__(self, addresses=SAFETY_ADDRESSES):
        self.addresses = addresses
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, ttc, path_ttc, pet, other, engaged):
        message = json.dumps({"ttc": ttc, "path_ttc": path_ttc, "pet": pet, "other": other, "engaged": bool(engaged),
                              "intersects": path_ttc is not None}).encode()
        for address in self.addresses:
            try:
                self.socket.sendto(message, address)
            except OSError:
                pass


class TelemetryReceiver:
    def __init__(self, address=TELEMETRY_ADDRESS):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(address)
        self.socket.setblocking(False)
        self.latest = None
        self.received_at = -math.inf

    def poll(self, now=None):
        """The newest message, or None if nothing has arrived for STALE_SECONDS."""
        now = time.monotonic() if now is None else now
        while True:
            try:
                data = self.socket.recv(65536)
            except OSError:  # nothing waiting
                break
            self.latest = json.loads(data)
            self.received_at = now
        return self.latest if now - self.received_at <= STALE_SECONDS else None

    def close(self):
        self.socket.close()


def SafetyReceiver(address=SAFETY_ADDRESSES[0]):
    return TelemetryReceiver(address)


# ---------------------------------------------------------------------------
# Projection and text (no pygame needed)
# ---------------------------------------------------------------------------
def lane_to_pixels(lane, spec, width, height, fov_degrees):
    """Lane points (x forward, y LEFT, on the road) as pixels in a camera on the car. spec is the camera's
    place on the car, x forward, y RIGHT, z up, and its pitch. Points behind the camera are left out."""
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    pitch = math.radians(spec["pitch"])
    place = spec["location"]
    pixels = []
    for x, y in lane:
        dx, dy, dz = x - place["x"], -y - place["y"], -place["z"]
        depth = dx * math.cos(pitch) + dz * math.sin(pitch)
        if depth < 1.0:
            continue
        up = -dx * math.sin(pitch) + dz * math.cos(pitch)
        pixels.append((width / 2.0 + focal * dy / depth, height / 2.0 - focal * up / depth))
    return pixels


def seen_lines(seen):
    """The model's input as lines of text."""
    route = ROUTE_NAMES.get(seen["route"], "STRAIGHT")
    stop = "none in range" if seen["stop_gap"] is None else f"{seen['stop_gap']:.1f} m ahead"
    cars = [math.hypot(c[0], c[1]) for c in seen["cars"]]
    walkers = [math.hypot(w[0], w[1]) for w in seen["walkers"]]
    crossing = "none due" if seen["crossing_ttc"] is None else f"{seen['crossing_ttc']:.1f} s"
    return [
        ("Speed", f"{seen['speed'] * 3.6:.1f} km/h"),
        ("Route", route),
        ("Red-light stop line", stop),
        ("Time to crossing car", crossing),
        ("Nearby cars", f"{len(cars)}" + (f", nearest {min(cars):.1f} m" if cars else "")),
        ("Nearby pedestrians", f"{len(walkers)}" + (f", nearest {min(walkers):.1f} m" if walkers else "")),
    ]


def gauge_text(telemetry, safety):
    """The top banner's gauges, from real values, with n/a where there's nothing to show."""
    actual = target = steer = None
    if telemetry is not None:
        actual, target, steer = float(telemetry["inputs"][0]), float(telemetry["output"][1]), float(telemetry["applied"][0])
    ttc = safety.get("ttc") if safety else None
    engaged = bool(safety and safety.get("engaged"))

    def kmh(value):
        return "n/a" if value is None or not math.isfinite(value) else f"{value * 3.6:.0f}"

    return (f"Target {kmh(target)} / Actual {kmh(actual)} km/h   Steer {'n/a' if steer is None else f'{steer:+.2f}'}   "
            f"Min TTC {'n/a' if ttc is None else f'{ttc:.2f} s'}   Brake {'ON' if engaged else 'off'}")


def control_status(telemetry):
    """(text, is the model in control), or a plain statement that there's no data."""
    if telemetry is None:
        return "NO MODEL DATA", False
    return ("MODEL DRIVING", True) if telemetry["model_driving"] else ("SAFETY BRAKE OVERRIDE", False)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
class ModelPanel:
    def __init__(self, width):
        import pygame

        self.pygame = pygame
        self.width = width
        self.title = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font = pygame.font.SysFont("Consolas", 19)
        self.small = pygame.font.SysFont("Consolas", 15)

    def draw(self, surface, top, telemetry):
        pygame = self.pygame
        column = self.width // 3
        pygame.draw.rect(surface, PANEL, (0, top, self.width, PANEL_HEIGHT))
        self._text(surface, "WHAT THE MODEL SEES", MARGIN, top + 10, self.title, TEXT)
        self._text(surface, "ITS VIEW OF THE ROAD", column + MARGIN, top + 10, self.title, TEXT)
        self._text(surface, "WHAT THE MODEL DECIDES", 2 * column + MARGIN, top + 10, self.title, TEXT)
        if telemetry is None:
            self._text(surface, "Waiting for live values from the driver...", MARGIN, top + 50, self.font, DIM)
            return
        seen = decode_inputs(np.asarray(telemetry["inputs"], dtype=np.float32))
        self._draw_seen(surface, MARGIN, top + 44, seen)
        self._draw_map(surface, column + MARGIN, top + 38, column - 2 * MARGIN, PANEL_HEIGHT - 48, seen)
        self._draw_decision(surface, 2 * column + MARGIN, top + 44, telemetry)

    def _text(self, surface, text, x, y, font, color):
        surface.blit(font.render(text, True, color), (x, y))

    def _draw_seen(self, surface, x, y, seen):
        for i, (name, value) in enumerate(seen_lines(seen)):
            self._text(surface, name, x, y + i * 34, self.font, DIM)
            self._text(surface, value, x + 250, y + i * 34, self.font, TEXT)

    def _draw_map(self, surface, x, y, width, height, seen):
        """The model's inputs from above, always facing up: x forward, y left."""
        pygame = self.pygame
        area = pygame.Rect(x, y, width, height)
        pygame.draw.rect(surface, MAP_BACK, area)
        previous_clip = surface.get_clip()
        surface.set_clip(area)
        centre_x, ego_y = area.centerx, area.bottom - MAP_BEHIND * MAP_SCALE

        def at(forward, left):
            return centre_x - left * MAP_SCALE, ego_y - forward * MAP_SCALE

        pygame.draw.lines(surface, LANE, False, [at(*point) for point in seen["lane"]], 3)
        if seen["stop_gap"] is not None:
            # Along the route, so its place on the lane is found by walking the points
            line = _point_along(seen["lane"], seen["stop_gap"])
            pygame.draw.line(surface, BRAKE, at(line[0], line[1] + 2.0), at(line[0], line[1] - 2.0), 3)
        for car in seen["cars"]:
            left_edge, top_edge = at(car[0] + 2.25, car[1] + 1.0)
            pygame.draw.rect(surface, CAR, (left_edge, top_edge, 2.0 * MAP_SCALE, 4.5 * MAP_SCALE), 2)
            pygame.draw.line(surface, CAR, at(car[0], car[1]), at(car[0] + car[2], car[1] + car[3]), 1)
        for walker in seen["walkers"]:
            pygame.draw.circle(surface, WALKER, at(walker[0], walker[1]), 4)
        left_edge, top_edge = at(2.25, 1.0)
        pygame.draw.rect(surface, ML, (left_edge, top_edge, 2.0 * MAP_SCALE, 4.5 * MAP_SCALE))
        surface.set_clip(previous_clip)
        legend = [("route", LANE), ("cars", CAR), ("people", WALKER), ("red line", BRAKE)]
        offset = 8
        for name, color in legend:
            self._text(surface, name, x + offset, y + 6, self.small, color)
            offset += self.small.size(name)[0] + 14

    def _draw_decision(self, surface, x, y, telemetry):
        pygame = self.pygame
        steer, throttle, brake = telemetry["applied"]
        target = float(telemetry["output"][1])
        side = "right" if steer > 0.0 else "left" if steer < 0.0 else "centre"
        self._text(surface, f"Steering  {abs(steer):.2f} {side}", x, y, self.font, TEXT)
        bar = pygame.Rect(x, y + 26, BAR_WIDTH, BAR_HEIGHT)
        pygame.draw.rect(surface, BAR_BACK, bar)
        pygame.draw.line(surface, TEXT, (bar.centerx, bar.top - 3), (bar.centerx, bar.bottom + 2), 2)
        extent = int(abs(steer) * BAR_WIDTH / 2)
        pygame.draw.rect(surface, STEER, (bar.centerx if steer > 0.0 else bar.centerx - extent, bar.top, extent, BAR_HEIGHT))
        shown = f"{target * 3.6:.1f} km/h" if math.isfinite(target) else "invalid"
        self._text(surface, f"Target speed  {shown}", x, y + 56, self.font, TEXT)
        pygame.draw.rect(surface, BAR_BACK, (x, y + 82, BAR_WIDTH, BAR_HEIGHT))
        filled = max(0.0, min(1.0, target / TARGET_BAR_FULL)) if math.isfinite(target) else 0.0
        pygame.draw.rect(surface, TARGET, (x, y + 82, int(filled * BAR_WIDTH), BAR_HEIGHT))
        half = BAR_WIDTH // 2 - 6
        self._text(surface, f"Throttle {throttle:.2f}", x, y + 112, self.small, DIM)
        pygame.draw.rect(surface, BAR_BACK, (x, y + 132, half, 12))
        pygame.draw.rect(surface, THROTTLE, (x, y + 132, int(throttle * half), 12))
        self._text(surface, f"Brake {brake:.2f}", x + half + 12, y + 112, self.small, DIM)
        pygame.draw.rect(surface, BAR_BACK, (x + half + 12, y + 132, half, 12))
        pygame.draw.rect(surface, BRAKE, (x + half + 12, y + 132, int(brake * half), 12))
        if telemetry["model_driving"]:
            self._text(surface, "A fixed controller sets these pedals to reach the target.", x, y + 154, self.small, DIM)
        else:
            self._text(surface, "What the model wants: the safety brake is overriding it.", x, y + 154, self.small, BRAKE)

def _point_along(lane, distance):
    """The (x, y) this far along a polyline, or its end."""
    walked = 0.0
    for a, b in pairwise(lane):
        step = math.hypot(b[0] - a[0], b[1] - a[1])
        if step > 0.0 and walked + step >= distance:
            f = (distance - walked) / step
            return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
        walked += step
    return lane[-1]
