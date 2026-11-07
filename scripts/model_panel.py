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

PANEL_HEIGHT = 284
MARGIN = 12              # around and between the cards
PADDING = 16             # inside a card
CARD_SHARES = (0.25, 0.19, 0.27, 0.29)  # of the width: sees, road, decides, safety
MAP_SCALE = 4.0          # pixels per metre
MAP_BEHIND = 8.0         # metres of road shown behind the car
BAR_HEIGHT = 8
TARGET_BAR_FULL = 10.0   # m/s at the end of the target speed bar
FONT = "segoeui,arial"
ROW_HEIGHT = 36
EVENT_LINE_HEIGHT = 19

# Quiet by default: one accent colour, and red only when something is wrong
TEXT = (222, 224, 228)
DIM = (136, 140, 148)
BACKGROUND = (16, 17, 20)
PANEL = (26, 27, 32)
MAP_BACK = (17, 18, 21)
ACCENT = (108, 150, 196)
OK = (98, 176, 136)
ALERT = (222, 92, 88)
ML = (186, 96, 186)
LANE = (98, 176, 136)
CAR = ACCENT
WALKER = (204, 158, 88)
BAR_BACK = (46, 48, 54)

ROUTE_NAMES = {LEFT: "LEFT", RIGHT: "RIGHT"}
SOURCE_NAMES = {"truth": "TRUE POSITIONS", "rsu": "ROADSIDE UNIT", "sight": "OWN SIGHT", "camera": "ROADSIDE CAMERA"}


# ---------------------------------------------------------------------------
# Telemetry (no pygame needed)
# ---------------------------------------------------------------------------
class TelemetrySender:
    """Used by the driver. A display must never get in the way of driving, so send errors are ignored."""

    def __init__(self, address=TELEMETRY_ADDRESS):
        self.address = address
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.source = None  # which view of the world the model is being given: see roadside_link.py
        self.fresh = True   # whether that view is up to date

    def send(self, inputs, output, applied, model_driving):
        message = {"inputs": [round(float(v), 3) for v in inputs], "output": [float(v) for v in output],
                   "applied": [float(v) for v in applied], "model_driving": bool(model_driving),
                   "source": self.source, "fresh": self.fresh}
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


def safety_lines(safety):
    """What the safety checks say, from real values, with n/a where there's nothing to show."""
    ttc = safety.get("ttc") if safety else None
    return [("Min TTC", f"{ttc:.2f} s" if ttc is not None and math.isfinite(ttc) else "n/a"),
            ("Safety brake", "ON" if safety and safety.get("engaged") else "off")]


def is_alert(event):
    return "SAFETY GATE" in event or "COLLISION" in event or "FAIL" in event


def wrap_text(text, measure, width):
    """Text split into lines no wider than width, breaking between words. measure gives a string's width."""
    lines = []
    line = ""
    for word in text.split():
        joined = f"{line} {word}" if line else word
        if line and measure(joined) > width:
            lines.append(line)
            line = word
        else:
            line = joined
    return [*lines, line] if line else lines


def control_status(telemetry):
    """(text, is the model in control), or a plain statement that there's no data."""
    if telemetry is None:
        return "NO MODEL DATA", False
    if not telemetry.get("fresh", True):
        return "NO ROADSIDE DATA: BRAKING", False
    if not telemetry["model_driving"]:
        return "SAFETY BRAKE OVERRIDE", False
    source = SOURCE_NAMES.get(telemetry.get("source"))
    return (f"MODEL DRIVING: {source}" if source else "MODEL DRIVING"), True


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
class ModelPanel:
    def __init__(self, width):
        import pygame

        self.pygame = pygame
        self.width = width
        self.title = pygame.font.SysFont(FONT, 15, bold=True)
        self.font = pygame.font.SysFont(FONT, 16)
        self.small = pygame.font.SysFont(FONT, 13)

    def draw(self, surface, top, telemetry, safety=None, events=()):
        pygame = self.pygame
        pygame.draw.rect(surface, BACKGROUND, (0, top, self.width, PANEL_HEIGHT))
        room = self.width - MARGIN * (len(CARD_SHARES) + 1)
        x = MARGIN
        cards = []
        for share, title in zip(CARD_SHARES, ("What the model sees", "Its view of the road", "What the model decides", "Safety and events")):
            card = pygame.Rect(x, top, int(share * room), PANEL_HEIGHT - MARGIN)
            pygame.draw.rect(surface, PANEL, card, border_radius=8)
            self._text(surface, title, card.x + PADDING, card.y + 12, self.title, DIM)
            cards.append(card)
            x = card.right + MARGIN
        self._draw_safety(surface, cards[3], safety, events)
        if telemetry is None:
            self._text(surface, "Waiting for live values from the driver...", cards[0].x + PADDING, cards[0].y + 46, self.font, DIM)
            return
        seen = decode_inputs(np.asarray(telemetry["inputs"], dtype=np.float32))
        self._draw_seen(surface, cards[0], seen)
        self._draw_map(surface, cards[1], seen)
        self._draw_decision(surface, cards[2], telemetry)

    def _text(self, surface, text, x, y, font, color):
        surface.blit(font.render(text, True, color), (x, y))

    def _row(self, surface, card, y, name, value, color=TEXT):
        """A name on the left and its value lined up on the right of the card."""
        self._text(surface, name, card.x + PADDING, y, self.font, DIM)
        rendered = self.font.render(value, True, color)
        surface.blit(rendered, (card.right - PADDING - rendered.get_width(), y))

    def _draw_seen(self, surface, card, seen):
        for i, (name, value) in enumerate(seen_lines(seen)):
            self._row(surface, card, card.y + 44 + i * ROW_HEIGHT, name, value)

    def _draw_map(self, surface, card, seen):
        """The model's inputs from above, always facing up: x forward, y left."""
        pygame = self.pygame
        area = pygame.Rect(card.x + PADDING, card.y + 40, card.width - 2 * PADDING, card.height - 76)
        pygame.draw.rect(surface, MAP_BACK, area, border_radius=4)
        previous_clip = surface.get_clip()
        surface.set_clip(area)
        centre_x, ego_y = area.centerx, area.bottom - MAP_BEHIND * MAP_SCALE

        def at(forward, left):
            return centre_x - left * MAP_SCALE, ego_y - forward * MAP_SCALE

        pygame.draw.lines(surface, LANE, False, [at(*point) for point in seen["lane"]], 2)
        if seen["stop_gap"] is not None:
            # Along the route, so its place on the lane is found by walking the points
            line = _point_along(seen["lane"], seen["stop_gap"])
            pygame.draw.line(surface, ALERT, at(line[0], line[1] + 2.0), at(line[0], line[1] - 2.0), 2)
        for car in seen["cars"]:
            left_edge, top_edge = at(car[0] + 2.25, car[1] + 1.0)
            pygame.draw.rect(surface, CAR, (left_edge, top_edge, 2.0 * MAP_SCALE, 4.5 * MAP_SCALE), 1)
            pygame.draw.line(surface, CAR, at(car[0], car[1]), at(car[0] + car[2], car[1] + car[3]), 1)
        for walker in seen["walkers"]:
            pygame.draw.circle(surface, WALKER, at(walker[0], walker[1]), 3)
        left_edge, top_edge = at(2.25, 1.0)
        pygame.draw.rect(surface, ML, (left_edge, top_edge, 2.0 * MAP_SCALE, 4.5 * MAP_SCALE))
        surface.set_clip(previous_clip)
        offset = 0
        for name, color in (("route", LANE), ("cars", CAR), ("people", WALKER), ("red line", ALERT)):
            self._text(surface, name, area.x + offset, area.bottom + 7, self.small, color)
            offset += self.small.size(name)[0] + 14

    def _draw_decision(self, surface, card, telemetry):
        pygame = self.pygame
        steer, throttle, brake = telemetry["applied"]
        target = float(telemetry["output"][1])
        side = "right" if steer > 0.0 else "left" if steer < 0.0 else "centre"
        x, y, width = card.x + PADDING, card.y + 44, card.width - 2 * PADDING

        self._row(surface, card, y, "Steering", f"{abs(steer):.2f} {side}")
        pygame.draw.rect(surface, BAR_BACK, (x, y + 26, width, BAR_HEIGHT), border_radius=4)
        extent = int(abs(steer) * width / 2)
        pygame.draw.rect(surface, ACCENT, (x + width // 2 if steer > 0.0 else x + width // 2 - extent, y + 26, extent, BAR_HEIGHT))
        pygame.draw.line(surface, TEXT, (x + width // 2, y + 23), (x + width // 2, y + 26 + BAR_HEIGHT + 2), 1)

        shown = f"{target * 3.6:.1f} km/h" if math.isfinite(target) else "invalid"
        self._row(surface, card, y + 50, "Target speed", shown)
        pygame.draw.rect(surface, BAR_BACK, (x, y + 76, width, BAR_HEIGHT), border_radius=4)
        filled = max(0.0, min(1.0, target / TARGET_BAR_FULL)) if math.isfinite(target) else 0.0
        pygame.draw.rect(surface, ACCENT, (x, y + 76, int(filled * width), BAR_HEIGHT), border_radius=4)

        half = width // 2 - 8
        self._text(surface, f"Throttle {throttle:.2f}", x, y + 100, self.small, DIM)
        pygame.draw.rect(surface, BAR_BACK, (x, y + 122, half, BAR_HEIGHT), border_radius=4)
        pygame.draw.rect(surface, ACCENT, (x, y + 122, int(throttle * half), BAR_HEIGHT), border_radius=4)
        self._text(surface, f"Brake {brake:.2f}", x + half + 16, y + 100, self.small, DIM)
        pygame.draw.rect(surface, BAR_BACK, (x + half + 16, y + 122, half, BAR_HEIGHT), border_radius=4)
        pygame.draw.rect(surface, ALERT, (x + half + 16, y + 122, int(brake * half), BAR_HEIGHT), border_radius=4)

        if telemetry["model_driving"]:
            self._text(surface, "A fixed controller sets these pedals to reach the target.", x, y + 146, self.small, DIM)
        else:
            self._text(surface, "What the model wants: the safety brake is overriding it.", x, y + 146, self.small, ALERT)

    def _draw_safety(self, surface, card, safety, events):
        x, width = card.x + PADDING, card.width - 2 * PADDING
        y = card.y + 44
        for name, value in safety_lines(safety):
            self._row(surface, card, y, name, value, ALERT if value == "ON" else TEXT)
            y += 28
        self._text(surface, "Recent events", x, y + 6, self.title, DIM)
        y += 30
        room = (card.bottom - y - 8) // EVENT_LINE_HEIGHT
        lines = []
        for event in events:
            lines += [(line, is_alert(event)) for line in wrap_text(event, lambda t: self.small.size(t)[0], width)]
        if not lines:
            self._text(surface, "None yet", x, y, self.small, DIM)
        for i, (line, alert) in enumerate(lines[-room:]):  # the newest, when they don't all fit
            self._text(surface, line, x, y + i * EVENT_LINE_HEIGHT, self.small, ALERT if alert else TEXT)


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
