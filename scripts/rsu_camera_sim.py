"""
Simulated roadside 3D camera tracks, for the camera_and_rsu study config. This is NOT a model: it is CARLA's true
positions of the actors a camera at a roadside mount has a clear line to, with errors like those of a 3D camera
detector added on top. The processor decides which actors have a clear line (semantic LiDARs at the same mounts, so
the shipping containers hide hazards from the cameras as they do from the LiDARs) and calls camera_observe.

The error figures are stated assumptions, not measurements, and weren't tuned on any test seed. Depth from a camera
degrades with the square of range, while across the line of sight it stays accurate. People are small and often
partly hidden, so they're missed more often the further away they are.

No carla or numpy imports, so it stays testable.
"""
import math

CAMERA_RANGE = 60.0         # m from the nearest mount: the same reach as the roadside LiDARs
DEPTH_SIGMA = (0.05, 4e-4)  # sigma_z = a + b * range**2, in m: 0.41 m at 30 m, 1.49 m at 60 m
LATERAL_SIGMA = 0.10        # m, across the line of sight
YAW_JITTER = 4.0            # degrees, standard deviation
CAR_RECALL = 0.98           # chance a car in view is reported on a tick
PERSON_RECALL = (0.9, 0.6)  # for a person next to a mount and at CAMERA_RANGE, falling linearly between


def depth_sigma(distance):
    return DEPTH_SIGMA[0] + DEPTH_SIGMA[1] * distance ** 2


def person_recall(distance):
    near, far = PERSON_RECALL
    return near + (far - near) * min(distance / CAMERA_RANGE, 1.0)


def camera_observe(actors, mounts, rng):
    """actors as (x, y, yaw degrees, is_person) and mounts as (x, y), in CARLA's frame; rng is a random.Random.
    Returns (index into actors, x, y, yaw) for each actor the cameras report on this tick. Each actor is seen by the
    mount nearest to it, and its error is drawn along and across the line from that mount."""
    reported = []
    for index, (x, y, yaw, is_person) in enumerate(actors):
        mx, my = min(mounts, key=lambda m: (x - m[0]) ** 2 + (y - m[1]) ** 2)
        distance = math.hypot(x - mx, y - my)
        if distance > CAMERA_RANGE:
            continue
        if rng.random() >= (person_recall(distance) if is_person else CAR_RECALL):
            continue
        ux, uy = ((x - mx) / distance, (y - my) / distance) if distance > 1e-6 else (1.0, 0.0)
        along, across = rng.gauss(0.0, depth_sigma(distance)), rng.gauss(0.0, LATERAL_SIGMA)
        reported.append((index, x + along * ux - across * uy, y + along * uy + across * ux, yaw + rng.gauss(0.0, YAW_JITTER)))
    return reported
