"""
Publishes simulated vehicle telemetry into Eclipse Ditto.

Usage:
    python ditto_publisher.py                       # default start point
    python ditto_publisher.py --x 100 --y -190 --yaw 0
"""
import argparse
import json
import math
import sys
import time

import requests

BASE_URL = "http://127.0.0.1:8081/api/2"
THING_ID = "org.example:vehicle_01"
POLICY_ID = "org.example:policy"
AUTH = ("ditto", "ditto")

THING_URL = f"{BASE_URL}/things/{THING_ID}"
POLICY_URL = f"{BASE_URL}/policies/{POLICY_ID}"

# Ditto only accepts PATCH with this content type (plain application/json -> 415)
MERGE_HEADERS = {"Content-Type": "application/merge-patch+json"}

session = requests.Session()
session.headers["x-ditto-pre-authenticated"] = "nginx:ditto"

def setup_twin(x, y, yaw, speed_kmh):
    """Create the policy and the thing if they don't exist yet."""
    policy_body = {
        "entries": {
            "DEFAULT": {
                "subjects": {"nginx:ditto": {"type": "pre-authenticated"}},
                "resources": {
                    "thing:/": {"grant": ["READ", "WRITE"], "revoke": []},
                    "policy:/": {"grant": ["READ", "WRITE"], "revoke": []},
                },
            }
        }
    }
    res = session.put(POLICY_URL, json=policy_body, timeout=5)
    print(f"[SETUP] Policy PUT -> {res.status_code}")
    if res.status_code not in (201, 204):
        print(f"        {res.text}")

    thing_body = {
        "policyId": POLICY_ID,
        "attributes": {
            "location": {"x": x, "y": y, "z": 0.5},
            "orientation": {"yaw": yaw},
            "kuksa": {"Vehicle.Speed": speed_kmh},
        },
    }
    res = session.put(THING_URL, json=thing_body, timeout=5)
    print(f"[SETUP] Thing PUT  -> {res.status_code}")
    if res.status_code not in (201, 204):
        print(f"        {res.text}")
        print("[SETUP] Could not create/update the thing. Is Ditto up on port 8080?")
        sys.exit(1)


def stream_telemetry(x, y, yaw, speed_kmh):
    print("[STREAMING] Publishing to Ditto at ~20 Hz. Ctrl+C to stop.\n")
    step = 0
    last_error = None

    while True:
        x += 0.4 * math.cos(math.radians(yaw))
        y += 0.4 * math.sin(math.radians(yaw))

        step += 1
        if step > 100:
            yaw = (yaw + 0.5) % 360.0

        attributes = {
            "location": {"x": round(x, 2), "y": round(y, 2), "z": 0.5},
            "orientation": {"yaw": round(yaw, 1)},
            "kuksa": {"Vehicle.Speed": speed_kmh},
        }

        try:
            res = session.patch(
                f"{THING_URL}/attributes",
                data=json.dumps(attributes),
                headers=MERGE_HEADERS,
                timeout=2.0,
            )
            if res.status_code in (200, 204):
                last_error = None
                print(
                    f"\r[DITTO PUBLISH] X: {x:8.2f} | Y: {y:8.2f} | "
                    f"Yaw: {yaw:5.1f} | Speed: {speed_kmh} km/h   ",
                    end="",
                    flush=True,
                )
            else:
                err = f"{res.status_code}: {res.text[:200]}"
                if err != last_error:  # don't spam the same error
                    print(f"\n[ERROR] PATCH failed -> {err}")
                    last_error = err
        except requests.RequestException as e:
            err = f"{type(e).__name__}: {e}"
            if err != last_error:
                print(f"\n[ERROR] {err}")
                last_error = err

        time.sleep(0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, default=100.0)
    parser.add_argument("--y", type=float, default=-190.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=35.0)
    args = parser.parse_args()

    setup_twin(args.x, args.y, args.yaw, args.speed)
    try:
        stream_telemetry(args.x, args.y, args.yaw, args.speed)
    except KeyboardInterrupt:
        print("\n[STOPPED]")