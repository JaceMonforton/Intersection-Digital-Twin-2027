"""
Provisions the twin in Eclipse Ditto from config/: policy, Things and the MQTT connection.
Safe to re-run, but note that re-running resets each Thing to its initial state.

Usage:
    python scripts/setup_entities.py                # provision everything
    python scripts/setup_entities.py --clear-brake  # only release a latched emergency brake
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# Use localhost, not 127.0.0.1: requests to 127.0.0.1:8080 get reset on Windows here.
DITTO_API = "http://localhost:8080/api/2"
DEVOPS_AUTH = ("devops", "foobar")  # Ditto's default; the connections API requires it

SAFETY_GATE_URL = f"{DITTO_API}/things/org.example:vehicle_ml/features/SafetyGate/properties"
CONNECTION_TIMEOUT = 30.0  # seconds to wait for the connection to reach the broker


def load(name):
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


def put(session, url, body, label, **kwargs):
    res = session.put(url, json=body, timeout=10, **kwargs)
    print(f"[SETUP] {label} -> {res.status_code}")
    if res.status_code not in (200, 201, 204):
        raise RuntimeError(f"{label} failed: {res.text[:500]}")


def wait_for_connection(session, connection_id):
    """Block until the connection's client is connected, so the first event isn't lost."""
    url = f"{DITTO_API}/connections/{connection_id}/status"
    deadline = time.time() + CONNECTION_TIMEOUT
    status = None
    while time.time() < deadline:
        res = session.get(url, auth=DEVOPS_AUTH, timeout=10)
        if res.status_code == 200:
            status = res.json()
            clients = status.get("clientStatus", [])
            if clients and all(c.get("status") == "open" for c in clients):
                print(f"[SETUP] Connection {connection_id} is open.")
                return
        time.sleep(1.0)
    raise RuntimeError(f"Connection {connection_id} did not open: {json.dumps(status)[:500]}")


def gate_released():
    return {
        "emergency_brake": False,
        "reason": None,
        "ttc": None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def provision(session):
    for policy_id, body in load("ditto_policies.json").items():
        put(session, f"{DITTO_API}/policies/{policy_id}", body, f"Policy {policy_id}")

    for connection in load("ditto_connections.json"):
        connection_id = connection["id"]
        put(session, f"{DITTO_API}/connections/{connection_id}", connection,
            f"Connection {connection_id}", auth=DEVOPS_AUTH)
        wait_for_connection(session, connection_id)

    for thing_id, body in load("ditto_entities.json").items():
        put(session, f"{DITTO_API}/things/{thing_id}", body, f"Thing {thing_id}")

    # A separate write, because the connection only forwards SafetyGate paths. This publishes
    # the retained "released" state that lets a freshly booted vehicle leave its fail-safe.
    put(session, SAFETY_GATE_URL, gate_released(), "SafetyGate released")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear-brake", action="store_true",
                        help="release a latched emergency brake and do nothing else")
    args = parser.parse_args()

    session = requests.Session()
    try:
        if args.clear_brake:
            put(session, SAFETY_GATE_URL, gate_released(), "SafetyGate released")
        else:
            provision(session)
    except (requests.RequestException, RuntimeError) as e:
        print(f"[SETUP] {e}\n[SETUP] Is the stack up? Ditto needs 60-90 s after `docker compose up`.")
        return 1
    print("[SETUP] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
