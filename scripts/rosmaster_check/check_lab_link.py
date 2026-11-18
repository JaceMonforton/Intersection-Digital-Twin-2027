"""
Checks the link to a Rosmaster from the laptop's end, while check_rosmaster.py, in this folder, runs on the car.
See README.md here for the steps.

    python scripts/rosmaster_check/check_lab_link.py              # the stack, 50 pings to the car, and the car's pose
    python scripts/rosmaster_check/check_lab_link.py --no-pose    # before the overhead camera (or the fake car) is publishing

It checks, printing PASS, FAIL, WARN or SKIP with a hint for each:
    Ditto           the stack answers and the vehicle_ml Thing exists
    the broker      it can connect
    the link        50 pings at 10 Hz to check_rosmaster.py: how long they take there and back, and how many are lost.
                    The car's gate holds it after 0.3 s without a command, so a link slower than that will stop it
    the car         its own report, from its replies: the board, the gate's state, and whether the gate is running
    the pose        the car's pose on telemetry/vehicle_ml reads as one and arrives at 15 Hz or more

The delay is timed on this laptop's clock alone, and holds for this Wi-Fi at this moment: check again if it changes.
Exit code 0 if nothing failed, 1 if something did, 2 if nothing passed at all.
"""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # scripts/, for lab_frame and the gate
import lab_frame
from check_rosmaster import CHECK_TOPIC, Result, exit_code

WATCHDOG = 0.3     # s: the car's gate holds it after this long without a command
P95_PASS = 0.15    # s: half the watchdog, to leave room for a slow moment
LATE_AFTER = 1.0   # s: a reply later than this counts as lost
LOST_WARN = 2      # pings of 50 that may be lost before it fails
POSE_MIN_HZ = 15.0
POSE_WINDOW = 5.0  # s of pose telemetry to count


def judge_latency(sent, replies):
    """sent: ping number -> the laptop's send time; replies: number -> the first reply's arrival, on the same clock."""
    trips = sorted(replies[n] - sent[n] for n in sent if n in replies and 0.0 <= replies[n] - sent[n] <= LATE_AFTER)
    lost = len(sent) - len(trips)
    if not trips:
        return Result("FAIL", "link", f"no replies to {len(sent)} pings: is check_rosmaster.py running on the car, on this broker?")
    median = trips[len(trips) // 2]
    p95 = trips[min(len(trips) - 1, int(0.95 * len(trips)))]
    slow = sum(t > WATCHDOG for t in trips)
    detail = f"median {1000 * median:.0f} ms, 95% under {1000 * p95:.0f} ms, {slow} over {WATCHDOG} s, {lost} lost of {len(sent)}"
    if slow or lost > LOST_WARN:
        return Result("FAIL", "link", detail + ": the car's gate would hold it. Closer to the router, or less Wi-Fi traffic?")
    if lost or p95 > P95_PASS:
        return Result("WARN", "link", detail + ": it works, with little room to spare")
    return Result("PASS", "link", detail)


def judge_pose(valid, invalid, window):
    if not valid and not invalid:
        return Result("FAIL", "pose", "none arrived: is the overhead camera (or scripts/fake_lab_car.py) publishing? Or use --no-pose")
    if invalid:
        return Result("FAIL", "pose", f"{invalid} of {valid + invalid} didn't read as a pose: check x, y, heading, speed and t")
    rate = valid / window
    if rate < POSE_MIN_HZ:
        return Result("FAIL", "pose", f"{rate:.1f} Hz, under {POSE_MIN_HZ:.0f} Hz: the twin and the brake would lag behind the car")
    return Result("PASS", "pose", f"{rate:.1f} Hz")


def check_ditto(car):
    import requests

    url = f"http://localhost:8080/api/2/things/org.example:{car}"
    try:
        response = requests.get(url, timeout=5)
    except requests.RequestException as e:
        return Result("FAIL", "Ditto", f"no answer from localhost:8080 ({type(e).__name__}): is the Docker stack up?")
    if response.status_code == 404:
        return Result("FAIL", "Ditto", f"there's no {car} Thing: run python scripts/setup_entities.py")
    if response.status_code != 200:
        return Result("FAIL", "Ditto", f"HTTP {response.status_code} for {url}")
    return Result("PASS", "Ditto", f"the {car} Thing is there")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--car", default="vehicle_ml")
    parser.add_argument("--pings", type=int, default=50)
    parser.add_argument("--no-pose", action="store_true", help="skip the pose check")
    args = parser.parse_args()
    if args.pings < 1:
        parser.error("--pings must be at least 1")

    import paho.mqtt.client as mqtt

    results = [check_ditto(args.car)]
    run, sent, replies, report, poses = uuid.uuid4().hex, {}, {}, {}, {"valid": 0, "invalid": 0, "counting": False}

    def on_message(client, userdata, msg):
        now = time.monotonic()
        if msg.topic == CHECK_TOPIC + "/reply":
            try:
                reply = json.loads(msg.payload)
            except ValueError:
                return
            if isinstance(reply, dict) and reply.get("run") == run and reply.get("n") in sent:
                replies.setdefault(reply["n"], now)  # the first answer counts; a duplicate doesn't
                report.update(reply.get("report") or {})
        elif poses["counting"]:
            try:
                message = json.loads(msg.payload)
                value = message["value"] if message.get("path") == "/features/pose/properties" else None
            except (ValueError, KeyError, TypeError, AttributeError):
                value = None
            ok = isinstance(value, dict) and lab_frame.read_pose({"features": {"pose": {"properties": value}}}) is not None
            poses["valid" if ok else "invalid"] += 1

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"check-lab-link-{run[:8]}")
    client.on_message = on_message
    try:
        client.connect(args.broker, args.port, keepalive=5)
    except OSError as e:
        results.append(Result("FAIL", "broker", f"can't reach {args.broker}:{args.port} ({e}): is the Docker stack up?"))
        return finish(results)
    results.append(Result("PASS", "broker", f"connected to {args.broker}:{args.port}"))
    client.subscribe([(CHECK_TOPIC + "/reply", 0), (f"telemetry/{args.car}", 0)])
    client.loop_start()
    time.sleep(0.5)

    print(f"[LAB LINK] Pinging the car {args.pings} times...", flush=True)
    for n in range(args.pings):
        sent[n] = time.monotonic()
        client.publish(CHECK_TOPIC, json.dumps({"run": run, "n": n}), qos=0)
        time.sleep(0.1)
    time.sleep(LATE_AFTER)
    results.append(judge_latency(sent, replies))
    if report:
        board = report.get("board") or {}
        results.append(Result("PASS" if report.get("heard_gate_state") else "WARN", "car's gate state",
                              "the car hears the SafetyGate" if report.get("heard_gate_state")
                              else "the car hasn't heard the SafetyGate's state: release it with setup_entities.py --clear-brake"))
        for name, line in board.items():
            print(f"[LAB LINK]       car's {name}: {line}", flush=True)
        if report.get("gate_process_running") is False:
            print("[LAB LINK]       the car's safety gate isn't running: start it there before driving.", flush=True)

    if args.no_pose:
        results.append(Result("SKIP", "pose", "--no-pose"))
    else:
        print(f"[LAB LINK] Counting the car's poses for {POSE_WINDOW:.0f} s...", flush=True)
        poses["counting"] = True
        time.sleep(POSE_WINDOW)
        poses["counting"] = False
        results.append(judge_pose(poses["valid"], poses["invalid"], POSE_WINDOW))
    client.loop_stop()
    client.disconnect()
    return finish(results)


def finish(results):
    for r in results:
        print(f"[LAB LINK] {r.status:<4}  {r.name}: {r.detail}", flush=True)
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
