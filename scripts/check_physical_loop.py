"""
A pass/fail check of the whole loop a real lab car will use, with scripts/fake_lab_car.py standing in for the car:
the processor ticks CARLA, the fake car runs the real vehicle gate and publishes its pose, and scripts/remote_driver.py
drives it through its twin. Partway through, the fake car's link goes silent for a moment without disconnecting.

    python scripts/check_physical_loop.py

It passes when:
    poses arrive          the car's pose arrives at 15 Hz or more, marked as a simulated stand-in
    it drives             the driver sends drive commands and the car moves
    silence stops it      after the outage the car is stopped, and stays stopped though commands resume: the gate's
                          watchdog latched it, and only an operator releases it
    a hold on exit        the driver's last command is a hold

Needs CARLA and the Docker stack running with the entities provisioned, and nothing else using CARLA: it starts and
stops the processor itself. Every number is from the fake car, so it checks the plumbing, not the real car.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
OUTAGE = (10.0, 11.0)  # s after the fake car starts that its link is silent
DRIVE_SECONDS = 16.0
SETTLE = 0.5           # s after the outage before the car must be found stopped


def judge(poses, commands, outage=OUTAGE):
    """[(name, passed, detail)] from the poses ((t on the car's clock, speed, source)) and the commands, in the order
    they arrived, each with the car-clock time of the latest pose before it: (t, command)."""
    before = [p for p in poses if p[0] < outage[0]]
    after = [p for p in poses if p[0] >= outage[1] + SETTLE]
    span = (poses[-1][0] - poses[0][0] - (outage[1] - outage[0])) if len(poses) > 1 else 0.0
    rate = len(poses) / span if span > 0 else 0.0
    drives = [c for _, c in commands if not c.get("hold")]
    resumed = [c for t, c in commands if t >= outage[1] and not c.get("hold")]
    top = max((p[1] for p in before), default=0.0)
    return [
        ("poses arrive", rate >= 15.0 and all(p[2] == "simulated stand-in" for p in poses), f"{rate:.1f} Hz from {len(poses)} poses"),
        ("it drives", bool(drives) and top >= 0.1, f"{len(drives)} drive commands, top speed {top:.2f} lab m/s before the outage"),
        ("silence stops it", bool(after) and bool(resumed) and all(p[1] == 0.0 for p in after),
         f"{len(after)} poses after it, fastest {max((p[1] for p in after), default=0.0):.2f} m/s, {len(resumed)} drive commands sent meanwhile"),
        ("a hold on exit", bool(commands) and commands[-1][1].get("hold") is True, "last command: " + (json.dumps(commands[-1][1]) if commands else "none")),
    ]


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    import paho.mqtt.client as mqtt
    from hidden_hazards import start_processor, stop_processor

    poses, commands, latest = [], [], [0.0]

    def on_message(client, userdata, msg):
        message = json.loads(msg.payload)
        if msg.topic.startswith("telemetry/"):
            value = message.get("value", {})
            latest[0] = value.get("t", latest[0])
            poses.append((value.get("t"), value.get("speed"), value.get("source")))
        else:
            commands.append((latest[0], message))

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.connect("localhost", 1883)
    client.subscribe([("telemetry/vehicle_ml", 0), ("device/vehicle_ml/drive", 0)])
    client.loop_start()
    OUTPUT.mkdir(exist_ok=True)

    def spawn(name, *args):
        log = open(OUTPUT / f"loop_check_{name}.log", "w", encoding="utf-8", errors="replace")  # noqa: SIM115
        return subprocess.Popen([sys.executable, "-u", *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP), log

    print("[LOOP CHECK] Starting the processor...", flush=True)
    processor = start_processor("truth", ROOT / "models" / "rsu_detector.candidate.onnx", OUTPUT / "loop_check_processor.log", extra=["--no-draw"])
    car = None
    try:
        car = spawn("fake_car", "scripts/fake_lab_car.py", "--outage", f"{OUTAGE[0]},{OUTAGE[1]}")
        time.sleep(1.5)
        subprocess.run([sys.executable, "scripts/setup_entities.py", "--clear-brake"], cwd=ROOT, capture_output=True, check=False)
        print(f"[LOOP CHECK] Driving the fake car for {DRIVE_SECONDS:.0f} s, with its link silent {OUTAGE[0]:.0f}-{OUTAGE[1]:.0f} s in...", flush=True)
        driver = spawn("remote_driver", "scripts/remote_driver.py", "--seconds", str(DRIVE_SECONDS))
        driver[0].wait(timeout=DRIVE_SECONDS + 90)
        driver[1].close()
        time.sleep(1.0)
    finally:
        if car is not None:
            stop_processor(*car)
        stop_processor(*processor)
        client.loop_stop()

    results = judge([p for p in poses if p[0] is not None], commands)
    for name, passed, detail in results:
        print(f"[LOOP CHECK] {'PASS' if passed else 'FAIL'}  {name}: {detail}", flush=True)
    print("[LOOP CHECK] A simulated stand-in: this checks the plumbing, not a real car.", flush=True)
    return 0 if all(passed for _, passed, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
