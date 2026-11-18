"""
Checks a Yahboom Rosmaster R2 is hooked up right, run ON the car. Pair it with check_lab_link.py, in this folder, on the
laptop, which pings this script and reports the link's delay from the other end. See README.md here for the steps.

    python3 check_rosmaster.py --broker 192.168.1.50               # the laptop's IP; listens for 60 s
    python3 check_rosmaster.py --broker 192.168.1.50 --listen 0    # until Ctrl+C
    python3 check_rosmaster.py --no-board                          # just the network, e.g. on a laptop as a stand-in
    python3 check_rosmaster.py --broker 192.168.1.50 --motors      # also spin the wheels briefly: WHEELS OFF THE GROUND

It checks, printing PASS, FAIL, WARN or SKIP with a hint for each:
    the board       Rosmaster_Lib opens it; its firmware version, battery, IMU, and wheel encoders while you turn a wheel
    the motors      only with --motors: a 1 s slow spin that the encoders must see, then a stop
    the network     the broker answers; the safety gate's state arrives; any drive commands are valid
    pings           while it listens, it answers check_lab_link.py's pings with its readings

The board's readings use Rosmaster_Lib calls that haven't been tried on a real car yet: one that errors shows as
WARN "unverified", meaning check the library, not that the car is broken. Run it with the safety gate
(physical_ml_subscriber.py) and Yahboom's ROS2 driver stopped, as only one program can hold the board's serial port.
With the gate stopped nothing holds the motors off, so a PASS here doesn't mean ready to drive: restart the gate after.

On the car, run it from a copy of the whole repository, or copy this folder with scripts/physical_ml_subscriber.py put
inside it. It needs only Python 3.6+, paho-mqtt and Rosmaster_Lib.
Exit code 0 if nothing failed, 1 if something did, 2 if nothing passed at all.
"""
import argparse
import json
import math
import signal
import sys
import time
from collections import namedtuple
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # scripts/, for the gate; a copy beside this file wins
from physical_ml_subscriber import (
    COMMAND_TOPIC,
    DRIVE_TOPIC,
    MQTT_PORT,
    parse_drive_command,
    parse_gate_event,
)

# shortcut: these are a 3S pack's usual limits and guesses for the R2; put in the measured ones after the first run
BATTERY_LOW, BATTERY_FULL, BATTERY_MAX = 10.5, 12.6, 13.5   # volts: under LOW charge it; over MAX the reading is wrong
LEVEL_LIMIT = 30.0          # degrees of roll or pitch beyond which the car isn't sitting level
WHEEL_SECONDS = 3.0         # to turn a wheel by hand while the encoders are watched
MOTOR_SPEED, MOTOR_SECONDS = 0.1, 1.0   # set_car_motion's forward speed for the spin, and how long, at most
CHECK_TOPIC = "device/vehicle_ml/check"
LISTENER = "check_rosmaster.py, not the safety gate"

Result = namedtuple("Result", "status name detail")


def unverified(name, error):
    return Result("WARN", name, f"couldn't read it ({error}): unverified Rosmaster_Lib call, check the library rather than the car")


def _finite(*values):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)


def judge_version(value, error):
    if error:
        return unverified("firmware", error)
    if not _finite(value) or value <= 0:
        return Result("FAIL", "firmware", f"the board answered {value!r}: is it powered, and the cable in?")
    return Result("PASS", "firmware", f"version {value}")


def judge_battery(volts, error):
    if error:
        return unverified("battery", error)
    if not _finite(volts) or volts <= 0 or volts > BATTERY_MAX:
        return Result("FAIL", "battery", f"{volts!r} V isn't a real reading: is the board powered?")
    if volts < BATTERY_LOW:
        return Result("FAIL", "battery", f"{volts:.2f} V, under {BATTERY_LOW} V: charge it before driving")
    if volts < BATTERY_LOW + 0.5:
        return Result("WARN", "battery", f"{volts:.2f} V: low, charge it soon")
    return Result("PASS", "battery", f"{volts:.2f} V")


def judge_imu(attitude, error):
    if error:
        return unverified("imu", error)
    if not isinstance(attitude, (tuple, list)) or len(attitude) != 3 or not _finite(*attitude):
        return Result("FAIL", "imu", f"expected roll, pitch and yaw, got {attitude!r}")
    roll, pitch, yaw = attitude
    if abs(roll) > LEVEL_LIMIT or abs(pitch) > LEVEL_LIMIT:
        return Result("WARN", "imu", f"roll {roll:.1f}, pitch {pitch:.1f} degrees: is the car on its side, or the board mounted crooked?")
    return Result("PASS", "imu", f"roll {roll:.1f}, pitch {pitch:.1f}, yaw {yaw:.1f} degrees")


def judge_encoders(before, after, error, name="encoders", hint="turn a wheel by hand while it watches"):
    if error:
        return unverified(name, error)
    if not (isinstance(before, (tuple, list)) and isinstance(after, (tuple, list)) and len(before) == len(after) and before):
        return Result("WARN", name, f"unexpected readings {before!r} then {after!r}: unverified, check the library")
    moved = [i + 1 for i, (a, b) in enumerate(zip(before, after)) if a != b]
    if not moved:
        return Result("WARN", name, f"no wheel moved ({after!r}): {hint}, or check the encoder cables")
    return Result("PASS", name, f"wheel {', '.join(map(str, moved))} counted: {before!r} to {after!r}")


def exit_code(results):
    """1 if anything failed; 2 if nothing passed, so an all-skipped run never looks healthy; 0 otherwise."""
    if any(r.status == "FAIL" for r in results):
        return 1
    return 0 if any(r.status == "PASS" for r in results) else 2


def parse_ping(payload):
    """(run, n) from a check_lab_link.py ping, or None. Nothing else in it is used."""
    try:
        ping = json.loads(payload)
    except (ValueError, TypeError):
        return None
    if not isinstance(ping, dict) or not isinstance(ping.get("run"), str):
        return None
    n = ping.get("n")
    if not isinstance(n, int) or isinstance(n, bool):
        return None
    return ping["run"], n


def ping_reply(run, n, board, heard_gate, heard_drive, gate_process):
    return json.dumps({"run": run, "n": n, "report": {
        "listener": LISTENER, "board": board, "heard_gate_state": heard_gate, "valid_drive_commands": heard_drive,
        "gate_process_running": gate_process}}).encode()


def gate_running(proc=Path("/proc")):
    """Whether the safety gate's process is running on this Linux machine; None where there's no /proc to look in."""
    if not Path(proc).is_dir():
        return None
    for cmdline in Path(proc).glob("[0-9]*/cmdline"):
        try:
            if b"physical_ml_subscriber" in cmdline.read_bytes():
                return True
        except OSError:
            continue
    return False


def read(call):
    """(value, None) or (None, the error), so one failing library call can't stop the rest."""
    try:
        return call(), None
    except Exception as e:  # noqa: BLE001  # the library's errors are unknown, and each one is reported
        return None, f"{type(e).__name__}: {e}"


def check_board(motors, gate):
    """The board's results."""
    try:
        from Rosmaster_Lib import Rosmaster
    except ImportError:
        return [Result("SKIP", "board", "Rosmaster_Lib isn't installed: run this on the car, or use --no-board")]
    bot, error = read(Rosmaster)
    if error:
        busy = any(word in error.lower() for word in ("busy", "permission", "could not open", "denied"))
        hint = ("stop the safety gate and Yahboom's ROS2 driver first, as only one program can hold the serial port"
                if busy else "is the board's USB cable in?")
        return [Result("FAIL", "board", f"couldn't open it ({error}): {hint}")]
    _, error = read(bot.create_receive_threading)
    time.sleep(0.5)  # the board's first readings arrive on that thread
    results = [Result("PASS", "board", "Rosmaster_Lib opened it")] if not error else [unverified("board", error)]
    results.append(judge_version(*read(bot.get_version)))
    results.append(judge_battery(*read(bot.get_battery_voltage)))
    results.append(judge_imu(*read(bot.get_imu_attitude_data)))
    before, error = read(bot.get_motor_encoder)
    after, later = None, None
    if not error:
        print(f"[ROSMASTER] Turn any wheel by hand now, for {WHEEL_SECONDS:.0f} s...", flush=True)
        time.sleep(WHEEL_SECONDS)
        after, later = read(bot.get_motor_encoder)
    results.append(judge_encoders(before, after, error or later))
    if motors:
        results.append(check_motors(bot, gate))
    return results


def check_motors(bot, gate):
    if gate:
        return Result("SKIP", "motors", "the safety gate is running: stop it first, as the two would fight over the motors")
    print("[ROSMASTER] The motor test spins the wheels for 1 s. Nothing else holds the motors off while it runs.", flush=True)
    if input("[ROSMASTER] Are the wheels OFF THE GROUND? Type yes to spin them: ").strip() != "yes":
        return Result("SKIP", "motors", "not confirmed")
    before, error = read(bot.get_motor_encoder)
    after, later = None, None
    try:
        if not error:
            _, error = read(lambda: bot.set_car_motion(MOTOR_SPEED, 0, 0))
            time.sleep(MOTOR_SECONDS)
            after, later = read(bot.get_motor_encoder)
    finally:
        for _ in range(3):  # one lost stop mustn't leave them turning
            read(lambda: bot.set_car_motion(0, 0, 0))
            read(lambda: bot.set_motor(0, 0, 0, 0))
            time.sleep(0.05)
    return judge_encoders(before, after, error or later, name="motors", hint="the motors didn't turn: check their power and cables")


def listen(broker, port, seconds, board_summary, gate):
    """The network's results: the broker, the gate's state, drive commands, and answering pings meanwhile."""
    import paho.mqtt.client as mqtt

    heard = {"gate": False, "drive": 0, "bad_drive": 0, "pings": 0, "connected": False}

    def on_connect(client, userdata, flags, reason_code, properties):
        if not reason_code.is_failure:
            heard["connected"] = True
            client.subscribe([(COMMAND_TOPIC, 1), (DRIVE_TOPIC, 0), (CHECK_TOPIC, 0)])

    def on_message(client, userdata, msg):
        if msg.topic == COMMAND_TOPIC:
            heard["gate"] = heard["gate"] or parse_gate_event(msg.payload) is not None
        elif msg.topic == DRIVE_TOPIC:
            key = "drive" if parse_drive_command(msg.payload) else "bad_drive"
            heard[key] += 1
        elif msg.topic == CHECK_TOPIC:
            ping = parse_ping(msg.payload)
            if ping:
                heard["pings"] += 1
                client.publish(CHECK_TOPIC + "/reply", ping_reply(*ping, board_summary, heard["gate"], heard["drive"], gate), qos=0)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"check-rosmaster-{int(time.time())}")
    client.on_connect, client.on_message = on_connect, on_message
    try:
        client.connect(broker, port, keepalive=5)
    except OSError as e:
        return [Result("FAIL", "broker", f"can't reach {broker}:{port} ({e}): same Wi-Fi as the laptop? Docker stack up? the right IP?")]
    client.loop_start()
    how_long = "until Ctrl+C" if seconds <= 0 else f"for {seconds:.0f} s"
    print(f"[ROSMASTER] Listening on {broker} {how_long}. Run check_lab_link.py on the laptop now.", flush=True)
    try:
        started = time.monotonic()
        while seconds <= 0 or time.monotonic() - started < seconds:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()
    results = [Result("PASS" if heard["connected"] else "FAIL", "broker",
                      f"connected to {broker}:{port}" if heard["connected"] else f"{broker}:{port} refused the connection")]
    results.append(Result("PASS", "gate state", "the SafetyGate's state arrived") if heard["gate"] else Result(
        "WARN", "gate state", "none arrived: release the brake from the laptop (setup_entities.py --clear-brake) to send one"))
    if heard["bad_drive"]:
        results.append(Result("FAIL", "drive commands", f"{heard['bad_drive']} invalid, {heard['drive']} valid: is the remote driver up to date?"))
    elif heard["drive"]:
        results.append(Result("PASS", "drive commands", f"{heard['drive']} valid ones arrived"))
    else:
        results.append(Result("SKIP", "drive commands", "none sent, which is right unless the remote driver is running"))
    results.append(Result("PASS", "pings", f"answered {heard['pings']}") if heard["pings"] else Result(
        "SKIP", "pings", "none arrived: run check_lab_link.py on the laptop while this listens"))
    return results


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    for name in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _raise_keyboard_interrupt)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--broker", default="localhost", help="the laptop running the Docker stack")
    parser.add_argument("--port", type=int, default=MQTT_PORT)
    parser.add_argument("--listen", type=float, default=60.0, help="seconds to listen for pings and commands; 0 until Ctrl+C")
    parser.add_argument("--no-board", action="store_true", help="skip the board, for running this off the car")
    parser.add_argument("--motors", action="store_true", help="also spin the wheels for 1 s, after you confirm they're off the ground")
    args = parser.parse_args()

    gate = gate_running()
    if gate:
        print("[ROSMASTER] The safety gate (physical_ml_subscriber.py) is running: it holds the board, so the board checks"
              " will fail. Stop it for those, and restart it afterwards.", flush=True)
    results = [Result("SKIP", "board", "--no-board")] if args.no_board else check_board(args.motors, gate)
    board_summary = {r.name: f"{r.status} {r.detail}" for r in results}
    for r in results:
        print(f"[ROSMASTER] {r.status:<4}  {r.name}: {r.detail}", flush=True)
    network = listen(args.broker, args.port, args.listen, board_summary, gate)
    for r in network:
        print(f"[ROSMASTER] {r.status:<4}  {r.name}: {r.detail}", flush=True)
    results += network
    if gate is False:
        print("[ROSMASTER] The safety gate isn't running, so nothing holds this car's motors off. Start it before driving.", flush=True)
    elif gate is None:
        print("[ROSMASTER] Couldn't tell whether the safety gate is running (no /proc here). Make sure it is before driving.", flush=True)
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
