"""
Runs the whole lab-car loop with one command: the fake RC car (scripts/fake_lab_car.py) behind its real safety gate,
its twin in CARLA, and the trained model driving it through that twin (scripts/remote_driver.py).

    python full_loop_sim.py                         # until Ctrl+C
    python full_loop_sim.py --seconds 120           # two minutes
    python full_loop_sim.py --outage 20,22          # 2 s of silence on the car's link, 20 s in: its watchdog holds it
    python full_loop_sim.py --drop 0.1 --replay 0.05 --no-view
    python full_loop_sim.py --leave-running         # keep CARLA and Docker up afterwards

In order, it starts whatever isn't running (Docker, the Ditto stack, CARLA), then the processor, which owns the clock,
then the fake car, then releases its gate, then opens the twin mirror's window and starts the driver. Ctrl+C, or the
end of --seconds, stops them in reverse: the driver first, so its hold is the last command the car gets. Then it shuts
down only what it started. Logs go to output/full_loop_*.log.

Only one copy at a time: a second one on the same CARLA disturbs the first. Everything here is a simulated stand-in.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
sys.path.insert(0, str(ROOT / "scripts"))


def fake_car_args(args):
    """fake_lab_car.py's command line for these options: only the faults that were asked for."""
    command = ["scripts/fake_lab_car.py"]
    if args.drop:
        command += ["--drop", str(args.drop)]
    if args.delay:
        command += ["--delay", args.delay]
    if args.replay:
        command += ["--replay", str(args.replay)]
    for outage in args.outage:
        command += ["--outage", outage]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=0.0, help="how long to drive; 0 until Ctrl+C")
    parser.add_argument("--drop", type=float, default=0.0, help="share of the car's messages lost, each way")
    parser.add_argument("--delay", metavar="MIN,MAX", help="seconds each of the car's messages is held back")
    parser.add_argument("--replay", type=float, default=0.0, help="share of the car's messages followed by an old one again")
    parser.add_argument("--outage", action="append", default=[], metavar="FROM,TO",
                        help="seconds after the car starts when its link is silent; may be repeated")
    parser.add_argument("--no-view", action="store_true", help="don't open the twin's chase-camera window")
    parser.add_argument("--dry-run", action="store_true", help="the driver prints its commands instead of sending them")
    parser.add_argument("--carla-exe", help="path to CarlaUE4.exe, if it isn't found automatically")
    parser.add_argument("--leave-running", action="store_true", help="keep CARLA and Docker up afterwards")
    args = parser.parse_args()
    if args.seconds < 0:
        parser.error("--seconds must be 0 or more")

    from hidden_hazards import start_processor, stop_processor

    from run_simulation import Launcher

    launcher = Launcher(args)  # for starting Docker, Ditto and CARLA, and stopping what it started
    running = []  # (name, process, log), in start order

    def spawn(name, *command):
        log = open(OUTPUT / f"full_loop_{name}.log", "w", encoding="utf-8", errors="replace")  # noqa: SIM115
        process = subprocess.Popen([sys.executable, "-u", *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        running.append((name, process, log))
        return process

    code = 1
    try:
        launcher.ensure_docker()
        launcher.ensure_ditto()
        launcher.ensure_carla()
        OUTPUT.mkdir(exist_ok=True)
        launcher.log("Starting the processor...")
        processor, log = start_processor("truth", ROOT / "models" / "rsu_detector.candidate.onnx", OUTPUT / "full_loop_processor.log")
        running.append(("processor", processor, log))
        launcher.log("Starting the fake car, a simulated stand-in, behind its real safety gate...")
        spawn("fake_car", *fake_car_args(args))
        time.sleep(1.5)
        subprocess.run([sys.executable, "scripts/setup_entities.py", "--clear-brake"], cwd=ROOT, capture_output=True, check=True)
        if not args.no_view:
            spawn("mirror", "scripts/ditto_carla_agent.py")  # the chase-camera window that follows the car
            time.sleep(8.0)
        driver = spawn("driver", "scripts/remote_driver.py", "--seconds", str(args.seconds), *(["--dry-run"] if args.dry_run else []))
        launcher.log(f"The model is driving the car through its twin{' (dry run)' if args.dry_run else ''}"
                     f"{f' for {args.seconds:.0f} s' if args.seconds else ''}. Ctrl+C to stop.")
        while driver.poll() is None:
            time.sleep(0.5)
            for name, process, _ in running:
                if process.poll() is not None and process is not driver and name != "mirror":
                    raise RuntimeError(f"{name} stopped. See output/full_loop_{name}.log.")
        code = driver.returncode
    except KeyboardInterrupt:
        launcher.log("Interrupted.")
        code = 0
    except (RuntimeError, subprocess.CalledProcessError) as e:
        launcher.log(f"FAILED: {e}")
    finally:
        # The driver first, so its hold is the car's last command, then the car straight away, before its watchdog
        # trips on the silence; the mirror's window can take a while to close
        order = {"driver": 0, "fake_car": 1, "mirror": 2, "processor": 3}  # the processor owns the clock, so it goes last
        for _, process, log in sorted(running, key=lambda entry: order[entry[0]]):
            stop_processor(process, log)
        launcher.shutdown()
    launcher.log("If the car latched, release it with: python scripts/setup_entities.py --clear-brake")
    return code


if __name__ == "__main__":
    sys.exit(main())
