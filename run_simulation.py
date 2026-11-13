"""
Starts the whole simulation with one command, runs each chosen config once on each chosen scenario, then shuts down
whatever it started. Each config drives with the trained driving model, the planner and the car's own safety monitor,
perceiving with its detectors; the other cars are CARLA's autopilot.

    python run_simulation.py                                   # every config on every scenario
    python run_simulation.py --lidar_only                      # one config, every scenario
    python run_simulation.py --left_turn --hidden_pedestrian   # every config, two scenarios
    python run_simulation.py --camera_and_rsu --red_light_runner
    python run_simulation.py --seed 3 --leave-running          # another situation, and keep CARLA and Docker up

configs:    --lidar_and_rsu, --camera_and_rsu, --lidar_only (see scripts/study_names.py)
scenarios:  --left_turn, --red_light_runner, --hidden_pedestrian (see scripts/scenario_variation.py)
Giving no config runs them all, and the same for scenarios.

It runs scripts/run_bench.py with one seed of each scenario from the tune split, in real time with the ML car's window
and each config's sensor window open. Runs and the report go to output/run, replaced each time.

Without models/driving_policy.onnx it refuses to run, since the ML car has no other driver. Anything already running
(Docker, the Ditto stack, CARLA) is reused and left alone. If CARLA isn't running it is launched from --carla-exe, the
CARLA_EXE environment variable, or the known install locations below. A first launch can take several minutes.
"""
import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from scenario_variation import SCENARIOS
from study_names import CONFIGS

COMPOSE = ["docker", "compose", "-f", str(ROOT / "docker" / "docker-compose.yml")]
RUN_DIR = ROOT / "output" / "run"
DITTO_THINGS = "http://localhost:8080/api/2/things"
CARLA_PORT = 2000
DRIVING_MODEL = ROOT / "models" / "driving_policy.onnx"

# Packaged builds run directly; a source build runs through the Unreal editor in game mode
PACKAGED_CARLA = [Path(r"C:\carla\Unreal\CarlaUE4\Saved\StagedBuilds\WindowsNoEditor\CarlaUE4.exe")]
SOURCE_EDITOR = Path(r"C:\CarlaUnreal\Engine\Binaries\Win64\UE4Editor.exe")
SOURCE_PROJECT = Path(r"C:\carla\Unreal\CarlaUE4\CarlaUE4.uproject")
# Full render quality, which the overlay colours are tuned for: at low quality they wash out
CARLA_FLAGS = ["-windowed", "-ResX=1280", "-ResY=720", f"-carla-rpc-port={CARLA_PORT}", "-nosound"]

DOCKER_TIMEOUT = 180.0
DITTO_TIMEOUT = 240.0
CARLA_TIMEOUT = 1500.0


class Launcher:
    def __init__(self, args):
        self.args = args
        self.started_docker = False
        self.started_stack = False
        self.carla = None

    def log(self, message):
        print(f"[RUN] {message}", flush=True)

    # --- infrastructure ---
    def docker_up(self):
        return subprocess.run(["docker", "info"], capture_output=True, check=False).returncode == 0

    def ensure_docker(self):
        if shutil.which("docker") is None:
            raise RuntimeError("Docker isn't installed or isn't on PATH.")
        if self.docker_up():
            return
        self.log("Starting Docker Desktop...")
        subprocess.run(["docker", "desktop", "start"], capture_output=True, check=False)
        self.started_docker = True
        wait_for(self.docker_up, DOCKER_TIMEOUT, "Docker to start")

    def ditto_up(self):
        try:
            return requests.get(DITTO_THINGS, timeout=3).status_code == 200
        except requests.RequestException:
            return False

    def ensure_ditto(self):
        if not self.ditto_up():
            self.log("Starting the Ditto stack...")
            subprocess.run([*COMPOSE, "up", "-d"], check=True, capture_output=True)
            self.started_stack = True
            wait_for(self.ditto_up, DITTO_TIMEOUT, "Ditto to answer")
        self.log("Provisioning twin entities...")
        result = None
        for _ in range(6):  # Ditto answers REST slightly before its connectivity service is ready
            result = subprocess.run([sys.executable, "scripts/setup_entities.py"], cwd=ROOT,
                                    capture_output=True, text=True, check=False)
            if result.returncode == 0:
                return
            time.sleep(10.0)
        raise RuntimeError(f"Provisioning failed:\n{result.stdout[-800:]}")

    # --- CARLA ---
    def carla_command(self):
        exe = self.args.carla_exe or os.environ.get("CARLA_EXE")
        if exe:
            return [exe, *CARLA_FLAGS]
        for packaged in PACKAGED_CARLA:
            if packaged.exists():
                return [str(packaged), *CARLA_FLAGS]
        if SOURCE_EDITOR.exists() and SOURCE_PROJECT.exists():
            return [str(SOURCE_EDITOR), str(SOURCE_PROJECT), "-game", *CARLA_FLAGS, "-unattended"]
        raise RuntimeError("Couldn't find CARLA. Start it yourself, or pass --carla-exe or set CARLA_EXE.")

    def carla_answers(self):
        """Probed in a child process with a fresh client each time: a CARLA client that has already
        timed out stays broken, and one can abort outright while the server is starting."""
        try:
            with socket.create_connection(("localhost", CARLA_PORT), timeout=2):
                pass
        except OSError:
            return False
        probe = (f"import carla; c = carla.Client('localhost', {CARLA_PORT}); c.set_timeout(30.0); "
                 "print(c.get_world().get_map().name)")
        try:
            return subprocess.run([sys.executable, "-c", probe], capture_output=True, timeout=45,
                                  check=False).returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def ensure_carla(self):
        if self.carla_answers():
            self.log("CARLA is already running.")
            return
        command = self.carla_command()
        self.log(f"Starting CARLA: {Path(command[0]).name}. This can take several minutes...")
        self.carla = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_for(self.carla_answers, CARLA_TIMEOUT, "CARLA to answer", interval=10.0)

    # --- simulation ---

    def run(self):
        if not DRIVING_MODEL.exists():
            raise RuntimeError(f"{DRIVING_MODEL} is missing, and the ML car has no other driver. Train it with training/train_driving.py.")
        self.ensure_docker()
        self.ensure_ditto()
        self.ensure_carla()
        args = self.args
        shutil.rmtree(RUN_DIR, ignore_errors=True)
        self.log(f"Running {', '.join(args.scenarios)} on {', '.join(args.configs)}. Watch for the magenta ML car. Ctrl+C to stop early.")
        return subprocess.run([sys.executable, "-u", "scripts/run_bench.py", "--configs", *args.configs, "--scenarios", *args.scenarios,
                               "--runs", "1", "--start", str(args.seed), "--out", str(RUN_DIR),
                               *([] if args.no_view else ["--view"]), *(["--quiet"] if args.quiet else [])], cwd=ROOT, check=False).returncode

    def shutdown(self):
        if self.args.leave_running:
            self.log("Leaving CARLA and Docker running.")
            return
        if self.carla is not None:
            self.log("Stopping CARLA...")
            kill_tree(self.carla.pid)
        if self.started_stack:
            self.log("Stopping the Ditto stack (data is kept)...")
            subprocess.run([*COMPOSE, "down"], capture_output=True, check=False)
        if self.started_docker:
            self.log("Stopping Docker Desktop...")
            subprocess.run(["docker", "desktop", "stop"], capture_output=True, check=False)


def kill_tree(pid):
    """The Unreal editor launches the game as a child, so its whole tree has to go."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    else:
        os.kill(pid, signal.SIGTERM)


def wait_for(condition, timeout, what, interval=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return
        time.sleep(interval)
    raise RuntimeError(f"Timed out after {timeout:.0f} s waiting for {what}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in CONFIGS:
        parser.add_argument(f"--{name}", dest="configs", action="append_const", const=name, help="run this config")
    for name in SCENARIOS:
        parser.add_argument(f"--{name}", dest="scenarios", action="append_const", const=name, help="run this scenario")
    parser.add_argument("--seed", type=int, default=0, help="which tune seed's situation each scenario gets: 0 is the first")
    parser.add_argument("--quiet", action="store_true", help="leave out the live table of values")
    parser.add_argument("--no-view", action="store_true", help="don't open the windows")
    parser.add_argument("--carla-exe", help="path to CarlaUE4.exe, if it isn't found automatically")
    parser.add_argument("--leave-running", action="store_true", help="keep CARLA and Docker up afterwards")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be 0 or more")
    args.configs = [c for c in CONFIGS if c in (args.configs or CONFIGS)]
    args.scenarios = [s for s in SCENARIOS if s in (args.scenarios or SCENARIOS)]

    launcher = Launcher(args)
    code = 1
    try:
        code = launcher.run()
    except KeyboardInterrupt:
        launcher.log("Interrupted.")
    except (RuntimeError, subprocess.CalledProcessError) as e:
        launcher.log(f"FAILED: {e}")
    finally:
        launcher.shutdown()
    launcher.log(f"Results are in {RUN_DIR}.")
    return code


if __name__ == "__main__":
    sys.exit(main())
