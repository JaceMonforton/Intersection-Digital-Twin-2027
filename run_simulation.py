"""
Starts the whole simulation with one command, runs it, then shuts down whatever it started. By default
that's the five-test demonstration in scripts/run_professor_test_suite.py. Either way the ML car is driven
by the trained driving model and the others by CARLA's autopilot.

    python run_simulation.py                  # the five tests
    python run_simulation.py --free-drive     # normal driving in traffic for 5 simulated minutes
    python run_simulation.py --free-drive --seconds 0   # until Ctrl+C
    python run_simulation.py --free-drive --traffic 30  # more cars
    python run_simulation.py --leave-running  # keep CARLA and Docker up, for faster repeat runs
    python run_simulation.py --roadside --lidar-view  # the tests on the roadside detector, with a window of what its LiDARs see

Without models/driving_policy.onnx it refuses to run, since the ML car has no other driver.

Anything already running (Docker, the Ditto stack, CARLA) is reused and left alone. If CARLA isn't
running it is launched from --carla-exe, the CARLA_EXE environment variable, or the known
install locations below. A first launch can take several minutes.
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
COMPOSE = ["docker", "compose", "-f", str(ROOT / "docker" / "docker-compose.yml")]
LOG_DIR = ROOT / "output" / "logs"
DITTO_THINGS = "http://localhost:8080/api/2/things"
CARLA_PORT = 2000
ROADSIDE_MODEL = ROOT / "models" / "rsu_detector.candidate.onnx"
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
MASTER_TIMEOUT = 420.0
MASTER_ATTEMPTS = 2  # a cold CARLA can abort the client during its first map load


class Launcher:
    def __init__(self, args):
        self.args = args
        self.processes = []  # (name, process), in start order
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
    def spawn(self, name, *args):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / f"{name}.log"
        with open(path, "w", encoding="utf-8", errors="replace") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
        self.processes.append((name, process))
        return process, path

    def start_processor(self):
        for attempt in range(1, MASTER_ATTEMPTS + 1):
            self.log(f"Starting the central processor (attempt {attempt})...")
            process, path = self.spawn("processor", "scripts/central_gpu_processor.py",
                                       *(["--detector", "onnx", "--model", str(ROADSIDE_MODEL)] if self.args.roadside else []),
                                       *(["--lidar-view"] if self.args.lidar_view else []))
            deadline = time.time() + MASTER_TIMEOUT
            while time.time() < deadline and process.poll() is None:
                if "Master tick" in path.read_text(encoding="utf-8", errors="replace"):
                    return
                time.sleep(1.0)
            self.stop_process("processor", process)
            self.processes.pop()
            self.log(f"Processor didn't start. Last output:\n{tail(path)}")
        raise RuntimeError(f"The central processor failed to start. See {path}.")

    def stop_process(self, name, process):
        if process.poll() is not None:
            return
        process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.log(f"{name} didn't exit, killing it.")
            process.kill()
            process.wait()

    def run(self):
        if not DRIVING_MODEL.exists():
            raise RuntimeError(f"{DRIVING_MODEL} is missing, and the ML car has no other driver. Train it with training/train_driving.py.")
        self.ensure_docker()
        self.ensure_ditto()
        self.ensure_carla()
        self.start_processor()
        self.spawn("subscriber", "scripts/physical_ml_subscriber.py", "--driver", "sim")
        self.spawn("predictor", "scripts/waymo_predictor_node.py")
        time.sleep(2.0)
        for name, process in self.processes:
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited early:\n{tail(LOG_DIR / f'{name}.log')}")
        if not self.args.no_view:
            self.spawn("view", "scripts/ml_car_view.py")  # only a display, so it may fail without stopping the run
        if self.args.lidar_view:
            self.spawn("lidar_view", "scripts/lidar_view.py")  # likewise

        code = self.run_with_traffic() if self.args.free_drive else self.run_suite()
        for name, process in self.processes:
            if process.poll() is not None:
                self.log(f"WARNING: {name} died during the run, so overlays or braking may be missing:\n"
                         f"{tail(LOG_DIR / f'{name}.log')}")
        return code

    def run_suite(self):
        self.log("Running the five-test demonstration. Watch for the magenta ML car. Ctrl+C to stop early.")
        return self.run_stage("scripts/run_professor_test_suite.py", "--attached", *(["--quiet"] if self.args.quiet else []),
                              *(["--perception", "rsu"] if self.args.roadside else []))

    def run_stage(self, *command):
        return subprocess.run([sys.executable, "-u", *command], cwd=ROOT, check=False).returncode

    def run_with_traffic(self):
        args = self.args
        self.log(f"Spawning {args.traffic} vehicles and {args.walkers} pedestrians...")
        command = ["scripts/test_traffic.py", "--asynch", "--near-view", "--safe", "-n", str(args.traffic),
                   "-w", str(args.walkers), "--tm-port", "8000", "--car-lights-on"]
        if args.seed is not None:
            command += ["--seed", str(args.seed)]
        process, path = self.spawn("traffic", *command)
        wait_for(lambda: "spawned" in path.read_text(encoding="utf-8", errors="replace") or process.poll() is not None,
                 120.0, "traffic to spawn")
        if process.poll() is not None:
            raise RuntimeError(f"test_traffic.py exited early:\n{tail(path)}")
        self.log("The model is driving the magenta ML car. Ctrl+C to stop early.")
        return self.run_stage("scripts/traffic_monitor.py", "--duration", str(args.seconds))

    def shutdown(self):
        # Reverse order: the processor owns the clock, so it goes last
        for name, process in reversed(self.processes):
            self.stop_process(name, process)
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


def tail(path, lines=15):
    text = path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    return "\n".join(line for line in text.splitlines() if not line.startswith("INFO"))[-1500:] or "(no output)"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--free-drive", action="store_true", help="normal driving in traffic instead of the five tests")
    parser.add_argument("--quiet", action="store_true", help="leave out the test suite's live table of values")
    parser.add_argument("--roadside", action="store_true", help="run the tests on the roadside detector and the car's own LiDAR, instead of CARLA's true positions")
    parser.add_argument("--lidar-view", action="store_true", help="with --roadside, open a window of what the two roadside LiDARs see")
    parser.add_argument("--no-view", action="store_true", help="don't open the ML car's split-screen window")
    parser.add_argument("--traffic", type=int, default=15, metavar="N", help="how many other vehicles")
    parser.add_argument("--walkers", type=int, default=8, help="how many pedestrians")
    parser.add_argument("--seconds", type=float, default=300.0, help="simulated seconds to run; 0 runs until Ctrl+C")
    parser.add_argument("--seed", type=int, help="makes traffic spawn the same way each run")
    parser.add_argument("--carla-exe", help="path to CarlaUE4.exe, if it isn't found automatically")
    parser.add_argument("--leave-running", action="store_true", help="keep CARLA and Docker up afterwards")
    args = parser.parse_args()
    if args.roadside and args.free_drive:
        parser.error("--roadside only works with the tests; use the commands under Roadside detection in README.md to drive with it")

    if args.lidar_view and not args.roadside:
        parser.error("--lidar-view needs --roadside, as the LiDARs only exist when the detector is running")

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
    launcher.log(f"Logs are in {LOG_DIR}.")
    return code


if __name__ == "__main__":
    sys.exit(main())
