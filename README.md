<div align="center">

# Smart Digital Twin of an Urban Intersection

**A CARLA and Eclipse Ditto simulation where a trained model drives a car through an intersection, while a separate safety system watches from above and brakes it before a collision.**

![Simulation](https://img.shields.io/badge/simulation-working-brightgreen)
![Hardware](https://img.shields.io/badge/physical%20vehicles-untested-orange)
![Python](https://img.shields.io/badge/python-3.10-blue)
![CARLA](https://img.shields.io/badge/CARLA-0.9.16-green)
![Docker](https://img.shields.io/badge/docker-compose-2496ED)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)

<img src="docs/images/near_miss.gif" width="720" alt="Overhead view of a magenta car turning left across an oncoming car. The brake stops it, the oncoming car passes close by, and the magenta car finishes its turn.">

*An unsafe left turn, from the earlier scripted no-traffic run. The safety brake stops the magenta ML car 0.21 m from the oncoming car, which passes, and then it finishes the turn.*

[What it does](#what-it-does) · [Run it](#run-it) · [Test results](#test-results) · [How it works](#how-it-works) · [Project status](#project-status) · [Limitations](#known-limitations) · [Training](#train-the-driving-model)

</div>

---

## What it does

You can't stage near-collisions on a real road, so this project studies them in a digital twin of an intersection. It's a university capstone research project and a proof of concept, not for use on public roads.

Running it builds five test situations at an intersection in CARLA's Town10. The magenta **ML car** is driven by a trained model, which chooses its steering and target speed, with a fixed PID controller working the pedals. In normal driving the other cars are CARLA's autopilot; the hazard cars in the tests are scripted. While that happens, the system:

- **Tracks every vehicle** at `Town10_Intersection_01` from an overhead gantry, keeping the ML car's state in an Eclipse Ditto digital twin.
- **Predicts each vehicle's next 3 seconds.** The ML car uses a [model trained on Waymo data](#train-the-trajectory-model) (the magenta line) and the other cars follow the road (the red lines).
- **Scores the risk** with surrogate safety metrics: time-to-collision (TTC), post-encroachment time (PET) and deceleration rate to avoid a crash (DRAC).
- **Brakes the ML car** when a collision is predicted, sending the command through Ditto and MQTT to a safety gate that cuts the motors. In the simulation those motors are simulated.
- **Follows the ML car** in a split-screen window, in first and third person, showing the model's live inputs and outputs, the safety checks, recent events, and who's in control.
- **Reports the result:** a pass or fail for each test, with what was measured, and a live table in the terminal while it runs.

<p align="center"><img src="docs/images/ml_car_view.jpg" width="800" alt="The ML car's window while it waits inside the junction to turn left. First and third person views are on top. Below them are cards for what the model sees, its view of the road, what it decides, and safety and events."></p>

*The ML car's window during the left-turn test. It's waiting inside the junction: the next car reaches its route in 2.8 s, so the model asks for almost no speed.*

These pictures are from the earlier scripted left turn, which shows how the safety brake works when a car really does pull out too soon:

| It pulls out too soon | The brake stops it | It finishes the turn | The safe turn, on green, in traffic |
| :---: | :---: | :---: | :---: |
| <img src="docs/images/unsafe_approach.jpg" alt="The magenta car has started its turn across the oncoming car."> | <img src="docs/images/near_miss.jpg" alt="The moment the magenta car is closest to the oncoming car."> | <img src="docs/images/finishes_turn.jpg" alt="The magenta car continues through the junction."> | <img src="docs/images/safe_turn.jpg" alt="The magenta car turning with its own green light."> |

The ML car has a magenta box and predicted path, other cars have yellow boxes and red paths.

---

## Run it

### You'll need

- Windows with **Docker Desktop** (WSL2 backend)
- **Python 3.10**
- **CARLA 0.9.16** and a GPU that can run it. The client and server versions must match; see the [CARLA install guide](https://carla.readthedocs.io/en/latest/start_quickstart/)

### Install

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install config\carla-0.9.16-cp310-cp310-win_amd64.whl
```

### Start everything with one command

```powershell
python run_simulation.py
```

This starts whatever isn't already running (Docker, the Ditto stack, CARLA), opens the ML car's window, runs the five tests in `scripts/run_professor_test_suite.py`, writes the results, then shuts down only what it started. A first CARLA launch can take several minutes. It needs `models/driving_policy.onnx`, which isn't in the repository: see [Train the driving model](#train-the-driving-model).

Only run one copy at a time. A second one on the same CARLA disturbs the first, because it re-provisions the twins and changes the world's settings.

| Test | What must happen |
| ---- | ---------------- |
| 1. Traffic light | Stops before a red line, waits, pulls away on green and gets back up to speed |
| 2. Left turn across traffic | Pulls into the junction, gives way to oncoming cars, and finishes the turn in a gap with no collision, at least 1.5 m from other cars and without the safety gate firing |
| 3. Blind corner | A speeding car from behind a truck runs its red light; the safety gate fires, reaches MQTT, and the car stops with no collision |
| 4. Cut-in | A car moves into the ML car's lane and slows; no collision and no leaving the lane |
| 5. Light change during a left turn | While the car waits inside the junction the light goes yellow; it finishes the turn once no oncoming car is still coming |

While the tests run, the terminal shows events as they happen above one table of live values, then a table of results at the end:

```text
[SCENARIO] === [TEST 2/5] UNPROTECTED LEFT TURN & GAP ACCEPTANCE ===
+-------------------------------------------------------------+
| [TEST 2/5] UNPROTECTED LEFT TURN & GAP ACCEPTANCE   t=3.6 s |
+-------------------------------------------------------------+
| Target speed       16.8 km/h | Min TTC                n/a s |
| Actual speed       18.0 km/h | Path TTC               n/a s |
| Steering               +0.01 | PET                    n/a s |
| Throttle                27 % | Safety brake             off |
| Brake pedal              0 % | Closest car            n/a m |
| Off lane              0.49 m | MQTT               connected |
+-------------------------------------------------------------+
```

Results go to `output/PROFESSOR_DEMO_SUMMARY.md` and `output/PROFESSOR_TEST_SUITE_RESULTS.json`, and every tick's values to `output/live_telemetry.csv`.

| Option | What it does |
| ------ | ------------ |
| `--free-drive` | Normal driving in traffic instead of the five tests |
| `--seconds 120` | With `--free-drive`, how long to drive; 0 runs until Ctrl+C |
| `--traffic 30` and `--walkers 15` | With `--free-drive`, changes the traffic, which defaults to 15 cars and 8 pedestrians |
| `--quiet` | Leaves out the test suite's live table of values |
| `--no-view` | Doesn't open the ML car's window |
| `--seed 1` | Spawns traffic the same way each run |
| `--leave-running` | Keeps CARLA and Docker up, for faster repeat runs |
| `--carla-exe <path>` | Points at `CarlaUE4.exe`, if it isn't found automatically. Setting `CARLA_EXE` does the same |

Logs go to `output/logs/`. The ML car has no driver without its model, so the launcher refuses to run. For the other models, the scripts use the stand-ins described under [Known limitations](#known-limitations) and say so at startup.

To run or repeat single tests with the simulation already up, use the suite directly. `--repeat` saves to `output/repeat_results.json` instead of the reports.

```powershell
python scripts/run_professor_test_suite.py --only 2
python scripts/run_professor_test_suite.py --only 2 5 --repeat 20 --seed 500
```

<details>
<summary><b>Run the pieces yourself</b></summary>

Run these from the repository root, each long-running one in its own terminal.

**1. Start the stack and set up the twins.** Wait 60 to 90 seconds for Ditto between these two commands. Re-running the second one resets the Things.

```powershell
docker compose -f docker/docker-compose.yml up -d
python scripts/setup_entities.py
```

**2. Start CARLA, then the processor.** It must come first, because it owns the simulation: it loads `Town10HD_Opt`, enables synchronous mode at 10 Hz and drives `world.tick()`. Wait for `Master tick running at 10 Hz`, and exit with Ctrl+C so it can restore CARLA's settings.

```powershell
python scripts/central_gpu_processor.py
```

**3. Start the safety gate.** `--driver sim` uses no motors. On a vehicle, use `--broker <central machine IP>`; it only needs `pip install paho-mqtt`.

```powershell
python scripts/physical_ml_subscriber.py --driver sim
```

**4. Start the predictor, the traffic and the ML car's window.** Traffic needs `--asynch`, so that it follows the processor's clock.

```powershell
python scripts/waymo_predictor_node.py
python scripts/test_traffic.py --asynch --near-view --safe -n 15 -w 8 --tm-port 8000 --car-lights-on
python scripts/ml_car_view.py
```

**5. Let the model drive the ML car,** with its report at the end.

```powershell
python scripts/traffic_monitor.py --duration 60
```

The earlier scripted left turns are `python scripts/traffic_left_turn.py --case both`, and with no traffic, `python scripts/left_turn_scenario.py --case both`.

**To mirror your own vehicles instead of running the scenario,** start an agent for each Thing. Each one opens a chase-camera window and spawns a CARLA vehicle named after the Thing, which is how the processor recognises it. These twins hold still while the brake is on, and `human_vehicle` has no safety gate.

```powershell
python scripts/ditto_carla_agent.py
python scripts/ditto_carla_agent.py --thing human_vehicle
```

**The brake latches.** Outside the scenario, release it with:

```powershell
python scripts/setup_entities.py --clear-brake
```

**Shut down** with this. Add `-v` only if you want to wipe the Ditto data.

```powershell
docker compose -f docker/docker-compose.yml down
```

</details>

---

## Test results

All five tests on the current model, each scored on what was measured, with no retries:

| Test | Runs passed | Closest gap, final run |
| ---- | :---: | :---: |
| 1. Traffic light | 5 of 5 | n/a |
| 2. Left turn across traffic | 20 of 20 | 3.70 m |
| 3. Blind corner | 5 of 5 | 8.78 m |
| 4. Cut-in | 5 of 5 | 3.55 m |
| 5. Light change during a left turn | 20 of 20 | 3.64 m |

Tests 2 and 5 vary the oncoming traffic on each run. A harder set, with gaps close to the safe limit, passed 40 of 40 left turns with no collisions and no use of the safety gate, and a closest approach of 3.54 m.

This doesn't prove the car passes every time: 60 of 60 left turns still allows a failure rate of up to about 1 in 20 (95% confidence). All of them were on one approach to one junction, with oncoming cars scripted at 12 m/s.

### Normal driving

Six drives of 300 simulated seconds in traffic of 15 cars and 8 pedestrians, with the safety brake ignored so that these are the model's own mistakes:

| | Distance | Collisions | Red lights | Times it lost its lane |
| - | :---: | :---: | :---: | :---: |
| General driver alone | 5.62 km | 0 | 0 | 0 |
| With the left-turn correction | 5.54 km | 1 | 0 | 0 |

The target is 0.11 collisions per km, which is 1.25 times autopilot's rate in the same traffic. The one collision was an Audi hitting the ML car while it was stopped, on a drive with no left turns, where the two models behave identically. Its cause wasn't examined. Together that's 1 in 11.2 km, which is too little driving to show the target is met, and the 10-run pass test in `training/pass_test.py` hasn't been run on this model.

---

## Earlier scripted left-turn results

These are from before the model drove. The ML car follows a scripted plan, so they measure the **safety brake**, not the driving model.

### In real traffic

The ML car waits at a red light to turn left across autopilot cars that have the green. The main junction's lights are held in that phase by the scenario, and every car obeys them.

- **Unsafe:** the ML car deliberately runs its red light, timed to meet an oncoming car.
- **Safe:** it waits while the oncoming light goes amber then red and its own turns green, then turns.

| Over 11 full runs | Unsafe | Safe |
| - | :---: | :---: |
| **Verdict** | UNSAFE, 11 of 11 | SAFE, 11 of 11 |
| **Contact** | none | none |
| **Brake fired** | 9 of 11 | never |
| **Closest gap between cars** | 0.18 m to 7.96 m | 1.35 m to 7.31 m |

The other cars are real, so the unsafe gap varies with how close an oncoming car is and how it reacts. In the 2 runs where the brake didn't fire, the gap stayed at 3.5 m or more. In free driving, over 2 runs of 45 s with the final red-light check, traffic ran 1 red light and the ML car ran none, stopping for each. The ML car touched another car once, in free driving.

A run is **UNSAFE** if the ML car ran a red, the car outlines overlap or come within 1 m, TTC drops under 1.5 s, PET drops under 1.5 s, DRAC goes over 3.35 m/s², or the brake fires. After the hazard passes, the scenario releases the brake as an operator would, so the car can finish its turn. Nothing else auto-releases it.

### Without traffic

`--clean` runs the same turn against one scripted car that never reacts, so it's repeatable, and compares it with the same plan replayed with no brake. These runs varied only the ML car's arrival time:

| Arrives after the oncoming car | Without brake | With brake | Closest gap with brake |
| :---: | :---: | :---: | :---: |
| 0 to 1.25 s | contact | no contact | 0.39 m |
| **1.5 s (the default)** | **contact** | **no contact** | **0.21 m** |
| 1.75 s | contact | no contact | 0.23 m |
| 2 s | miss by 0.11 m | never fires | 0.11 m |
| 2.5 s | miss by 0.67 m | never fires | 0.67 m |
| 3 s (safe) | miss by 1.10 m | never fires | 1.10 m |

Every colliding timing tested was avoided here. A brake that relies on the trained model needs room: on Town01's smaller junction, the same brake prevented none of them, because the model only recognises a turn once the car is already swinging across the lane.

---

## How it works

Each vehicle has a **digital twin** in Eclipse Ditto that holds its latest state. Anything that writes the same telemetry, a real robot or a test script, drives everything downstream the same way, so the vehicles are decoupled from the simulator.

```mermaid
flowchart LR
    subgraph GPU["Central GPU machine"]
        C["CARLA<br/>Town10"] --> P["central_gpu_processor<br/>clock, tracking, TTC / PET / DRAC"]
        C --> W["waymo_predictor_node<br/>3 s paths"]
        S["ml_driver + driving model<br/>tests or free driving"] --> C
        T["test_traffic<br/>autopilot cars"] --> C
        V["ml_car_view<br/>split screen"] --- C
    end
    subgraph Infra["Docker stack"]
        N["nginx :8080"] --> D["Eclipse Ditto<br/>twin state"]
        Q["Mosquitto<br/>MQTT :1883"]
    end
    subgraph Vehicle["Vehicle, or simulated motors"]
        G["physical_ml_subscriber<br/>safety gate"] --> M["Motors"]
    end
    N -->|WebSocket events| P
    P -->|"HTTP: emergency_brake"| N
    D <--> Q
    Q -->|device/vehicle_ml/commands| G
    S -->|polls the gate over HTTP| N
```

**The brake fires** when it can help: never for a car coming up from behind, or while the ML car is already stopped. It has two triggers. Straight-line TTC under 1.5 s catches anything dead ahead, except stopped cars while the ML car is mid-turn, when it's only pointing at them momentarily. Predicted paths meeting within 3 s catches crossing traffic, and ignores stopped cars and cars being followed, which are ordinary traffic.

**The driving model** takes 76 numbers: its speed, route, any stop line that isn't green, the road ahead, nearby cars and pedestrians, and on a left turn, the time-to-collision with the first car due to reach its route. It outputs steering and a target speed 1 s ahead. A fixed PID controller, which isn't AI, turns that speed into pedals, and the car moves with real CARLA physics. It sees positions from CARLA, not cameras or LiDAR.

**It's two networks exported as one file.** The general driver copies CARLA's autopilot, from recordings plus takeovers where autopilot rescued the model. Those recordings have no left turns across moving traffic, because this junction gives one approach a green at a time. So a fixed rule built on time-to-collision teaches them, and a small correction network learns from it. Retraining the whole model on those lessons made other driving worse (4 collisions in 5.51 km), so the general driver stays frozen and the correction only acts on a left route. Off a left route, the output matched the general driver exactly on 17,751 recorded samples. The rule isn't used when driving.

**On a left turn** the car pulls about 9 m into the junction and waits short of the oncoming lanes. It sets off once no car is due within its time to get across. If the light goes yellow while it waits, it finishes the turn once no oncoming car is still coming.

**The ML car's predicted path uses only the trajectory model**, for both its line and the brake. It's a separate model from the driving model, and it can only predict. Other cars are real CARLA physics on autopilot.

**The safety gate fails safe.** It holds the motors off before it receives its first state, whenever it loses the broker, and until it hears an explicit release.

| Thing | Role |
| ----- | ---- |
| `org.example:intersection_01` | The intersection; holds the latest `SafetyMetrics` |
| `org.example:vehicle_ml` | The ML vehicle; the only one with a `SafetyGate` |
| `org.example:human_vehicle` | The other vehicle |

<details>
<summary><b>Repository layout</b></summary>

```
run_simulation.py    Starts everything and runs the five tests, or normal driving
docker/              Ditto stack: docker-compose.yml, mosquitto.conf, nginx.conf
config/              Ditto policies, Things and connections, and the CARLA 0.9.16 client
models/              Where trained models go, and the contract they must meet
training/
  record_driving.py           Records autopilot, for the general driver
  record_corrections.py       Records autopilot rescuing the model
  train_driving.py            Trains the general driver
  record_left_turns.py        Records left-turn lessons from a time-to-collision rule
  train_left_turns.py         Adds them to the frozen general driver as a left-only correction
  pass_test.py                The 10-run pass test against autopilot
  convert_waymo.py, train_trajectory.py    Trains the trajectory model
scripts/
  run_professor_test_suite.py The five tests and their reports
  ml_driver.py                Drives the ML car from the model, and its driving report
  driving_features.py         The model's inputs and outputs, and the PID controller
  model_panel.py              The window's cards, and the live values sent to it
  central_gpu_processor.py    Simulation clock, overhead sensors, fusion, metrics, brake trigger
  waymo_predictor_node.py     History buffer, 3 s path prediction, drawn lines
  physical_ml_subscriber.py   Safety gate that cuts the motors (runs on the vehicle)
  traffic_left_turn.py        Earlier scripted safe vs unsafe left turn across real traffic
  traffic_monitor.py          The ML car in normal traffic, its lane and route logic, and the traffic report
  ml_car_view.py              Split-screen window following the ML car
  left_turn_scenario.py       Earlier scripted left turn with no traffic, and its analysis
  setup_entities.py           Provisions Ditto from config/
  ditto_carla_agent.py        Mirrors one Thing into CARLA, with a chase camera
  test_carla_connection.py    CARLA connection check
  test_traffic.py             Spawns the autopilot cars and pedestrians
tests/               Unit tests and the end-to-end latency benchmark
hardware/            KUKSA to Ditto publisher, for physical vehicles
bridge/              Older standalone Ditto to CARLA sync
docs/images/         The pictures on this page
```

</details>

<details>
<summary><b>Ports and telemetry format</b></summary>

| Port | Service |
| ---- | ------- |
| 8080 | nginx, proxying the Ditto API and WebSocket. It injects the `x-ditto-pre-authenticated` header |
| 8081 | Ditto gateway, direct access |
| 1883 / 9001 | Mosquitto (MQTT / WebSocket) |
| 27017 | MongoDB (data is kept in the `mongo-data` volume) |

> **Warning:** Nothing here authenticates. Anyone who can reach 8080 can write to the twins, and anyone who can reach 1883 can publish telemetry or brake commands. Keep these ports on a trusted lab network.

Vehicles publish [Ditto Protocol](https://eclipse.dev/ditto/protocol-overview.html) JSON to `telemetry/<thing name>`. The processor reads `location` + `velocity` (m/s), or `kinematics` as written by `hardware/kuksa_to_ditto.py` (`x`, `y`, `heading` in degrees, `speed` in km/h). Poses must already be in the CARLA frame.

```json
{
  "topic": "org.example/human_vehicle/things/twin/commands/merge",
  "headers": { "content-type": "application/merge-patch+json", "response-required": false },
  "path": "/features",
  "value": {
    "location": { "properties": { "x": 12.0, "y": 4.5, "z": 0.0 } },
    "velocity": { "properties": { "vx": 1.5, "vy": 0.0 } }
  }
}
```

</details>

---

## Project status

| Area | State |
| ---- | ----- |
| Docker stack, Ditto twins, MQTT safety path | **Working**, tested live |
| One-command launcher | **Working**, including starting CARLA and the Ditto stack. Starting Docker Desktop itself isn't tested |
| Master tick, overhead view, on-screen overlays | **Working** in CARLA |
| Driving model | **Working in simulation.** All five tests pass. The 10-run pass test against autopilot hasn't been run on this model |
| Left turn across traffic | **Working.** Tests 2 and 5 passed 20 of 20 each on one approach to one junction, with scripted oncoming cars at 12 m/s |
| Free driving, reports, live terminal table | **Working** |
| Split-screen ML car window | **Working**, showing the model's real inputs and outputs |
| Path prediction | **Working.** The ML car uses a [model trained on Waymo data](#train-the-trajectory-model), falling back to road-following without it. Other cars follow the road |
| Vehicle detection | **CARLA's true positions, as a stand-in.** Two corner LiDARs are in place and their sweeps are merged for a model, but there's no model yet, so they only spawn with `--detector onnx` |
| Physical vehicles (Rosmaster R2, KUKSA) | **Written, not yet run on hardware.** The motor drivers and KUKSA publisher have never run |
| Fully autonomous physical vehicle (Autoware) | **Not started.** Planned last |
| Lab-to-CARLA coordinate mapping, SkyEye view, Kafka logging, AI recommendations | **Not started** |

### Engineering targets

These are the targets from the project report, against what has been measured so far.

| Requirement | Target | Measured so far |
| ----------- | ------ | --------------- |
| End-to-end latency (REQ-SYS-002) | At most 100 ms, sensor to CARLA rendering | About 60 ms (p95) from Ditto write to simulated motor cut, on one machine. The sensor-to-rendering path isn't measured |
| Update rate (REQ-SYS-003) | At least 10 Hz (target 20 Hz) | 10.0 Hz simulation tick, at real-time speed with 15 cars |
| Safety metrics (REQ-SYS-011) | TTC and headway; flag TTC under 1.5 s | TTC, PET and DRAC done, and the 1.5 s flag works. Headway isn't implemented |
| Telemetry payload (REQ-SYS-001) | Full state vector, JSON over MQTT/HTTP | Position and velocity only |
| Position and heading accuracy (REQ-SYS-004, 005) | Within 5 cm and 2 degrees | Not measured; needs physical vehicles |
| Bird's-eye rendering (REQ-SYS-006, 007) | At least 30 FPS, at most 100 ms delay | Not measured |
| Scalability (REQ-SYS-008, 009) | 10 vehicles within 150 ms | Not measured |
| Logging (REQ-SYS-010) | 10 Hz, UTC timestamps at 1 ms precision | Scenario data only, in simulation time |

---

## Known limitations

- **Left turns are only trained and tested on one approach to one junction,** against oncoming cars scripted at 12 m/s. The correction acts on every left route, so left turns elsewhere have changed, but have only had normal-driving use.
- **The model's time-to-collision assumes each car keeps its speed and heading,** so it won't foresee a car turning across its route. A car that runs its red light with room to stop is left to the safety gate.
- **The driving model is limited by what it copied.** It tops out at about 20 km/h, as autopilot did in the recordings, and sees positions from CARLA rather than cameras or LiDAR.
- **The brake is only tested at one junction.** On Town01's smaller junction it prevented none of the same collisions, so don't assume it carries over.
- **The trajectory model can't see a turn signal or the map.** It only recognises a turn once the car has started it.
- **Pedestrians aren't tracked.** They walk around, but neither the brake nor the reports consider them.
- **More traffic is less stable.** At 40 cars, the brake's repeated stops could strand the ML car in the junction and cause contact. The default of 15 didn't.
- **In the earlier scripted left turns the ML car isn't physics-driven.** They move it directly along its plan. When the model drives, it uses real physics.
- **Cars are recycled.** Autopilot routes wander off, so cars far from the intersection are moved back onto nearby roads, out of sight, to keep it busy.
- **Other cars' boxes show in the ML car's window.** CARLA draws debug shapes into every camera. The ML car's own box is hidden by drawing it above its cameras' view.
- **The path check only works on CARLA actors.** Telemetry alone and model detections fall back to the 1.5 s TTC rule. That includes physical vehicles until lab positions are mapped into CARLA.
- **No trained models are in the repo.** The driving and trajectory models are trained locally, and the trajectory model stays out of git until the Waymo licence is checked. There's no detection model at all, and a stock CenterPoint is unlikely to work from an overhead sensor. See [models/README.md](models/README.md).
- **The trajectory model can't anticipate a turn.** It only sees 1 s of motion, with no map or turn signal. See [its results](#train-the-trajectory-model).
- **The vehicle can't tell if the processor has died.** The gate fails safe if the broker link drops, but a crashed processor simply stops sending hazards.
- **A silent Wi-Fi drop took 3.8 s to detect** in testing, which is longer than the 1.5 s brake window. Lowering `MQTT_KEEPALIVE` shortens it.
- **Live PET is predicted.** The `SafetyMetrics` Thing holds PET extrapolated from current velocities. The scenario report's PET is measured after the fact.
- **`bridge/ditto_carla_sync.py` is out of date.** It still points at `org.ovin:my-vehicle` on `192.168.0.100`.

---

## Train the driving model

Recording needs the main environment, which has `carla`, and CARLA with the processor running. Training needs an environment with `torch`, like the one in the next section's step 1. Each step uses what the one before it saved under `data/driving/`, which git ignores.

**1. Record autopilot, then convert it.** Recording also needs `test_traffic.py` running. Conversion doesn't need CARLA.

```powershell
python training\record_driving.py --minutes 60
python training\record_driving.py --convert
```

**2. Train the general driver.** It must go to `driving_general.onnx`, because the next steps build on it.

```powershell
python training\train_driving.py --output models\driving_general.onnx
```

**3. Record takeovers and train again,** which teaches it to recover from its own mistakes. This needs a model to drive, so copy `driving_general.onnx` to `driving_policy.onnx` first. Good numbers from training only mean it copies well: only driving it in CARLA shows that it can drive.

```powershell
python training\record_corrections.py --episodes 6 --minutes 5
python training\train_driving.py --output models\driving_general.onnx
```

**4. Add left turns.** Record lessons from the rule, then record more with the model driving and the rule labelling, which captures the model's own mistakes. Train between rounds, so that the model-driven round uses the newest model.

```powershell
python training\record_left_turns.py --runs 20
python training\train_left_turns.py
python training\record_left_turns.py --runs 20 --model-share 1.0 --light-change-share 0.25
python training\train_left_turns.py
```

**5. Check it,** with the tests, then the pass test against autopilot. Run the pass test once with the brake ignored to measure the model alone, and once with it active. It needs `test_traffic.py` running.

```powershell
python scripts\run_professor_test_suite.py --only 2 5 --repeat 20
python training\pass_test.py --ignore-brake
python training\pass_test.py
```

---

## Train the trajectory model

This trains `models/waymo_trajectory.onnx` on the [Waymo Open Motion Dataset](https://waymo.com/open), which needs a free account and is licensed for non-commercial use. The predictor and processor load the file automatically on their next run, and deleting it returns everything to road-following.

**1. Install the training tools,** in a separate environment. The `carla` package isn't needed here.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r training\requirements.txt
```

**2. Download some shards** from `gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/` into `data/waymo/training/` and `data/waymo/validation/`. Git ignores `data/`.

**3. Convert them, then train.** Training takes about a minute on an RTX 4080 with 10 shards.

```powershell
python training\convert_waymo.py data\waymo\training data\waymo\converted\training.npz
python training\convert_waymo.py data\waymo\validation data\waymo\converted\validation.npz
python training\train_trajectory.py
```

The main Python environment also needs `onnxruntime` to load the model, which `requirements.txt` installs.

### Results

The model is a small GRU that learns a correction on top of constant velocity. These are mean position errors over 3 s, on 2 Waymo validation shards, after training on 10 of the 1,000 training shards. The best epoch was picked on those same shards, so they're slightly optimistic.

| Vehicles | Samples | Constant velocity | Trained model |
| -------- | :---: | :---: | :---: |
| All | 4,865 | 1.54 m | **0.82 m** |
| Going straight | 4,482 | 1.40 m | **0.79 m** |
| Turning left | 219 | 3.16 m | **1.14 m** |
| Turning right | 164 | 3.12 m | **1.26 m** |

In a CARLA left turn, road-following predicts better, though it's only an option when the road and the car's intent are known:

| The turning car, with a turn coming within 3 s | Mean error |
| ---- | :---: |
| Constant velocity | 3.25 m |
| Trained model | 1.89 m |
| Road-following | 1.03 m |

The road-following figure is a best case, scored along the lane the car really took. The model's weakness is what it can't see: at the stop line its line goes straight on even with the car signalling left, and at the end of a turn it keeps curving after the road straightens. Those numbers are from Town01.

---

## Testing

```powershell
pytest
```

| Test file | Needs |
| --------- | ----- |
| `test_safety_logic.py`, `test_left_turn.py`, `test_ml_driving.py`, `test_traffic_monitor.py`, `test_driving_features.py`, `test_ml_driver.py`, `test_model_panel.py`, `test_pass_test.py`, `test_professor_suite.py`, `test_record_driving.py`, `test_record_corrections.py`, `test_record_left_turns.py` | Nothing running. Some need `paho` or `pygame`, and skip themselves without them |
| `test_agent_brake.py`, `test_ml_car_view.py` | The `carla` and `pygame` packages; skip themselves without them |
| `test_training_data.py`, `test_train_driving.py` | Nothing running. Their model tests need PyTorch, and skip themselves without it |
| `test_mqtt_pipeline.py` | The Docker stack, not CARLA. Skips itself when the stack is down, as on CI, and resets Thing state. Add `-s` to see the timings |

---

## Troubleshooting

<details>
<summary><b>The launcher fails, or a script dies during a run</b></summary>

Check `output/logs/`, which has a log for each background script. A CARLA client aborting with `_episode != nullptr` was seen on a freshly started server; the launcher retries the processor once.

</details>

<details>
<summary><b>A test fails or times out, or a port is already in use</b></summary>

Check that only one copy is running. A second `run_simulation.py` on the same CARLA stops with `Only one usage of each socket address`, and on its way out it can make the first copy's test fail, as happened to a cut-in test that lost its lane. Run it again on its own.

</details>

<details>
<summary><b>CARLA import or connection errors</b></summary>

The client and server versions must match. The wheel in `config/` is for CARLA 0.9.16 and Python 3.10 on Windows.

</details>

<details>
<summary><b>Other scripts conflict with the processor</b></summary>

Only one process may tick the world. `scripts/test_traffic.py` ticks unless you pass `--asynch`, and `scripts/test_carla_connection.py` always does, so stop the processor first or use `test_traffic.py --asynch`.

</details>

<details>
<summary><b>Ditto resets or won't answer on Windows</b></summary>

Use `localhost`, not `127.0.0.1`. Requests to `127.0.0.1:8080` were reset in our setup while `localhost` worked.

</details>

<details>
<summary><b>PATCH requests are rejected</b></summary>

Ditto only accepts `Content-Type: application/merge-patch+json` for PATCH. Plain `application/json` is rejected, so use PUT if you want to send plain JSON.

</details>

<details>
<summary><b>Ditto data disappeared</b></summary>

`down -v` deletes the `mongo-data` volume. Plain `down` keeps it.

</details>

---

### Acknowledgements

Built on [CARLA](https://carla.org/), [Eclipse Ditto](https://eclipse.dev/ditto/), [Eclipse KUKSA](https://eclipse.dev/kuksa/), and [Eclipse Mosquitto](https://mosquitto.org/).
