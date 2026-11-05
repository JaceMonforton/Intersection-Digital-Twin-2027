# Models

None of these models are in the repository. Weights are ignored by git, because this repo is public and
the Waymo Open Dataset licence should be checked before sharing anything trained on it. Without them
the scripts run on labelled stand-ins and say so at startup.

Train `waymo_trajectory.onnx` yourself with `training/`; see "Train the trajectory model" in the main
README. It predicts the ML car's path, for both its line and the brake. Other cars use road-following.

| File | Used by | Without it |
| ---- | ------- | ---------- |
| `waymo_trajectory.onnx` | `scripts/waymo_predictor_node.py` | Follows the road at the current speed |
| `centerpoint_3d.onnx` | `scripts/central_gpu_processor.py` | CARLA ground-truth actor positions and boxes |
| `driving_policy.onnx` | The ML car's driver (being built) | Refuses to run: there is no stand-in, so nothing can be mistaken for model driving |

## `driving_policy.onnx`

`scripts/driving_features.py` builds the input and decodes the output, so that recording, training and driving all agree.

- **Input:** `float32 [N, 76]`, in this order:
  - speed in m/s (1)
  - route one-hot: left, straight, right (3)
  - stop line of a light that isn't green: present, then distance along the route in metres, capped at 35 m (2)
  - route points `(x, y)`: the one nearest the car, then 10 more 2 m apart (22)
  - the 6 nearest cars within 40 m, nearest first: `(present, x, y, vx, vy)` (30)
  - the 3 nearest pedestrians within 20 m, nearest first: `(present, x, y, vx, vy)` (15)
  - on a left turn: a car is due within 10 s (present), the time-to-collision (seconds until the first car reaches the route, keeping its speed and heading, ignoring cars with a light that isn't green before the route), and metres along the route to the start of the turn, negative once inside it (3)
- **Output:** `float32 [N, 2]`: steering from -1 to 1, positive to the right as in CARLA, and the target speed 1 s from now, in m/s. A small fixed controller in `scripts/driving_features.py` turns the target speed into throttle and brake; it isn't part of the model.
- **Frame:** the car's own: origin at the car, x forward, y **left**, in metres and m/s. Empty actor slots are all zeros. Scaling lives inside the model.
- **Bad output:** a value that isn't finite makes the car brake fully.
- **Two parts:** `training/train_left_turns.py` exports one model: the general driver from `driving_general.onnx`, frozen, plus a correction network that only acts on a left route. Off a left route the output is exactly the general driver's. The general driver doesn't use the three left-turn inputs.

## `waymo_trajectory.onnx`

- **Input:** `float32 [N, 10, 4]`: N actors, 10 past frames at 10 Hz (oldest first), each `(x, y, vx, vy)`.
- **Output:** `float32 [N, 30, 2]`: 30 future positions `(x, y)` at 10 Hz.
- **Frame, for both:** each actor's own ego frame at the latest history frame: origin at the actor,
  x forward, y **left**. `waymo_predictor_node.py` handles the conversion from CARLA's left-handed frame.

## `centerpoint_3d.onnx`

- **Input:** `float32 [N, 4]` or `[1, N, 4]`: one LiDAR sweep as `(x, y, z, intensity)`. If N is a fixed number, the sweep is evenly thinned or padded with zero points to fit.
- **Output:** `float32 [M, 9]`: `(x, y, z, length, width, height, yaw, score, class)`. x, y, z is the box centre, sizes are full lengths in metres, and yaw is in radians, counter-clockwise.
- **Frame, for both:** x forward, y **left**, z up, centred on the junction, with the road at `z = MODEL_GROUND_Z` (-1.8 m). The sweeps from both LiDARs are merged into this one frame. `central_gpu_processor.py` converts from CARLA's left-handed sensor frame. Boxes scoring below `DETECTION_THRESHOLD` (0.5) are dropped.

The sensors are two `sensor.lidar.ray_cast` at opposite corners of the junction, 7 m up: 64 channels, 300,000 points/s each, 80 m range, aimed from 0 to -45 degrees. They only spawn when a detector is in use. Measured with `training/check_lidar_coverage.py` on the 570 road points through the junction and 45 m either side: a car gets at least 3 points at 94.7% of them, and at 100% within 30 m of the centre, with a median of 128 points there. A pedestrian gets at least 3 at 57.4%, and at 92.5% within 30 m, with a median of 10 there. A merged sweep has about 58,000 points.

**A stock CenterPoint is unlikely to work on this data, even with the shapes matching.** Published models are trained on roof-mounted LiDAR, and this is a roadside view merged from two poles. Expect to train on these sweeps. Many CenterPoint exports also take voxelized tensors rather than raw points, and those need a different input step than `fit_model_input`.
