# Models

Neither model is in the repository. Weights are ignored by git, because this repo is public and
the Waymo Open Dataset licence should be checked before sharing anything trained on it. Without them
the scripts run on labelled stand-ins and say so at startup.

Train `waymo_trajectory.onnx` yourself with `training/`; see "Train the trajectory model" in the main
README. It predicts the ML car's path, for both its line and the brake. Other cars use road-following.

| File | Used by | Without it |
| ---- | ------- | ---------- |
| `waymo_trajectory.onnx` | `scripts/waymo_predictor_node.py` | Follows the road at the current speed |
| `centerpoint_3d.onnx` | `scripts/central_gpu_processor.py` | CARLA ground-truth actor positions and boxes |

## `waymo_trajectory.onnx`

- **Input:** `float32 [N, 10, 4]`: N actors, 10 past frames at 10 Hz (oldest first), each `(x, y, vx, vy)`.
- **Output:** `float32 [N, 30, 2]`: 30 future positions `(x, y)` at 10 Hz.
- **Frame, for both:** each actor's own ego frame at the latest history frame: origin at the actor,
  x forward, y **left**. `waymo_predictor_node.py` handles the conversion from CARLA's left-handed frame.

## `centerpoint_3d.onnx`

- **Input:** `float32 [N, 4]` or `[1, N, 4]`: one LiDAR sweep as `(x, y, z, intensity)`. If N is a fixed number, the sweep is evenly thinned or padded with zero points to fit.
- **Output:** `float32 [M, 9]`: `(x, y, z, length, width, height, yaw, score, class)`. x, y, z is the box centre, sizes are full lengths in metres, and yaw is in radians, counter-clockwise.
- **Frame, for both:** x forward, y **left**, z up, centred under the gantry, with the road at `z = MODEL_GROUND_Z` (-1.8 m). `central_gpu_processor.py` converts from CARLA's left-handed sensor frame. Boxes scoring below `DETECTION_THRESHOLD` (0.5) are dropped.

The sensor is a `sensor.lidar.ray_cast` on the gantry 40 m above the junction: 64 channels, 100,000 points/s, 50 m range. Its FOV points down (-45 to -90 degrees), because the default FOV would never reach the road from that height. At that height and range it covers a radius of about 30 m and gives about 7,700 points per sweep, with roughly 35 on a car.

**A stock CenterPoint is unlikely to work on this data, even with the shapes matching.** Published models are trained on roof-mounted LiDAR, which sees the sides of nearby cars at high density. This sensor sees only their roofs, sparsely, from above. The -1.8 m ground shift puts the points in the height range such models expect, but it doesn't change that viewpoint. Expect to retrain on overhead sweeps, or lower the gantry. Many CenterPoint exports also take voxelized tensors rather than raw points, and those need a different input step than `fit_model_input`.
