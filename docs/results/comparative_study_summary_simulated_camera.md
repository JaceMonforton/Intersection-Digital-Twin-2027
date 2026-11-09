# Comparative study

Generated 2026-11-07T07:52:04+00:00. 270 runs; every figure is measured.

## Which config is best

Each config is scored out of 100 from its measured results: 50 no collision, 30 pass, 10 no false alarm, 10 TTC margin points (TTC margin: the low tail, P5, against 3 s). The weights were fixed before any results were seen.

### 0 background cars

| Rank | Config | Score | Collision rate | Pass rate | False alarms per run | TTC P5 |
| ---- | ------ | ----- | -------------- | --------- | -------------------- | ------ |
| 1 | camera_and_rsu (simulated) | **91.7** | 2.2 % | 97.8 % | 0.11 | 1.36 s |
| 2 | lidar_and_rsu | **81.3** | 11.1 % | 88.9 % | 0.04 | 0.18 s |
| 3 | lidar_only | **60.2** | 35.6 % | 60.0 % | 0.00 | 0.0 s |

Best: **camera_and_rsu** (91.7), then lidar_and_rsu (81.3), then lidar_only (60.2).
camera_and_rsu is simulated from CARLA's true positions and never sees a false object, so its score is an upper bound for a real camera, not a measured camera result. Of the configs with real perception, the best is **lidar_and_rsu**.

### 15 background cars

| Rank | Config | Score | Collision rate | Pass rate | False alarms per run | TTC P5 |
| ---- | ------ | ----- | -------------- | --------- | -------------------- | ------ |
| 1 | camera_and_rsu (simulated) | **72.6** | 9.1 % | 63.6 % | 0.39 | 0.58 s |
| 2 | lidar_and_rsu | **71.3** | 9.1 % | 61.4 % | 0.27 | 0.04 s |
| 3 | lidar_only | **50.9** | 42.2 % | 40.0 % | 0.00 | 0.0 s |

Best: **camera_and_rsu** (72.6), then lidar_and_rsu (71.3), then lidar_only (50.9).
camera_and_rsu is simulated from CARLA's true positions and never sees a false object, so its score is an upper bound for a real camera, not a measured camera result. Of the configs with real perception, the best is **lidar_and_rsu**.

### All traffic loads

| Rank | Config | Score | Collision rate | Pass rate | False alarms per run | TTC P5 |
| ---- | ------ | ----- | -------------- | --------- | -------------------- | ------ |
| 1 | camera_and_rsu (simulated) | **81.5** | 5.6 % | 80.9 % | 0.25 | 0.76 s |
| 2 | lidar_and_rsu | **76.2** | 10.1 % | 75.3 % | 0.16 | 0.06 s |
| 3 | lidar_only | **55.5** | 38.9 % | 50.0 % | 0.00 | 0.0 s |

Best: **camera_and_rsu** (81.5), then lidar_and_rsu (76.2), then lidar_only (55.5).
camera_and_rsu is simulated from CARLA's true positions and never sees a false object, so its score is an upper bound for a real camera, not a measured camera result. Of the configs with real perception, the best is **lidar_and_rsu**.

## 0 background cars, by scenario

| Config | Scenario | Score | Ran | Pass rate | Collision rate | False alarms | TTC mean | TTC P95 | TTC P5 | Clearance |
| ------ | -------- | ----- | --- | --------- | -------------- | ------------ | -------- | ------- | ------ | --------- |
| lidar_and_rsu | traffic_light | 100.0 | 5 of 5 | 100.0 % | 0.0 % | 0 | n/a | n/a | n/a | n/a (0) |
| lidar_and_rsu | left_turn | 100.0 | 5 of 5 | 100.0 % | 0.0 % | 0 | 6.34 s | 7.39 s | 5.29 s | 39.64 s (5) |
| lidar_and_rsu | blind_corner | 100.0 | 5 of 5 | 100.0 % | 0.0 % | 0 | 3.03 s | 3.03 s | 3.03 s | n/a (0) |
| lidar_and_rsu | cut_in | 100.0 | 5 of 5 | 100.0 % | 0.0 % | 0 | 3.24 s | 3.24 s | 3.24 s | 11.3 s (5) |
| lidar_and_rsu | yellow_light | 100.0 | 5 of 5 | 100.0 % | 0.0 % | 0 | 4.19 s | 5.25 s | 3.64 s | 15.94 s (5) |
| lidar_and_rsu | red_light_runner | 93.9 | 5 of 5 | 100.0 % | 0.0 % | 2 | 2.77 s | 3.03 s | 2.36 s | n/a (0) |
| lidar_and_rsu | hidden_pedestrian | 58.1 | 5 of 5 | 60.0 % | 40.0 % | 0 | 0.35 s | 0.69 s | 0.02 s | n/a (0) |
| lidar_and_rsu | oncoming_turner | 43.2 | 5 of 5 | 40.0 % | 60.0 % | 0 | 0.37 s | 0.37 s | 0.37 s | n/a (0) |
| lidar_and_rsu | clear_road | 100.0 | 5 of 5 | 100.0 % | 0.0 % | 0 | n/a | n/a | n/a | n/a (0) |
| lidar_and_rsu | **all** | 81.3 | 45 of 45 | 88.9 % | 11.1 % | 2 | 2.89 s | 5.4 s | 0.18 s | 22.29 s (15) |
| camera_and_rsu | traffic_light | 100.0 | 5 of 5 | 100.0 % | 0.0 % | 0 | n/a | n/a | n/a | n/a (0) |
| camera_and_rsu | left_turn | 100.0 | 5 of 5 | 100.0 % | 0.0 % | 0 | 6.0 s | 6.79 s | 5.22 s | 39.44 s (5) |
| camera_and_rsu | blind_corner | 98.9 | 5 of 5 | 100.0 % | 0.0 % | 0 | 2.84 s | 3.06 s | 2.66 s | n/a (0) |
| camera_and_rsu | cut_in | 96.0 | 5 of 5 | 100.0 % | 0.0 % | 2 | 3.32 s | 3.69 s | 3.12 s | 11.34 s (5) |
| camera_and_rsu | yellow_light | 100.0 | 5 of 5 | 100.0 % | 0.0 % | 0 | 4.16 s | 5.01 s | 3.65 s | 16.24 s (5) |
| camera_and_rsu | red_light_runner | 73.8 | 5 of 5 | 80.0 % | 20.0 % | 1 | 2.27 s | 3.0 s | 0.53 s | 9.1 s (1) |
| camera_and_rsu | hidden_pedestrian | 94.5 | 5 of 5 | 100.0 % | 0.0 % | 0 | 1.85 s | 2.28 s | 1.34 s | n/a (0) |
| camera_and_rsu | oncoming_turner | 91.0 | 5 of 5 | 100.0 % | 0.0 % | 2 | 2.05 s | 2.42 s | 1.5 s | n/a (0) |
| camera_and_rsu | clear_road | 100.0 | 5 of 5 | 100.0 % | 0.0 % | 0 | n/a | n/a | n/a | n/a (0) |
| camera_and_rsu | **all** | 91.7 | 45 of 45 | 97.8 % | 2.2 % | 5 | 2.98 s | 5.17 s | 1.36 s | 21.51 s (16) |
| lidar_only | traffic_light | 100.0 | 5 of 5 | 100.0 % | 0.0 % | 0 | n/a | n/a | n/a | n/a (0) |
| lidar_only | left_turn | 94.0 | 5 of 5 | 80.0 % | 0.0 % | 0 | 4.86 s | 7.9 s | 3.06 s | 39.06 s (5) |
| lidar_only | blind_corner | 10.0 | 5 of 5 | 0.0 % | 100.0 % | 0 | 0.0 s | 0.0 s | 0.0 s | 12.0 s (5) |
| lidar_only | cut_in | 100.0 | 5 of 5 | 100.0 % | 0.0 % | 0 | 3.24 s | 3.24 s | 3.24 s | 11.4 s (5) |
| lidar_only | yellow_light | 94.0 | 5 of 5 | 80.0 % | 0.0 % | 0 | 4.2 s | 5.18 s | 3.19 s | 15.8 s (5) |
| lidar_only | red_light_runner | 26.0 | 5 of 5 | 20.0 % | 80.0 % | 0 | 0.26 s | 0.78 s | 0.0 s | 11.3 s (3) |
| lidar_only | hidden_pedestrian | 10.0 | 5 of 5 | 0.0 % | 100.0 % | 0 | 0.0 s | 0.0 s | 0.0 s | n/a (0) |
| lidar_only | oncoming_turner | 58.2 | 5 of 5 | 60.0 % | 40.0 % | 0 | 0.05 s | 0.05 s | 0.05 s | n/a (0) |
| lidar_only | clear_road | 100.0 | 5 of 5 | 100.0 % | 0.0 % | 0 | n/a | n/a | n/a | n/a (0) |
| lidar_only | **all** | 60.2 | 45 of 45 | 60.0 % | 35.6 % | 0 | 1.83 s | 4.99 s | 0.0 s | 18.49 s (23) |

## 15 background cars, by scenario

| Config | Scenario | Score | Ran | Pass rate | Collision rate | False alarms | TTC mean | TTC P95 | TTC P5 | Clearance |
| ------ | -------- | ----- | --- | --------- | -------------- | ------------ | -------- | ------- | ------ | --------- |
| lidar_and_rsu | traffic_light | 82.9 | 5 of 5 | 60.0 % | 0.0 % | 0 | 4.56 s | 11.37 s | 1.47 s | n/a (0) |
| lidar_and_rsu | left_turn | 36.0 | 5 of 5 | 20.0 % | 40.0 % | 5 | 2.17 s | 6.15 s | 0.0 s | 39.77 s (3) |
| lidar_and_rsu | blind_corner | 96.9 | 5 of 5 | 100.0 % | 0.0 % | 0 | 2.77 s | 3.05 s | 2.08 s | n/a (0) |
| lidar_and_rsu | cut_in | 80.4 | 5 of 5 | 60.0 % | 0.0 % | 0 | 2.7 s | 4.74 s | 0.71 s | 8.9 s (1) |
| lidar_and_rsu | yellow_light | 76.0 | 5 of 5 | 20.0 % | 0.0 % | 0 | 4.78 s | 5.95 s | 3.52 s | 17.9 s (5) |
| lidar_and_rsu | red_light_runner | 92.9 | 4 of 5 | 100.0 % | 0.0 % | 2 | 2.87 s | 3.23 s | 2.36 s | n/a (0) |
| lidar_and_rsu | hidden_pedestrian | 70.5 | 5 of 5 | 80.0 % | 20.0 % | 2 | 1.1 s | 2.28 s | 0.14 s | n/a (0) |
| lidar_and_rsu | oncoming_turner | 75.2 | 5 of 5 | 80.0 % | 20.0 % | 0 | 1.22 s | 2.84 s | 0.37 s | n/a (0) |
| lidar_and_rsu | clear_road | 69.3 | 5 of 5 | 40.0 % | 0.0 % | 3 | 18.57 s | 66.68 s | 0.99 s | n/a (0) |
| lidar_and_rsu | **all** | 71.3 | 44 of 45 | 61.4 % | 9.1 % | 12 | 4.56 s | 7.04 s | 0.04 s | 24.19 s (9) |
| camera_and_rsu | traffic_light | 73.9 | 5 of 5 | 40.0 % | 0.0 % | 1 | 3.61 s | 9.02 s | 1.18 s | n/a (0) |
| camera_and_rsu | left_turn | 36.0 | 5 of 5 | 20.0 % | 40.0 % | 9 | 2.78 s | 7.54 s | 0.0 s | 39.77 s (4) |
| camera_and_rsu | blind_corner | 96.4 | 5 of 5 | 100.0 % | 0.0 % | 0 | 2.44 s | 2.87 s | 1.91 s | n/a (0) |
| camera_and_rsu | cut_in | 70.1 | 5 of 5 | 60.0 % | 20.0 % | 0 | 2.19 s | 3.56 s | 0.64 s | 8.9 s (1) |
| camera_and_rsu | yellow_light | 76.0 | 5 of 5 | 20.0 % | 0.0 % | 0 | 5.42 s | 7.73 s | 3.71 s | 19.0 s (5) |
| camera_and_rsu | red_light_runner | 76.6 | 4 of 5 | 75.0 % | 25.0 % | 1 | 3.25 s | 3.63 s | 2.73 s | 9.2 s (1) |
| camera_and_rsu | hidden_pedestrian | 89.0 | 5 of 5 | 100.0 % | 0.0 % | 2 | 1.62 s | 2.37 s | 0.89 s | n/a (0) |
| camera_and_rsu | oncoming_turner | 90.8 | 5 of 5 | 100.0 % | 0.0 % | 2 | 1.96 s | 2.75 s | 1.45 s | n/a (0) |
| camera_and_rsu | clear_road | 77.4 | 5 of 5 | 60.0 % | 0.0 % | 2 | 5.23 s | 12.44 s | 1.01 s | n/a (0) |
| camera_and_rsu | **all** | 72.6 | 44 of 45 | 63.6 % | 9.1 % | 17 | 3.17 s | 8.55 s | 0.58 s | 24.75 s (11) |
| lidar_only | traffic_light | 93.8 | 5 of 5 | 100.0 % | 0.0 % | 0 | 4.8 s | 14.66 s | 1.15 s | n/a (0) |
| lidar_only | left_turn | 57.0 | 5 of 5 | 20.0 % | 20.0 % | 0 | 1.8 s | 3.25 s | 0.3 s | 27.7 s (3) |
| lidar_only | blind_corner | 42.0 | 5 of 5 | 40.0 % | 60.0 % | 0 | 1.22 s | 3.88 s | 0.0 s | 11.08 s (4) |
| lidar_only | cut_in | 62.3 | 5 of 5 | 60.0 % | 40.0 % | 0 | 2.2 s | 3.2 s | 1.28 s | 9.88 s (5) |
| lidar_only | yellow_light | 70.0 | 5 of 5 | 0.0 % | 0.0 % | 0 | 4.59 s | 5.24 s | 3.76 s | 17.3 s (5) |
| lidar_only | red_light_runner | 42.0 | 5 of 5 | 40.0 % | 60.0 % | 0 | 1.28 s | 2.39 s | 0.0 s | 9.3 s (4) |
| lidar_only | hidden_pedestrian | 10.0 | 5 of 5 | 0.0 % | 100.0 % | 0 | 0.0 s | 0.0 s | 0.0 s | n/a (0) |
| lidar_only | oncoming_turner | 10.2 | 5 of 5 | 0.0 % | 100.0 % | 0 | 1.27 s | 4.69 s | 0.06 s | n/a (0) |
| lidar_only | clear_road | 93.7 | 5 of 5 | 100.0 % | 0.0 % | 0 | 4.97 s | 12.26 s | 1.11 s | n/a (0) |
| lidar_only | **all** | 50.9 | 45 of 45 | 40.0 % | 42.2 % | 0 | 2.46 s | 6.23 s | 0.0 s | 14.31 s (21) |

## How each figure is measured

- **Pass rate**: the suite's tests by their own checks; the hazards with no collision; clear_road with no collision, no safety brake and no stop. Of the runs that ran.
- **Collision rate**: runs with at least one collision. `background_collisions` in the runs file counts those with background traffic.
- **False alarms**: safety-brake engagements after which CARLA's true TTC would have stayed at or above 3 s for 3 s had the car carried on at its speed and heading without braking (and was at or above 3 s for the 1 s before), with no collision then; in clear_road every brake. Straight-line TTC can't see a car turning across, so a brake for the oncoming turner can still count, which makes this an upper bound.
- **TTC**: each run's lowest true straight-line TTC to any car or person, as the gate measures it but from CARLA's true positions. Mean, P95 and P5 are over the runs where anything was ever on a collision course. P5 is the dangerous tail.
- **Clearance**: seconds from first entering junction 189 to first leaving it, over the runs that got through (the count in brackets). The hidden hazards and clear_road end about 10 m past their conflict point, still inside the junction, so only the suite's tests give one.
- lidar_only has no roadside unit, so no safety gate: its false alarms are always 0.
- camera_and_rsu is SIMULATED: CARLA's true positions of the actors the roadside mounts have a clear line to, with assumed camera errors (scripts/rsu_camera_sim.py). It can't mistake the containers for parked cars, so it's optimistic.
- red_light_runner is blind_corner's speeding car with its timing varied between repeats.
