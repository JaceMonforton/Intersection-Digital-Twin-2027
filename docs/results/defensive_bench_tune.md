# Bench: tune seeds

Each run is a different situation drawn from its seed. Collision rates come with a 95 % Wilson interval. "Oracle crashed" counts seeds where even the teaching rule, driving with true positions, hit something; "avoidable" leaves those out. Skipped ticks are ones the driver never saw, so 0 is what a run should have; "monitor acted" counts the times the car's own safety monitor held or slowed it.

| Config | Scenario | Ran | Collisions | Collision rate (95 % range) | Passed | Oracle crashed | Avoidable collisions (95 % range) | Skipped ticks | Monitor acted |
| ------ | -------- | --- | ---------- | --------------------------- | ------ | -------------- | --------------------------------- | ------------- | ------------- |
| oracle | left_turn | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | n/a | n/a | 1 | 0 |
| oracle | red_light_runner | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | n/a | n/a | 0 | 0 |
| oracle | hidden_pedestrian | 10 of 10 | 1 | 10.0 % (1.8-40.4 %) | 9 | n/a | n/a | 0 | 0 |
| lidar_and_rsu | left_turn | 10 of 10 | 5 | 50.0 % (23.7-76.3 %) | 4 | 0 | 5 of 10 (23.7-76.3 %) | 11 | 0 |
| lidar_and_rsu | red_light_runner | 10 of 10 | 5 | 50.0 % (23.7-76.3 %) | 5 | 0 | 5 of 10 (23.7-76.3 %) | 3 | 0 |
| lidar_and_rsu | hidden_pedestrian | 10 of 10 | 2 | 20.0 % (5.7-51.0 %) | 8 | 1 | 2 of 9 (6.3-54.7 %) | 4 | 0 |
| camera_and_rsu | left_turn | 10 of 10 | 3 | 30.0 % (10.8-60.3 %) | 3 | 0 | 3 of 10 (10.8-60.3 %) | 14 | 0 |
| camera_and_rsu | red_light_runner | 10 of 10 | 7 | 70.0 % (39.7-89.2 %) | 3 | 0 | 7 of 10 (39.7-89.2 %) | 42 | 0 |
| camera_and_rsu | hidden_pedestrian | 10 of 10 | 4 | 40.0 % (16.8-68.7 %) | 6 | 1 | 4 of 9 (18.9-73.3 %) | 3 | 0 |
| lidar_only | left_turn | 10 of 10 | 4 | 40.0 % (16.8-68.7 %) | 4 | 0 | 4 of 10 (16.8-68.7 %) | 32 | 0 |
| lidar_only | red_light_runner | 10 of 10 | 6 | 60.0 % (31.3-83.2 %) | 4 | 0 | 6 of 10 (31.3-83.2 %) | 61 | 0 |
| lidar_only | hidden_pedestrian | 10 of 10 | 2 | 20.0 % (5.7-51.0 %) | 8 | 1 | 2 of 9 (6.3-54.7 %) | 47 | 0 |
| lidar_and_rsu+planner | left_turn | 10 of 10 | 1 | 10.0 % (1.8-40.4 %) | 6 | 0 | 1 of 10 (1.8-40.4 %) | 14 | 0 |
| lidar_and_rsu+planner | red_light_runner | 10 of 10 | 1 | 10.0 % (1.8-40.4 %) | 9 | 0 | 1 of 10 (1.8-40.4 %) | 6 | 0 |
| lidar_and_rsu+planner | hidden_pedestrian | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | 1 | 0 of 9 (0.0-29.9 %) | 7 | 0 |
| camera_and_rsu+planner | left_turn | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 7 | 0 | 0 of 10 (0.0-27.8 %) | 13 | 0 |
| camera_and_rsu+planner | red_light_runner | 10 of 10 | 2 | 20.0 % (5.7-51.0 %) | 8 | 0 | 2 of 10 (5.7-51.0 %) | 8 | 0 |
| camera_and_rsu+planner | hidden_pedestrian | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | 1 | 0 of 9 (0.0-29.9 %) | 12 | 0 |
| lidar_only+planner | left_turn | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 8 | 0 | 0 of 10 (0.0-27.8 %) | 59 | 0 |
| lidar_only+planner | red_light_runner | 10 of 10 | 2 | 20.0 % (5.7-51.0 %) | 8 | 0 | 2 of 10 (5.7-51.0 %) | 45 | 0 |
| lidar_only+planner | hidden_pedestrian | 9 of 10 | 0 | 0.0 % (0.0-29.9 %) | 9 | 0 | 0 of 9 (0.0-29.9 %) | 30 | 0 |
| lidar_and_rsu+planner+monitor | left_turn | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 8 | 0 | 0 of 10 (0.0-27.8 %) | 13 | 0 |
| lidar_and_rsu+planner+monitor | red_light_runner | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | 0 | 0 of 10 (0.0-27.8 %) | 4 | 0 |
| lidar_and_rsu+planner+monitor | hidden_pedestrian | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | 1 | 0 of 9 (0.0-29.9 %) | 6 | 0 |
| camera_and_rsu+planner+monitor | left_turn | 10 of 10 | 1 | 10.0 % (1.8-40.4 %) | 7 | 0 | 1 of 10 (1.8-40.4 %) | 28 | 0 |
| camera_and_rsu+planner+monitor | red_light_runner | 10 of 10 | 1 | 10.0 % (1.8-40.4 %) | 9 | 0 | 1 of 10 (1.8-40.4 %) | 17 | 0 |
| camera_and_rsu+planner+monitor | hidden_pedestrian | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | 1 | 0 of 9 (0.0-29.9 %) | 18 | 0 |
| lidar_only+planner+monitor | left_turn | 10 of 10 | 1 | 10.0 % (1.8-40.4 %) | 5 | 0 | 1 of 10 (1.8-40.4 %) | 34 | 0 |
| lidar_only+planner+monitor | red_light_runner | 10 of 10 | 1 | 10.0 % (1.8-40.4 %) | 9 | 0 | 1 of 10 (1.8-40.4 %) | 49 | 0 |
| lidar_only+planner+monitor | hidden_pedestrian | 9 of 10 | 0 | 0.0 % (0.0-29.9 %) | 9 | 0 | 0 of 9 (0.0-29.9 %) | 126 | 0 |
