# Bench: tune seeds

Each run is a different situation drawn from its seed. Collision rates come with a 95 % Wilson interval. "Oracle crashed" counts seeds where even the teaching rule, driving with true positions, hit something; "avoidable" leaves those out.

| Config | Scenario | Ran | Collisions | Collision rate (95 % range) | Passed | Oracle crashed | Avoidable collisions (95 % range) |
| ------ | -------- | --- | ---------- | --------------------------- | ------ | -------------- | --------------------------------- |
| oracle | left_turn | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 6 | n/a | n/a |
| oracle | red_light_runner | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | n/a | n/a |
| oracle | hidden_pedestrian | 10 of 10 | 1 | 10.0 % (1.8-40.4 %) | 9 | n/a | n/a |
| lidar_and_rsu | left_turn | 10 of 10 | 3 | 30.0 % (10.8-60.3 %) | 3 | 0 | 3 of 10 (10.8-60.3 %) |
| lidar_and_rsu | red_light_runner | 10 of 10 | 5 | 50.0 % (23.7-76.3 %) | 5 | 0 | 5 of 10 (23.7-76.3 %) |
| lidar_and_rsu | hidden_pedestrian | 10 of 10 | 2 | 20.0 % (5.7-51.0 %) | 8 | 1 | 2 of 9 (6.3-54.7 %) |
| camera_and_rsu | left_turn | 10 of 10 | 3 | 30.0 % (10.8-60.3 %) | 5 | 0 | 3 of 10 (10.8-60.3 %) |
| camera_and_rsu | red_light_runner | 10 of 10 | 6 | 60.0 % (31.3-83.2 %) | 4 | 0 | 6 of 10 (31.3-83.2 %) |
| camera_and_rsu | hidden_pedestrian | 10 of 10 | 2 | 20.0 % (5.7-51.0 %) | 8 | 1 | 2 of 9 (6.3-54.7 %) |
| lidar_only | left_turn | 10 of 10 | 5 | 50.0 % (23.7-76.3 %) | 3 | 0 | 5 of 10 (23.7-76.3 %) |
| lidar_only | red_light_runner | 10 of 10 | 6 | 60.0 % (31.3-83.2 %) | 4 | 0 | 6 of 10 (31.3-83.2 %) |
| lidar_only | hidden_pedestrian | 10 of 10 | 3 | 30.0 % (10.8-60.3 %) | 7 | 1 | 2 of 9 (6.3-54.7 %) |
| lidar_and_rsu+planner | left_turn | 10 of 10 | 1 | 10.0 % (1.8-40.4 %) | 4 | 0 | 1 of 10 (1.8-40.4 %) |
| lidar_and_rsu+planner | red_light_runner | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | 0 | 0 of 10 (0.0-27.8 %) |
| lidar_and_rsu+planner | hidden_pedestrian | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | 1 | 0 of 9 (0.0-29.9 %) |
| camera_and_rsu+planner | left_turn | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 6 | 0 | 0 of 10 (0.0-27.8 %) |
| camera_and_rsu+planner | red_light_runner | 10 of 10 | 1 | 10.0 % (1.8-40.4 %) | 9 | 0 | 1 of 10 (1.8-40.4 %) |
| camera_and_rsu+planner | hidden_pedestrian | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | 1 | 0 of 9 (0.0-29.9 %) |
| lidar_only+planner | left_turn | 10 of 10 | 2 | 20.0 % (5.7-51.0 %) | 3 | 0 | 2 of 10 (5.7-51.0 %) |
| lidar_only+planner | red_light_runner | 8 of 10 | 1 | 12.5 % (2.2-47.1 %) | 7 | 0 | 1 of 8 (2.2-47.1 %) |
| lidar_only+planner | hidden_pedestrian | 10 of 10 | 1 | 10.0 % (1.8-40.4 %) | 9 | 1 | 1 of 9 (2.0-43.5 %) |
