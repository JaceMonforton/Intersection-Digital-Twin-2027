# Bench: tune seeds

Each run is a different situation drawn from its seed. Collision rates come with a 95 % Wilson interval. "Oracle crashed" counts seeds where even the teaching rule, driving with true positions, hit something; "avoidable" leaves those out. Skipped ticks are ones the driver never saw, so 0 is what a run should have; "monitor acted" counts the times the car's own safety monitor held or slowed it.

| Config | Scenario | Ran | Collisions | Collision rate (95 % range) | Passed | Oracle crashed | Avoidable collisions (95 % range) | Skipped ticks | Monitor acted |
| ------ | -------- | --- | ---------- | --------------------------- | ------ | -------------- | --------------------------------- | ------------- | ------------- |
| oracle | left_turn | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 9 | n/a | n/a | 0 | 0 |
| oracle | red_light_runner | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | n/a | n/a | 0 | 0 |
| oracle | hidden_pedestrian | 10 of 10 | 1 | 10.0 % (1.8-40.4 %) | 9 | n/a | n/a | 0 | 0 |
| lidar_and_rsu+planner+monitor | left_turn | 10 of 10 | 1 | 10.0 % (1.8-40.4 %) | 8 | 0 | 1 of 10 (1.8-40.4 %) | 0 | 0 |
| lidar_and_rsu+planner+monitor | red_light_runner | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | 0 | 0 of 10 (0.0-27.8 %) | 0 | 0 |
| lidar_and_rsu+planner+monitor | hidden_pedestrian | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | 1 | 0 of 9 (0.0-29.9 %) | 1 | 1 |
| camera_and_rsu+planner+monitor | left_turn | 10 of 10 | 1 | 10.0 % (1.8-40.4 %) | 7 | 0 | 1 of 10 (1.8-40.4 %) | 1 | 0 |
| camera_and_rsu+planner+monitor | red_light_runner | 10 of 10 | 1 | 10.0 % (1.8-40.4 %) | 9 | 0 | 1 of 10 (1.8-40.4 %) | 0 | 1 |
| camera_and_rsu+planner+monitor | hidden_pedestrian | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | 1 | 0 of 9 (0.0-29.9 %) | 1 | 0 |
| lidar_only+planner+monitor | left_turn | 10 of 10 | 1 | 10.0 % (1.8-40.4 %) | 9 | 0 | 1 of 10 (1.8-40.4 %) | 1 | 0 |
| lidar_only+planner+monitor | red_light_runner | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | 0 | 0 of 10 (0.0-27.8 %) | 0 | 1 |
| lidar_only+planner+monitor | hidden_pedestrian | 10 of 10 | 1 | 10.0 % (1.8-40.4 %) | 9 | 1 | 0 of 9 (0.0-29.9 %) | 3 | 1 |
