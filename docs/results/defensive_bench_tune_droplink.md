# Bench: tune_droplink seeds

Each run is a different situation drawn from its seed. Collision rates come with a 95 % Wilson interval. "Oracle crashed" counts seeds where even the teaching rule, driving with true positions, hit something; "avoidable" leaves those out. Skipped ticks are ones the driver never saw, so 0 is what a run should have; "monitor acted" counts the times the car's own safety monitor held or slowed it.

| Config | Scenario | Ran | Collisions | Collision rate (95 % range) | Passed | Oracle crashed | Avoidable collisions (95 % range) | Skipped ticks | Monitor acted |
| ------ | -------- | --- | ---------- | --------------------------- | ------ | -------------- | --------------------------------- | ------------- | ------------- |
| lidar_and_rsu+planner | left_turn | 10 of 10 | 1 | 10.0 % (1.8-40.4 %) | 7 | n/a | n/a | 13 | 0 |
| lidar_and_rsu+planner | red_light_runner | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | n/a | n/a | 5 | 0 |
| lidar_and_rsu+planner | hidden_pedestrian | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | n/a | n/a | 6 | 0 |
| camera_and_rsu+planner | left_turn | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 8 | n/a | n/a | 17 | 0 |
| camera_and_rsu+planner | red_light_runner | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | n/a | n/a | 17 | 0 |
| camera_and_rsu+planner | hidden_pedestrian | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | n/a | n/a | 28 | 0 |
| lidar_and_rsu+planner+monitor | left_turn | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 8 | n/a | n/a | 13 | 61 |
| lidar_and_rsu+planner+monitor | red_light_runner | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | n/a | n/a | 6 | 27 |
| lidar_and_rsu+planner+monitor | hidden_pedestrian | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | n/a | n/a | 8 | 57 |
| camera_and_rsu+planner+monitor | left_turn | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 9 | n/a | n/a | 30 | 58 |
| camera_and_rsu+planner+monitor | red_light_runner | 10 of 10 | 1 | 10.0 % (1.8-40.4 %) | 9 | n/a | n/a | 25 | 20 |
| camera_and_rsu+planner+monitor | hidden_pedestrian | 10 of 10 | 0 | 0.0 % (0.0-27.8 %) | 10 | n/a | n/a | 21 | 49 |
