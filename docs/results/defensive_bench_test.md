# Bench: test seeds

Each run is a different situation drawn from its seed. Collision rates come with a 95 % Wilson interval. "Oracle crashed" counts seeds where even the teaching rule, driving with true positions, hit something; "avoidable" leaves those out. Skipped ticks are ones the driver never saw, so 0 is what a run should have; "monitor acted" counts the times the car's own safety monitor held or slowed it.

| Config | Scenario | Ran | Collisions | Collision rate (95 % range) | Passed | Oracle crashed | Avoidable collisions (95 % range) | Skipped ticks | Monitor acted |
| ------ | -------- | --- | ---------- | --------------------------- | ------ | -------------- | --------------------------------- | ------------- | ------------- |
| oracle | left_turn | 50 of 50 | 3 | 6.0 % (2.1-16.2 %) | 42 | n/a | n/a | 0 | 0 |
| oracle | red_light_runner | 50 of 50 | 1 | 2.0 % (0.4-10.5 %) | 49 | n/a | n/a | 0 | 0 |
| oracle | hidden_pedestrian | 50 of 50 | 1 | 2.0 % (0.4-10.5 %) | 49 | n/a | n/a | 0 | 0 |
| lidar_and_rsu+planner+monitor | left_turn | 50 of 50 | 0 | 0.0 % (0.0-7.1 %) | 34 | 3 | 0 of 47 (0.0-7.6 %) | 4 | 2 |
| lidar_and_rsu+planner+monitor | red_light_runner | 50 of 50 | 3 | 6.0 % (2.1-16.2 %) | 47 | 1 | 3 of 49 (2.1-16.5 %) | 0 | 5 |
| lidar_and_rsu+planner+monitor | hidden_pedestrian | 50 of 50 | 1 | 2.0 % (0.4-10.5 %) | 49 | 1 | 1 of 49 (0.4-10.7 %) | 0 | 1 |
| camera_and_rsu+planner+monitor | left_turn | 50 of 50 | 1 | 2.0 % (0.4-10.5 %) | 36 | 3 | 1 of 47 (0.4-11.1 %) | 4 | 0 |
| camera_and_rsu+planner+monitor | red_light_runner | 47 of 50 | 1 | 2.1 % (0.4-11.1 %) | 46 | 1 | 1 of 46 (0.4-11.3 %) | 0 | 3 |
| camera_and_rsu+planner+monitor | hidden_pedestrian | 50 of 50 | 0 | 0.0 % (0.0-7.1 %) | 50 | 1 | 0 of 49 (0.0-7.3 %) | 0 | 2 |
| lidar_only+planner+monitor | left_turn | 50 of 50 | 1 | 2.0 % (0.4-10.5 %) | 43 | 3 | 1 of 47 (0.4-11.1 %) | 21 | 1 |
| lidar_only+planner+monitor | red_light_runner | 47 of 50 | 0 | 0.0 % (0.0-7.6 %) | 47 | 1 | 0 of 46 (0.0-7.7 %) | 19 | 2 |
| lidar_only+planner+monitor | hidden_pedestrian | 49 of 50 | 3 | 6.1 % (2.1-16.5 %) | 46 | 1 | 3 of 48 (2.1-16.8 %) | 25 | 3 |
