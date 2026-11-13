# Bench: tune seeds

Each run is a different situation drawn from its seed. Collision rates come with a 95 % Wilson interval. "Oracle crashed" counts seeds where even the teaching rule, driving with true positions, hit something; "avoidable" leaves those out. Skipped ticks are ones the driver never saw, so 0 is what a run should have; "monitor acted" counts the times the car's own safety monitor held or slowed it.

| Config | Scenario | Ran | Collisions | Collision rate (95 % range) | Passed | Oracle crashed | Avoidable collisions (95 % range) | Skipped ticks | Monitor acted |
| ------ | -------- | --- | ---------- | --------------------------- | ------ | -------------- | --------------------------------- | ------------- | ------------- |
| oracle | left_turn | 50 of 50 | 2 | 4.0 % (1.1-13.5 %) | 38 | n/a | n/a | 0 | 0 |
| oracle | red_light_runner | 50 of 50 | 2 | 4.0 % (1.1-13.5 %) | 48 | n/a | n/a | 0 | 0 |
| oracle | hidden_pedestrian | 50 of 50 | 5 | 10.0 % (4.3-21.4 %) | 45 | n/a | n/a | 0 | 0 |
| lidar_and_rsu+planner+monitor | left_turn | 50 of 50 | 2 | 4.0 % (1.1-13.5 %) | 40 | 2 | 2 of 48 (1.2-14.0 %) | 1 | 0 |
| lidar_and_rsu+planner+monitor | red_light_runner | 49 of 50 | 0 | 0.0 % (0.0-7.3 %) | 49 | 2 | 0 of 47 (0.0-7.6 %) | 0 | 1 |
| lidar_and_rsu+planner+monitor | hidden_pedestrian | 50 of 50 | 3 | 6.0 % (2.1-16.2 %) | 47 | 5 | 1 of 45 (0.4-11.6 %) | 1 | 4 |
| camera_and_rsu+planner+monitor | left_turn | 50 of 50 | 1 | 2.0 % (0.4-10.5 %) | 41 | 2 | 1 of 48 (0.4-10.9 %) | 2 | 0 |
| camera_and_rsu+planner+monitor | red_light_runner | 49 of 50 | 2 | 4.1 % (1.1-13.7 %) | 47 | 2 | 2 of 47 (1.2-14.2 %) | 0 | 3 |
| camera_and_rsu+planner+monitor | hidden_pedestrian | 49 of 50 | 1 | 2.0 % (0.4-10.7 %) | 48 | 4 | 1 of 45 (0.4-11.6 %) | 1 | 5 |
| lidar_only+planner+monitor | left_turn | 50 of 50 | 1 | 2.0 % (0.4-10.5 %) | 36 | 2 | 1 of 48 (0.4-10.9 %) | 1 | 1 |
| lidar_only+planner+monitor | red_light_runner | 48 of 50 | 1 | 2.1 % (0.4-10.9 %) | 47 | 2 | 1 of 46 (0.4-11.3 %) | 0 | 1 |
| lidar_only+planner+monitor | hidden_pedestrian | 50 of 50 | 3 | 6.0 % (2.1-16.2 %) | 47 | 5 | 0 of 45 (0.0-7.9 %) | 4 | 2 |
