import argparse
import csv
import math
import struct
from types import SimpleNamespace

import numpy as np
import pytest
from rsu_camera_view import aim_at_centre, box_corners_world, project
from run_professor_test_suite import false_brakes, roof_sweep_to_model_frame, true_ttc
from run_study import (
    FINAL_SEED,
    ranking,
    report_rows,
    score,
    study_seeds,
    summarise,
    write_report,
)
from study_names import (
    CONFIGS,
    SCENARIOS,
    add_config_flags,
    processor_args,
    scenarios_from,
)


def run(scenario="hidden_pedestrian", config="lidar_and_rsu", traffic=0, **values):
    row = {"config": config, "traffic": traffic, "scenario": scenario, "ran": True, "passed": True, "collisions": 0,
           "false_brakes": 0, "true_min_ttc": None, "clearance_s": None}
    row.update(values)
    return row


def test_true_ttc_sees_a_car_ahead_and_ignores_one_coming_up_from_behind():
    ego = (0.0, 0.0, 10.0, 0.0, 0.0)
    assert true_ttc(ego, [(20.0, 0.0, 0.0, 0.0, 0.0)], []) == pytest.approx((20.0 - 2.5) / 10.0)
    assert math.isinf(true_ttc(ego, [(-20.0, 0.0, 15.0, 0.0, 0.0)], []))
    assert true_ttc(ego, [], [(10.0, 0.0, 0.0, 0.0)]) == pytest.approx(0.75)  # people have no heading, so always count
    assert math.isinf(true_ttc(ego, [], []))


def test_a_brake_is_a_false_alarm_only_with_no_danger_or_collision_around_it():
    safe = [(t / 10, 8.0) for t in range(100)]
    assert false_brakes([5.0], safe, []) == 1
    danger = [(t, 1.0 if t == 7.0 else ttc) for t, ttc in safe]
    assert false_brakes([5.0], danger, []) == 0     # true TTC fell under 3 s within 3 s after it
    assert false_brakes([5.0], safe, [4.5]) == 0    # a collision just before it
    assert false_brakes([5.0], [(t, 1.0 if t == 8.5 else ttc) for t, ttc in safe], []) == 1  # danger only after the window


def test_the_study_never_uses_the_final_test_seed():
    seeds = study_seeds(40)
    assert FINAL_SEED not in seeds and len(seeds) == 40 and seeds[:3] == [1, 2, 3]


def test_summarise_counts_only_runs_that_ran_and_reports_the_low_ttc_tail():
    rows = [run(true_min_ttc=t, clearance_s=c, passed=p, collisions=k, false_brakes=f)
            for t, c, p, k, f in ((0.5, 4.0, False, 1, 0), (2.0, 5.0, True, 0, 1), (4.0, None, True, 0, 0), (None, 6.0, True, 0, 0))]
    rows.append(run(ran=False, passed=False))
    s = summarise(rows)
    assert (s["runs"], s["ran"], s["pass_rate_pct"], s["collision_rate_pct"], s["false_interventions"]) == (5, 4, 75.0, 25.0, 1)
    assert s["ttc_runs"] == 3 and s["ttc_mean_s"] == pytest.approx(2.17, abs=0.01)
    assert s["ttc_p5_s"] < s["ttc_mean_s"] < s["ttc_p95_s"]
    assert (s["clearance_mean_s"], s["cleared"]) == (5.0, 3)
    assert summarise([run(ran=False)])["pass_rate_pct"] is None


def test_the_report_has_a_row_per_scenario_then_all_and_writes_its_files(tmp_path):
    runs = [run("cut_in", "lidar_only"), run("hidden_pedestrian", "lidar_only", passed=False, collisions=1), run("cut_in", "lidar_and_rsu", traffic=15)]
    rows = report_rows(runs)
    assert [(r["config"], r["traffic"], r["scenario"]) for r in rows] == [
        ("lidar_only", 0, "cut_in"), ("lidar_only", 0, "hidden_pedestrian"), ("lidar_only", 0, "all"),
        ("lidar_and_rsu", 15, "cut_in"), ("lidar_and_rsu", 15, "all")]
    write_report(runs, tmp_path)
    with open(tmp_path / "comparative_study_report.csv", encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == len(rows)
    summary = (tmp_path / "comparative_study_summary.md").read_text(encoding="utf-8")
    assert "## 0 background cars" in summary and "## 15 background cars" in summary and "SIMULATED" in summary


def test_the_score_rewards_safety_most_and_the_ranking_puts_the_best_first():
    perfect = summarise([run(), run()])
    assert score(perfect) == 100.0
    crashes = summarise([run(collisions=1, passed=False, true_min_ttc=0.0), run()])
    alarms = summarise([run(false_brakes=1), run(false_brakes=1)])
    assert score(crashes) == pytest.approx(50 * 0.5 + 30 * 0.5 + 10 + 0)
    assert score(alarms) == 90.0 and score(crashes) < score(alarms) < score(perfect)
    assert score(summarise([run(ran=False)])) is None
    runs = [run(config="lidar_only", collisions=1, passed=False, true_min_ttc=0.0), run(config="lidar_and_rsu"), run(config="camera_and_rsu", false_brakes=1)]
    (load, zero), (every, overall) = ranking(runs)
    assert load == 0 and every == "all"
    assert [c for c, _ in zero] == ["lidar_and_rsu", "camera_and_rsu", "lidar_only"] == [c for c, _ in overall]


def test_the_summary_opens_with_the_ranking_and_the_old_rows_lose_passes_that_didnt_run(tmp_path):
    rows = write_report([run(config="lidar_only"), run(config="lidar_and_rsu", ran=False, passed=True)], tmp_path)
    summary = (tmp_path / "comparative_study_summary.md").read_text(encoding="utf-8")
    assert summary.index("## Which config is best") < summary.index("by scenario")
    assert "Best: **lidar_only** (100.0)." in summary
    assert all(r["config"] != "lidar_and_rsu" or r["score"] is None for r in rows)


def test_scenario_names_expand_and_keep_the_study_order():
    assert scenarios_from(["all"]) == list(SCENARIOS)
    assert scenarios_from(["clear_road", "traffic_light", "clear_road"]) == ["traffic_light", "clear_road"]


def test_each_config_gets_its_own_detector_and_only_one_config_flag_at_a_time():
    assert processor_args("rsu", "m.onnx") == ["--detector", "onnx", "--model", "m.onnx", "--lidar-view"]
    assert processor_args("camera", "m.onnx", view=False) == ["--detector", "camera"]
    assert processor_args("sight", "m.onnx") == ["--detector", "groundtruth"]
    parser = argparse.ArgumentParser()
    add_config_flags(parser)
    parser.set_defaults(perception="truth")
    assert parser.parse_args([]).perception == "truth"
    assert {flag: parser.parse_args([f"--{flag}"]).perception for flag in CONFIGS} == CONFIGS
    with pytest.raises(SystemExit):
        parser.parse_args(["--lidar_only", "--camera_and_rsu"])


def test_a_roof_sweep_lands_in_the_junction_frame_turned_by_the_cars_heading():
    raw = struct.pack("<ffffII", 10.0, 0.0, -1.9, 1.0, 7, 10)  # 10 m ahead of the sensor, on the road
    center = SimpleNamespace(x=100.0, y=50.0, z=0.0)
    points = roof_sweep_to_model_frame(raw, SimpleNamespace(x=90.0, y=50.0, z=1.9), 90.0, center)
    # Facing +y (CARLA's right-hand side, the model's -y), the point is at world (90, 60): 10 m before the centre, 10 m to its model right
    assert points[0, :2] == pytest.approx((-10.0, -10.0), abs=1e-4)


def test_a_camera_projects_what_it_faces_and_nothing_behind_it():
    pitch, yaw = aim_at_centre((10.0, 0.0, 10.0))
    assert pitch == pytest.approx(-45.0) and abs(yaw) == pytest.approx(180.0)  # looking back along -x, at the centre
    facing_x = np.eye(4)  # a camera at the origin looking along +x
    pixels = project(np.array([[10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [-5.0, 0.0, 0.0]]), facing_x, 640, 360, 90.0)
    assert pixels[0] == pytest.approx((320.0, 180.0))
    assert pixels[1] == pytest.approx((640.0, 180.0))  # 45 degrees right is the edge at a 90 degree field of view
    assert np.isnan(pixels[2]).all()


def test_box_corners_come_back_in_carla_frame():
    corners = box_corners_world((1.0, 2.0, 0.75 + 0.0, 4.0, 2.0, 1.5, 0.0), (10.0, 20.0, 0.0))
    assert corners.shape == (8, 3)
    assert corners[:, 0].min() == pytest.approx(9.0) and corners[:, 0].max() == pytest.approx(13.0)
    assert corners[:, 1].mean() == pytest.approx(18.0)  # the model's y is left, CARLA's right
