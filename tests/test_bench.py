"""The scenario variation and the many-seed bench's figures. No CARLA needed."""
import pytest
from run_bench import summarise, wilson, write_report
from scenario_variation import SCENARIOS, SPLITS, draw, seeds, split_of


def test_the_splits_never_share_a_seed_and_each_seed_has_one_split():
    ranges = [set(r) for r in SPLITS.values()]
    assert all(not (a & b) for i, a in enumerate(ranges) for b in ranges[i + 1:])
    assert split_of(1000) == "train" and split_of(9000) == "test" and split_of(7) is None
    assert seeds("tune", 3) == [5000, 5001, 5002]
    with pytest.raises(ValueError):
        seeds("tune", len(SPLITS["tune"]) + 1)


def test_a_seed_always_gives_the_same_situation_and_seeds_differ():
    for scenario in SCENARIOS:
        assert draw(scenario, 5000) == draw(scenario, 5000)
        assert len({draw(scenario, s) for s in range(5000, 5020)}) == 20
    assert draw("left_turn", 5000, traffic=0).traffic == 0


def test_the_variation_covers_easy_and_hard_situations():
    turners = [draw("oncoming_turner", s) for s in range(5000, 5300)]
    leads = [v.lead for v in turners]
    assert min(leads) < -1.0 and max(leads) > 1.0 and any(abs(x) < 0.2 for x in leads)
    assert 0.6 < sum(v.turner_turns for v in turners) / len(turners) < 0.95  # some carry straight on
    assert any(not v.hidden for v in turners) and any(v.hidden for v in turners)
    assert draw("hidden_pedestrian", 5000).as_row()["var_walk_speed"] == draw("hidden_pedestrian", 5000).walk_speed


def test_wilson_bounds_a_clean_record_away_from_zero_and_tighten_with_runs():
    low, high = wilson(0, 300)
    assert low == 0.0 and high == pytest.approx(0.0126, abs=1e-3)  # 0 in 300: under about 1.3 %
    assert wilson(0, 2)[1] > 0.6  # two clean runs say almost nothing
    assert wilson(5, 10)[0] < 0.5 < wilson(5, 10)[1]
    assert wilson(0, 0) == (0.0, 1.0)


def run(seed, collisions=0, ran=True, scenario="hidden_pedestrian", **extra):
    return {"scenario": scenario, "seed": seed, "ran": ran, "collisions": collisions, "passed": ran and not collisions, **extra}


def test_seeds_the_oracle_crashed_on_are_left_out_of_the_avoidable_figures():
    rows = [run(1, 1), run(2, 1), run(3), run(4, ran=False)]
    s = summarise(rows, oracle_crashed=frozenset({1}))
    assert (s["runs"], s["ran"], s["collisions"], s["oracle_crashed"]) == (4, 3, 2, 1)
    assert (s["avoidable_runs"], s["avoidable_collisions"]) == (2, 1)


def test_the_report_lines_up_the_configs_against_the_oracle(tmp_path, monkeypatch):
    import json

    import run_bench

    monkeypatch.setattr(run_bench, "BENCH_DIR", tmp_path)
    for config, rows in {"oracle": [run(1, 1), run(2)], "lidar_only": [run(1, 1), run(2, 1)]}.items():
        (tmp_path / f"{config}_tune.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    text = "\n".join(write_report("tune", tmp_path))
    assert "| lidar_only | hidden_pedestrian | 2 of 2 | 2 | 100.0 %" in text
    assert "| 1 | 1 of 1 (" in text  # the oracle crashed on one seed, so one of lidar_only's two collisions was avoidable
    assert (tmp_path / "report_tune.csv").exists()
