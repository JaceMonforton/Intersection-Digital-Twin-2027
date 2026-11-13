"""
The bench: each scenario run on many seeds, each seed a different situation (scenario_variation.py), so that a result
means something. Two runs can't tell a safe car from a lucky one; 300 runs with no collision put the collision rate
under 1 % with 95 % confidence.

Every config (study_names.CONFIGS) drives with the planner choosing the speed and the car's own safety monitor after
it. oracle is the teaching rule of training/record_hazards.py driving with CARLA's true positions of everything. When
even it collides on a seed, the report counts that seed as one nobody here avoided and gives the rates both with and
without those seeds, rather than counting it against the car. It's a strong driver, not a perfect one, so "the oracle
crashed" is evidence that a seed is very hard, not proof that it's impossible.

    python scripts/run_bench.py --configs oracle lidar_and_rsu camera_and_rsu lidar_only --runs 50 --speed 0 --quiet
    python scripts/run_bench.py --report                                     # rebuild the report, running nothing
    python scripts/run_bench.py --configs lidar_and_rsu --split test --final --runs 50   # the final measurement, once

Needs CARLA and the Docker stack running with the entities provisioned, and nothing else using CARLA: it starts and
stops the processor itself, once per config. Runs are saved after each one to output/bench/<config>_<split>.jsonl,
so a stopped bench carries on where it left off; the report goes to output/bench/report_<split>.md and .csv.
run_simulation.py runs it with --view, one seed of each scenario, into output/run.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from scenario_variation import SCENARIOS, SPLITS, draw, seeds
from study_names import CONFIGS, VIEWS

BENCH_DIR = ROOT / "output" / "bench"
BENCH_CONFIGS = ("oracle", *CONFIGS)
Z95 = 1.96


def wilson(hits, runs, z=Z95):
    """(low, high) of the 95 % Wilson interval for a rate of hits in runs, as fractions; (0, 1) with no runs."""
    if runs == 0:
        return 0.0, 1.0
    p = hits / runs
    middle = (p + z * z / (2 * runs)) / (1 + z * z / runs)
    half = z * math.sqrt(p * (1 - p) / runs + z * z / (4 * runs * runs)) / (1 + z * z / runs)
    return max(0.0, middle - half), min(1.0, middle + half)


def runs_path(config, split, directory=None):
    return (directory or BENCH_DIR) / f"{config}_{split}.jsonl"


def load_runs(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarise(rows, oracle_crashed=frozenset()):
    """One report row's figures, for rows of one config and scenario. oracle_crashed holds the seeds the oracle hit
    something on, which are left out of the avoidable figures."""
    ran = [r for r in rows if r["ran"]]
    crashed = sum(r["collisions"] > 0 for r in ran)
    avoidable = [r for r in ran if r["seed"] not in oracle_crashed]
    avoidable_crashed = sum(r["collisions"] > 0 for r in avoidable)
    low, high = wilson(crashed, len(ran))
    a_low, a_high = wilson(avoidable_crashed, len(avoidable))
    return {"runs": len(rows), "ran": len(ran), "collisions": crashed, "collision_pct": 100.0 * crashed / len(ran) if ran else None,
            "collision_ci_pct": (round(100 * low, 1), round(100 * high, 1)),
            "passed": sum(r["passed"] for r in ran), "oracle_crashed": len(ran) - len(avoidable),
            "avoidable_runs": len(avoidable), "avoidable_collisions": avoidable_crashed,
            "avoidable_ci_pct": (round(100 * a_low, 1), round(100 * a_high, 1)),
            "skipped_ticks": sum(r.get("skipped_ticks") or 0 for r in ran),
            "monitor_acted": sum(rule["times"] for r in ran for rule in (r.get("monitor") or {}).values())}


def write_report(split, directory=None):
    directory = directory or BENCH_DIR
    every = {config: load_runs(runs_path(config, split, directory)) for config in BENCH_CONFIGS}
    oracle_crashed = defaultdict(set)
    for r in every["oracle"]:
        if r["ran"] and r["collisions"]:
            oracle_crashed[r["scenario"]].add(r["seed"])
    table, lines = [], [f"# Bench: {split} seeds", "",
                        ("Each run is a different situation drawn from its seed. Collision rates come with a 95 % Wilson interval. "
                         "\"Oracle crashed\" counts seeds where even the teaching rule, driving with true positions, hit something; "
                         "\"avoidable\" leaves those out. Skipped ticks are ones the driver never saw, so 0 is what a run "
                         "should have; \"monitor acted\" counts the times the car's own safety monitor held or slowed it."), "",
                        "| Config | Scenario | Ran | Collisions | Collision rate (95 % range) | Passed | Oracle crashed | Avoidable collisions (95 % range) | Skipped ticks | Monitor acted |",
                        "| ------ | -------- | --- | ---------- | --------------------------- | ------ | -------------- | --------------------------------- | ------------- | ------------- |"]
    for config, rows in every.items():
        for scenario in SCENARIOS:
            mine = [r for r in rows if r["scenario"] == scenario]
            if not mine:
                continue
            s = summarise(mine, frozenset() if config == "oracle" else frozenset(oracle_crashed[scenario]))
            table.append({"config": config, "scenario": scenario, **s})
            rate = "n/a" if s["collision_pct"] is None else f"{s['collision_pct']:.1f} % ({s['collision_ci_pct'][0]}-{s['collision_ci_pct'][1]} %)"
            avoid = "n/a" if config == "oracle" or not every["oracle"] else (
                f"{s['avoidable_collisions']} of {s['avoidable_runs']} ({s['avoidable_ci_pct'][0]}-{s['avoidable_ci_pct'][1]} %)")
            lines.append(f"| {config} | {scenario} | {s['ran']} of {s['runs']} | {s['collisions']} | {rate} | {s['passed']} | "
                         f"{s['oracle_crashed'] if config != 'oracle' and every['oracle'] else 'n/a'} | {avoid} | {s['skipped_ticks']} | {s['monitor_acted']} |")
    lines.append("")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"report_{split}.md").write_text("\n".join(lines), encoding="utf-8")
    if table:
        with open(directory / f"report_{split}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    return lines


def run_scenario(suite, name, lead):
    """One run of a scenario, as a flat row of what was measured. Each passes with no collision; the red-light runner
    is judged on the car alone, as the gate doesn't hold it in the simulation."""
    stopped = False
    try:
        if name == "left_turn":
            result, ran = suite.test_left_turn(), True
            passed = result.passed
        else:
            result, _, stopped, ran = suite.run_runner(lead) if name == "red_light_runner" else suite.run_pedestrian(lead)
            passed = ran and not result.collisions
    except RuntimeError as error:
        suite.console.line("ALERT", f"{name} couldn't run: {error}")
        return {"scenario": name, "ran": False, "passed": False, "error": str(error)}
    r = asdict(result)
    if suite.perception == "onboard":  # no roadside unit, so its processor's ground-truth gate isn't an intervention
        r["brake_events"] = r["false_brakes"] = 0
    return {"scenario": name, "lead": None if name == "left_turn" else lead, "ran": bool(ran), "passed": bool(passed and ran),
            "collisions": len(r["collisions"]), "background_collisions": sum(c.get("role") == "background" for c in r["collisions"]),
            "brake_events": r["brake_events"], "false_brakes": r["false_brakes"], "true_min_ttc": r["true_min_ttc"],
            "clearance_s": r["clearance_s"], "traffic_spawned": r["traffic"], "duration_s": r["duration"], "skipped_ticks": r["skipped_ticks"],
            "monitor": r["monitor"], "notes": "; ".join(r["notes"]),
            "failed_checks": ", ".join(check for check, ok in r["checks"].items() if not ok), "stopped": bool(stopped)}


def spawn_window(script_args, log_path):
    """A display beside the run. It only watches, so it may fail without stopping anything."""
    log = open(log_path, "w", encoding="utf-8", errors="replace")  # noqa: SIM115
    process = subprocess.Popen([sys.executable, "-u", *script_args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
    return process, log


def run_config(config, args):
    """Every seed for one config, with its own processor, saving each run as it finishes."""
    import carla
    from hidden_hazards import HiddenHazards, start_processor, stop_processor
    from ml_driver import DrivingPolicy as Policy
    from scenario_suite import (
        CARLA_HOST,
        CARLA_PORT,
        JUNCTION,
        OUTPUT_DIR,
        Console,
    )

    mode = "truth" if config == "oracle" else CONFIGS[config]
    path = runs_path(config, args.split + ("_droplink" if args.drop_link else ""), args.out)  # kept out of the report, which is normal runs
    done = {(r["scenario"], r["seed"]) for r in load_runs(path)}
    todo = [(name, seed) for name in args.scenarios for seed in seeds(args.split, args.runs, args.start) if (name, seed) not in done]
    if not todo:
        print(f"[BENCH] {config}: every run is already done.", flush=True)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    process, log = start_processor(mode, args.model, OUTPUT_DIR / f"bench_processor_{config}.log", view=args.view,
                                   extra=["--no-draw", "--lockstep", "--speed", str(args.speed)])
    windows = []
    if args.view:
        windows.append(spawn_window(["scripts/ml_car_view.py"], OUTPUT_DIR / f"bench_{config}_car_view.log"))
        if mode in VIEWS:
            windows.append(spawn_window(VIEWS[mode], OUTPUT_DIR / f"bench_{config}_sensor_view.log"))
    try:
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(30.0)
        if config == "oracle":
            from record_hazards import HazardTeacher
            policy = HazardTeacher(Policy())
        else:
            policy = Policy()
        suite = HiddenHazards(carla, client, policy, Console(args.quiet), seed=1, perception=mode)
        suite.link_drop = args.drop_link
        if config == "oracle":
            policy.world = suite.world
        else:
            from planner import Planner
            from roadside_link import ROADSIDE_MODES

            suite.planner, suite.monitor = Planner(JUNCTION, has_roadside=mode in ROADSIDE_MODES), True
        try:
            for index, (name, seed) in enumerate(todo, 1):
                variation = draw(name, seed, args.traffic)
                suite.variation, suite.seed, suite.traffic = variation, seed, variation.traffic
                if config == "oracle":
                    policy.reset()
                row = run_scenario(suite, name, variation.lead)
                if config == "oracle":
                    policy.inputs.clear()  # its lessons aren't kept here; record_hazards.py records those
                row = {"config": config, "split": args.split, "seed": seed, "link_drop": args.drop_link, **row, **variation.as_row()}
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                suite.console.end_status()
                verdict = "did not run" if not row["ran"] else "COLLISION" if row.get("collisions") else "PASS" if row["passed"] else "FAIL, no collision"
                print(f"[BENCH] {config} {name} seed {seed} ({index}/{len(todo)}): {verdict}", flush=True)
        finally:
            suite.close()
    finally:
        for window in windows:
            stop_processor(*window)
        stop_processor(process, log)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", choices=BENCH_CONFIGS, default=list(CONFIGS))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--split", choices=SPLITS, default="tune")
    parser.add_argument("--final", action="store_true", help="needed to run the test split, which is only for the final measurement")
    parser.add_argument("--runs", type=int, default=50, help="seeds of each scenario")
    parser.add_argument("--start", type=int, default=0, help="skip this many seeds of the split first")
    parser.add_argument("--traffic", type=int, help="background cars in every run, instead of each seed's own")
    parser.add_argument("--speed", type=float, default=1.0, help="the processor's --speed: 0 runs as fast as CARLA can")
    parser.add_argument("--model", default=str(ROOT / "models" / "rsu_detector.candidate.onnx"))
    parser.add_argument("--drop-link", type=lambda text: tuple(float(v) for v in text.split(",")), metavar="FROM,TO",
                        help="cut the roadside link FROM to TO s into every run, to test what the car does without it; "
                             "saved apart, as <config>_<split>_droplink.jsonl")
    parser.add_argument("--out", type=Path, default=BENCH_DIR, help="where the runs and the report go")
    parser.add_argument("--view", action="store_true", help="open the ML car's window and the config's sensor window")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--report", action="store_true", help="rebuild the report from the saved runs, running nothing")
    args = parser.parse_args()
    if args.split == "test" and not args.final and not args.report:
        parser.error("the test split is for the final measurement only: add --final if that's what this is")
    if args.runs < 1 or args.speed < 0:
        parser.error("--runs must be at least 1 and --speed 0 or more")

    if not args.report:
        if len(args.configs) > 1:
            # Each config gets its own process, as a Traffic Manager doesn't survive being reused after its processor changes
            common = ["--split", args.split, "--runs", str(args.runs), "--start", str(args.start), "--speed", str(args.speed),
                      "--model", args.model, "--scenarios", *args.scenarios, "--out", str(args.out), *(["--final"] if args.final else []),
                      *(["--traffic", str(args.traffic)] if args.traffic is not None else []), *(["--quiet"] if args.quiet else []),
                      *(["--view"] if args.view else []),
                      *(["--drop-link", ",".join(str(v) for v in args.drop_link)] if args.drop_link else [])]
            child = {**os.environ, "BENCH_CHILD": "1"}  # only this process prints the report
            failed = [config for config in args.configs
                      if subprocess.call([sys.executable, "-u", __file__, *common, "--configs", config], cwd=ROOT, env=child) != 0]
        else:
            run_config(args.configs[0], args)
            failed = []
    report = write_report(args.split + ("_droplink" if args.drop_link else ""), args.out)
    if not os.environ.get("BENCH_CHILD"):
        print("\n".join(report))
    if not args.report and failed:
        print(f"[BENCH] These configs stopped with an error, so their runs may be missing: {', '.join(failed)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
