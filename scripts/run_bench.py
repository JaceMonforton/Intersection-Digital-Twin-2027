"""
The many-seed bench: each scenario run on many seeds, each seed a different situation (scenario_variation.py), so that
a result means something. Two runs can't tell a safe car from a lucky one; 300 runs with no collision put the
collision rate under 1 % with 95 % confidence.

    python scripts/run_bench.py --configs lidar_and_rsu --runs 50            # 50 tune seeds of each scenario
    python scripts/run_bench.py --configs oracle lidar_and_rsu lidar_only --runs 50 --speed 0
    python scripts/run_bench.py --report                                     # rebuild the report, running nothing
    python scripts/run_bench.py --configs lidar_and_rsu --split test --final --runs 300   # the final measurement, once

oracle is the teaching rule of training/record_hazards.py driving with CARLA's true positions of everything. When even
it collides on a seed, the report counts that seed as one nobody here avoided and gives the rates both with and
without those seeds, rather than counting it against the car. It's a strong driver, not a perfect one, so "the
oracle crashed" is evidence that a seed is very hard, not proof that it's impossible.

Needs CARLA and the Docker stack running with the entities provisioned, and nothing else using CARLA: it starts and
stops the processor itself, once per config. Runs are saved after each one to output/bench/<config>_<split>.jsonl,
so a stopped bench carries on where it left off; the report goes to output/bench/report.md and report.csv.
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from scenario_variation import SCENARIOS, SPLITS, draw, seeds
from study_names import CONFIGS

BENCH_DIR = ROOT / "output" / "bench"
PLANNED = "+planner"     # a config's name with this drives with planner.Planner choosing the speed
BENCH_CONFIGS = ("oracle", *CONFIGS, *(f"{c}{PLANNED}" for c in CONFIGS))
Z95 = 1.96


def wilson(hits, runs, z=Z95):
    """(low, high) of the 95 % Wilson interval for a rate of hits in runs, as fractions; (0, 1) with no runs."""
    if runs == 0:
        return 0.0, 1.0
    p = hits / runs
    middle = (p + z * z / (2 * runs)) / (1 + z * z / runs)
    half = z * math.sqrt(p * (1 - p) / runs + z * z / (4 * runs * runs)) / (1 + z * z / runs)
    return max(0.0, middle - half), min(1.0, middle + half)


def runs_path(config, split):
    return BENCH_DIR / f"{config}_{split}.jsonl"


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
            "stopped_for_nothing": sum(bool(r.get("stopped_for_nothing")) for r in ran)}


def write_report(split, directory=BENCH_DIR):
    every = {config: load_runs(runs_path(config, split)) for config in BENCH_CONFIGS}
    oracle_crashed = defaultdict(set)
    for r in every["oracle"]:
        if r["ran"] and r["collisions"]:
            oracle_crashed[r["scenario"]].add(r["seed"])
    table, lines = [], [f"# Bench: {split} seeds", "",
                        ("Each run is a different situation drawn from its seed. Collision rates come with a 95 % Wilson interval. "
                         "\"Oracle crashed\" counts seeds where even the teaching rule, driving with true positions, hit something; "
                         "\"avoidable\" leaves those out."), "",
                        "| Config | Scenario | Ran | Collisions | Collision rate (95 % range) | Passed | Oracle crashed | Avoidable collisions (95 % range) | Stopped for nothing |",
                        "| ------ | -------- | --- | ---------- | --------------------------- | ------ | -------------- | --------------------------------- | ------------------- |"]
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
                         f"{s['oracle_crashed'] if config != 'oracle' and every['oracle'] else 'n/a'} | {avoid} | {s['stopped_for_nothing']} |")
    lines += ["", ("- **Stopped for nothing** counts oncoming_turner runs where the oncoming car carried straight on and the ML car still "
                   "stopped: giving way to a car that was never going to cross is a mistake too."), ""]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"report_{split}.md").write_text("\n".join(lines), encoding="utf-8")
    if table:
        with open(directory / f"report_{split}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    return lines


def run_config(config, args):
    """Every seed for one config, with its own processor, saving each run as it finishes."""
    import carla
    from run_professor_test_suite import (
        CARLA_HOST,
        CARLA_PORT,
        OUTPUT_DIR,
        Console,
        Policy,
    )
    from run_rsu_comparison import Comparison, start_processor, stop_processor
    from run_study import run_scenario

    mode = "truth" if config == "oracle" else CONFIGS[config.removesuffix(PLANNED)]
    path = runs_path(config, args.split)
    done = {(r["scenario"], r["seed"]) for r in load_runs(path)}
    todo = [(name, seed) for name in args.scenarios for seed in seeds(args.split, args.runs, args.start) if (name, seed) not in done]
    if not todo:
        print(f"[BENCH] {config}: every run is already done.", flush=True)
        return
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    process, log = start_processor(mode, args.model, OUTPUT_DIR / f"bench_processor_{config}.log", view=False,
                                   extra=["--no-draw", "--speed", str(args.speed)])
    try:
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(30.0)
        if config == "oracle":
            from record_hazards import HazardTeacher
            policy = HazardTeacher(Policy())
        else:
            policy = Policy()
        suite = Comparison(carla, client, policy, Console(args.quiet), seed=1, perception=mode)
        if config == "oracle":
            policy.world = suite.world
        if config.endswith(PLANNED):
            from planner import Planner
            from roadside_link import ROADSIDE_MODES
            from run_professor_test_suite import JUNCTION

            suite.planner = Planner(JUNCTION, has_roadside=mode in ROADSIDE_MODES)
        try:
            for index, (name, seed) in enumerate(todo, 1):
                variation = draw(name, seed, args.traffic)
                suite.variation, suite.seed, suite.traffic = variation, seed, variation.traffic
                if config == "oracle":
                    policy.reset()
                row = run_scenario(suite, name, variation.lead)
                if config == "oracle":
                    policy.inputs.clear()  # its lessons aren't kept here; record_hazards.py records those
                row = {"config": config, "split": args.split, "seed": seed, **row, **variation.as_row()}
                row["stopped_for_nothing"] = bool(name == "oncoming_turner" and not variation.turner_turns and row.get("stopped"))
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                suite.console.end_status()
                print(f"[BENCH] {config} {name} seed {seed} ({index}/{len(todo)}): "
                      f"{'COLLISION' if row.get('collisions') else 'no collision'}{'' if row['ran'] else ' (did not run)'}", flush=True)
        finally:
            suite.close()
    finally:
        stop_processor(process, log)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", choices=BENCH_CONFIGS, default=["lidar_and_rsu"])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--split", choices=SPLITS, default="tune")
    parser.add_argument("--final", action="store_true", help="needed to run the test split, which is only for the final measurement")
    parser.add_argument("--runs", type=int, default=50, help="seeds of each scenario")
    parser.add_argument("--start", type=int, default=0, help="skip this many seeds of the split first")
    parser.add_argument("--traffic", type=int, help="background cars in every run, instead of each seed's own")
    parser.add_argument("--speed", type=float, default=1.0, help="the processor's --speed: 0 runs as fast as CARLA can")
    parser.add_argument("--model", default=str(ROOT / "models" / "rsu_detector.candidate.onnx"))
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
                      "--model", args.model, "--scenarios", *args.scenarios, *(["--final"] if args.final else []),
                      *(["--traffic", str(args.traffic)] if args.traffic is not None else []), *(["--quiet"] if args.quiet else [])]
            for config in args.configs:
                subprocess.call([sys.executable, "-u", __file__, *common, "--configs", config], cwd=ROOT)
        else:
            run_config(args.configs[0], args)
    print("\n".join(write_report(args.split)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
