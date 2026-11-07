"""
The comparative study: the suite's five tests and the hidden hazards, with the ML car told about other actors three
ways (see study_names.py), under each background traffic load. Every figure in the report is measured from CARLA's
true positions, not from what the config perceived, so the configs are scored the same way.

    traffic_light, left_turn, blind_corner, cut_in, yellow_light   the suite's tests 1-5, graded by their own checks
    red_light_runner     test 3's speeding car, with its timing varied between repeats; passes on test 3's checks
    hidden_pedestrian    a person steps out from behind containers; passes with no collision
    oncoming_turner      an oncoming car hidden by containers turns across the ML car; passes with no collision
    clear_road           the containers but no hazard; passes with no collision, no safety brake and no stop

Needs CARLA and the Docker stack running, with the entities provisioned, and nothing else using CARLA: it starts and
stops the processor itself, once per config, with that config's windows unless --no-view. Runs go to
output/comparative_study_runs.json and .csv, and the report to output/comparative_study_report.csv and _summary.md.
    python scripts/run_study.py --matrix --traffic 0 15 --repeat 5
    python scripts/run_study.py --configs camera_and_rsu --scenarios hidden_pedestrian --repeat 1
"""
import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from run_professor_test_suite import CARLA_HOST, CARLA_PORT, OUTPUT_DIR, Console, Policy
from run_rsu_comparison import LEADS, Comparison, start_processor, stop_processor
from study_names import CONFIGS, SCENARIOS, SUITE_TESTS, VIEWS, scenarios_from

ROOT = Path(__file__).resolve().parent.parent
FINAL_SEED = 29  # the roadside detector's final test: never run here, so nothing is tuned on it
RUNS_PATH = OUTPUT_DIR / "comparative_study_runs.json"
REPORT_COLUMNS = ("config", "traffic", "scenario", "score", "runs", "ran", "pass_rate_pct", "collision_rate_pct", "false_interventions",
                  "ttc_mean_s", "ttc_p95_s", "ttc_p5_s", "ttc_runs", "clearance_mean_s", "cleared")
# Points out of 100 for each part of the score, fixed before any results were seen: safety first, then getting the job done
SCORE_WEIGHTS = {"no collision": 50, "pass": 30, "no false alarm": 10, "TTC margin": 10}
SAFE_MARGIN = 3.0  # s of TTC in the low tail (P5) that earns all of its points


def study_seeds(count):
    return list(itertools.islice((s for s in itertools.count(1) if s != FINAL_SEED), count))


def gate_ignored(suite):
    """In sight the processor still runs a ground-truth safety gate, but that car has no roadside unit and drives on
    through it (see Suite.run_stage), so its engagements are no interventions and aren't counted."""
    return suite.perception == "sight"


def run_scenario(suite, name, lead):
    """One run of a scenario, as a flat row of what was measured."""
    hazard = None
    try:
        if name in SUITE_TESTS:
            tests = {1: suite.test_lights, 2: suite.test_left_turn, 3: suite.test_blind_corner, 4: suite.test_cut_in, 5: suite.test_light_change}
            result, ran = tests[SUITE_TESTS[name]](), True
            passed = result.passed
        elif name == "red_light_runner":
            result, _, _, ran = suite.run_runner(lead)
            passed = result.passed
        else:
            hazard = {"hidden_pedestrian": lambda: suite.run_pedestrian(lead), "oncoming_turner": lambda: suite.run_turner(lead),
                      "clear_road": lambda: suite.run_pedestrian(lead, launch=False)}[name]
            result, _, stopped, ran = hazard()
            passed = ran and not result.collisions
            if name == "clear_road":
                passed = passed and not stopped and (gate_ignored(suite) or not result.brake_events)
                result.false_brakes = result.brake_events  # with no hazard, every brake is a false alarm
    except RuntimeError as error:
        suite.console.line("ALERT", f"{name} couldn't run: {error}")
        return {"scenario": name, "ran": False, "passed": False, "error": str(error)}
    r = asdict(result)
    if gate_ignored(suite):
        r["brake_events"] = r["false_brakes"] = 0
    return {"scenario": name, "lead": lead if hazard or name == "red_light_runner" else None, "ran": bool(ran), "passed": bool(passed and ran),
            "collisions": len(r["collisions"]), "background_collisions": sum(c.get("role") == "background" for c in r["collisions"]),
            "brake_events": r["brake_events"], "false_brakes": r["false_brakes"], "true_min_ttc": r["true_min_ttc"],
            "clearance_s": r["clearance_s"], "traffic_spawned": r["traffic"], "duration_s": r["duration"], "notes": "; ".join(r["notes"]),
            "failed_checks": ", ".join(check for check, ok in r["checks"].items() if not ok)}


def run_runs(suite, names, traffic_loads, repeat, config):
    rows = []
    for traffic in traffic_loads:
        suite.traffic = traffic
        for name in names:
            for index, seed in enumerate(study_seeds(repeat)):
                suite.seed = seed
                row = {"config": config, "traffic": traffic, "repeat": index + 1, "seed": seed, **run_scenario(suite, name, LEADS[index % len(LEADS)])}
                suite.console.end_status()
                suite.console.line("SCENARIO", f"{config} traffic {traffic} {name} {index + 1}/{repeat}: {'PASS' if row['passed'] else 'FAIL'}"
                                   + ("" if row["ran"] else " (didn't run)"))
                rows.append(row)
    return rows


def run_attached(carla, client, names, perception, traffic, repeat, console, roof_view=False):
    """For the test suite's --scenario: the same runs, on a processor someone else started."""
    config = next((flag for flag, mode in CONFIGS.items() if mode == perception), perception)
    suite = Comparison(carla, client, Policy(), console, seed=1, perception=perception, roof_view=roof_view)
    try:
        return run_runs(suite, names, [traffic], repeat, config)
    finally:
        suite.close()


def summarise(rows):
    """The report's figures for a group of runs. Rates are of the runs that ran. TTC is each run's lowest true TTC,
    over the runs where something was ever on a collision course; the low tail (P5) is the dangerous one."""
    ran = [r for r in rows if r["ran"]]
    ttcs = [r["true_min_ttc"] for r in ran if r.get("true_min_ttc") is not None]
    cleared = [r["clearance_s"] for r in ran if r.get("clearance_s") is not None]

    def rounded(value, digits=2):
        return None if value is None else round(float(value), digits)

    return {"runs": len(rows), "ran": len(ran),
            "pass_rate_pct": rounded(100.0 * sum(r["passed"] for r in ran) / len(ran), 1) if ran else None,
            "collision_rate_pct": rounded(100.0 * sum(r["collisions"] > 0 for r in ran) / len(ran), 1) if ran else None,
            "false_interventions": sum(r["false_brakes"] for r in ran),
            "ttc_mean_s": rounded(np.mean(ttcs)) if ttcs else None, "ttc_p95_s": rounded(np.percentile(ttcs, 95)) if ttcs else None,
            "ttc_p5_s": rounded(np.percentile(ttcs, 5)) if ttcs else None, "ttc_runs": len(ttcs),
            "clearance_mean_s": rounded(np.mean(cleared)) if cleared else None, "cleared": len(cleared)}


def score(s):
    """A config's score out of 100 from its summary, weighted by SCORE_WEIGHTS: the share of runs without a collision,
    the pass rate, false alarms per run (one or more a run scores nothing), and how much of SAFE_MARGIN the low TTC
    tail keeps (all of it if nothing was ever on a collision course). None if nothing ran."""
    if not s["ran"]:
        return None
    w = SCORE_WEIGHTS
    margin = 1.0 if s["ttc_p5_s"] is None else min(s["ttc_p5_s"] / SAFE_MARGIN, 1.0)
    return round(w["no collision"] * (1 - s["collision_rate_pct"] / 100) + w["pass"] * s["pass_rate_pct"] / 100
                 + w["no false alarm"] * max(0.0, 1 - s["false_interventions"] / s["ran"]) + w["TTC margin"] * margin, 1)


def report_rows(runs):
    """A row for each config, traffic load and scenario, then one for all the scenarios together."""
    rows = []
    for config, traffic in sorted({(r["config"], r["traffic"]) for r in runs}, key=lambda k: (k[1], list(CONFIGS).index(k[0]) if k[0] in CONFIGS else 9)):
        mine = [r for r in runs if r["config"] == config and r["traffic"] == traffic]
        for name in [*(n for n in SCENARIOS if any(r["scenario"] == n for r in mine)), "all"]:
            group = mine if name == "all" else [r for r in mine if r["scenario"] == name]
            summary = summarise(group)
            rows.append({"config": config, "traffic": traffic, "scenario": name, "score": score(summary), **summary})
    return rows


def ranking(runs):
    """The configs best first, for each traffic load and for every load together, as (load, [(config, summary)])."""
    loads = [*sorted({r["traffic"] for r in runs}), "all"]
    ranked = []
    for load in loads:
        mine = [r for r in runs if load == "all" or r["traffic"] == load]
        scored = [(config, {"score": score(s), **s}) for config in CONFIGS if (s := summarise([r for r in mine if r["config"] == config]))["ran"]]
        ranked.append((load, sorted(scored, key=lambda c: -c[1]["score"])))
    return ranked


def ranking_lines(runs):
    lines = ["## Which config is best", "",
             (f"Each config is scored out of 100 from its measured results: {', '.join(f'{p} {n}' for n, p in SCORE_WEIGHTS.items())} points "
             f"(TTC margin: the low tail, P5, against {SAFE_MARGIN:.0f} s). The weights were fixed before any results were seen."), ""]
    for load, scored in ranking(runs):
        if not scored:
            continue
        lines += [f"### {'All traffic loads' if load == 'all' else f'{load} background cars'}", "",
                  "| Rank | Config | Score | Collision rate | Pass rate | False alarms per run | TTC P5 |",
                  "| ---- | ------ | ----- | -------------- | --------- | -------------------- | ------ |"]
        for rank, (config, s) in enumerate(scored, 1):
            lines.append(f"| {rank} | {config} | **{s['score']}** | {s['collision_rate_pct']} % | {s['pass_rate_pct']} % "
                         f"| {s['false_interventions'] / s['ran']:.2f} | {shown(s['ttc_p5_s'], ' s')} |")
        best, *rest = scored
        lines += ["", f"Best: **{best[0]}** ({best[1]['score']})" + "".join(f", then {c} ({s['score']})" for c, s in rest) + ".", ""]
    return lines


def shown(value, unit=""):
    return "n/a" if value is None else f"{value}{unit}"


def write_report(runs, directory=OUTPUT_DIR):
    runs = [{**r, "passed": bool(r["passed"] and r["ran"])} for r in runs]  # runs saved before that rule was in run_scenario
    rows = report_rows(runs)
    with open(directory / "comparative_study_report.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with open(directory / "comparative_study_runs.csv", "w", newline="", encoding="utf-8") as file:
        columns = list(dict.fromkeys(key for r in runs for key in r))
        writer = csv.DictWriter(file, columns)
        writer.writeheader()
        writer.writerows(runs)
    lines = ["# Comparative study", "", (f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
             f"{len(runs)} runs; every figure is measured."), "", *ranking_lines(runs)]
    for traffic in sorted({r["traffic"] for r in rows}):
        lines += [f"## {traffic} background cars, by scenario", "",
                  "| Config | Scenario | Score | Ran | Pass rate | Collision rate | False alarms | TTC mean | TTC P95 | TTC P5 | Clearance |",
                  "| ------ | -------- | ----- | --- | --------- | -------------- | ------------ | -------- | ------- | ------ | --------- |"]
        for r in (r for r in rows if r["traffic"] == traffic):
            name = "**all**" if r["scenario"] == "all" else r["scenario"]
            lines.append(f"| {r['config']} | {name} | {shown(r['score'])} | {r['ran']} of {r['runs']} | {shown(r['pass_rate_pct'], ' %')} | {shown(r['collision_rate_pct'], ' %')} "
                         f"| {r['false_interventions']} | {shown(r['ttc_mean_s'], ' s')} | {shown(r['ttc_p95_s'], ' s')} | {shown(r['ttc_p5_s'], ' s')} "
                         f"| {shown(r['clearance_mean_s'], ' s')} ({r['cleared']}) |")
        lines.append("")
    lines += ["## How each figure is measured", "",
              "- **Pass rate**: the suite's tests by their own checks; the hazards with no collision; clear_road with no collision, no safety brake and no stop. Of the runs that ran.",
              "- **Collision rate**: runs with at least one collision. `background_collisions` in the runs file counts those with background traffic.",
              ("- **False alarms**: safety-brake engagements while CARLA's true TTC stayed at or above 3 s from 1 s before to 3 s after, with no collision then; in clear_road every brake. "
              "A brake that prevented a conflict can also meet that rule, so this is an upper bound."),
              ("- **TTC**: each run's lowest true straight-line TTC to any car or person, as the gate measures it but from CARLA's true positions. "
              "Mean, P95 and P5 are over the runs where anything was ever on a collision course. P5 is the dangerous tail."),
              ("- **Clearance**: seconds from first entering junction 189 to first leaving it, over the runs that got through (the count in brackets). "
               "The hidden hazards and clear_road end about 10 m past their conflict point, still inside the junction, so only the suite's tests give one."),
              "- lidar_only has no roadside unit, so no safety gate: its false alarms are always 0.",
              ("- camera_and_rsu is SIMULATED: CARLA's true positions of the actors the roadside mounts have a clear line to, with assumed camera errors "
              "(scripts/rsu_camera_sim.py). It can't mistake the containers for parked cars, so it's optimistic."),
              "- red_light_runner is blind_corner's speeding car with its timing varied between repeats.", ""]
    (directory / "comparative_study_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


def spawn_window(args, name):
    log = open(OUTPUT_DIR / f"study_{name}.log", "w", encoding="utf-8", errors="replace")  # noqa: SIM115
    process = subprocess.Popen([sys.executable, "-u", *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
    return process, log


def run_config(config, args):
    """Every run for one config, with its own processor and windows."""
    import carla

    mode = CONFIGS[config]
    console = Console(args.quiet)
    console.line("SCENARIO", f"=== Config: {config} ===")
    process, log = start_processor(mode, args.model, OUTPUT_DIR / f"study_processor_{config}.log", view=not args.no_view)
    windows = [] if args.no_view else [spawn_window(["scripts/ml_car_view.py"], f"{config}_car_view"), spawn_window(VIEWS[mode], f"{config}_view")]
    try:
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(30.0)
        suite = Comparison(carla, client, Policy(), console, seed=1, perception=mode, roof_view=not args.no_view)
        try:
            rows = run_runs(suite, scenarios_from(args.scenarios), args.traffic, args.repeat, config)
        finally:
            suite.close()
        shutil.copy(OUTPUT_DIR / "live_telemetry.csv", OUTPUT_DIR / f"study_telemetry_{config}.csv")
    finally:
        for window in windows:
            stop_processor(*window)
        stop_processor(process, log)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", action="store_true", help="every config and every scenario")
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--scenarios", nargs="+", choices=("all", *SCENARIOS), default=["all"])
    parser.add_argument("--traffic", type=int, nargs="+", default=[0, 15], help="background car counts, each run in turn")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--model", default=str(ROOT / "models" / "rsu_detector.candidate.onnx"))
    parser.add_argument("--no-view", action="store_true", help="don't open each config's windows, which is quicker")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="rebuild the report from output/comparative_study_runs.json, running nothing")
    parser.add_argument("--append", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.report_only:
        write_report(json.loads(RUNS_PATH.read_text(encoding="utf-8"))["runs"])
        print((OUTPUT_DIR / "comparative_study_summary.md").read_text(encoding="utf-8"))
        return 0
    if args.matrix:
        args.configs, args.scenarios = list(CONFIGS), ["all"]
    if min(args.traffic) < 0 or args.repeat < 1:
        parser.error("--traffic must be 0 or more, and --repeat at least 1")

    OUTPUT_DIR.mkdir(exist_ok=True)
    if not args.append:
        RUNS_PATH.unlink(missing_ok=True)
    if len(args.configs) > 1:
        # Each config gets its own process, as a Traffic Manager doesn't survive being reused after its processor changes
        rest = [a for a in sys.argv[1:] if a not in (*args.configs, "--configs", "--matrix")]
        codes = [subprocess.call([sys.executable, "-u", __file__, *rest, *(["--scenarios", "all"] if args.matrix else []), "--append", "--configs", config], cwd=ROOT)
                 for config in args.configs]
        runs = json.loads(RUNS_PATH.read_text(encoding="utf-8"))["runs"] if RUNS_PATH.exists() else []
        if runs:
            write_report(runs)
        print("\n" + (OUTPUT_DIR / "comparative_study_summary.md").read_text(encoding="utf-8") if runs else "[STUDY] No runs finished.")
        return max(codes)

    rows = run_config(args.configs[0], args)
    runs = (json.loads(RUNS_PATH.read_text(encoding="utf-8"))["runs"] if RUNS_PATH.exists() else []) + rows
    RUNS_PATH.write_text(json.dumps({"generated": datetime.now(timezone.utc).isoformat(), "model": args.model, "runs": runs}, indent=2), encoding="utf-8")
    write_report(runs)
    return 0 if all(r["passed"] for r in rows if r["ran"]) else 2


if __name__ == "__main__":
    sys.exit(main())
