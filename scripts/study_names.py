"""
The comparative study's configs, scenarios and windows, shared by run_simulation.py, the test suite and run_study.py.
No imports beyond the standard library, so the launcher can check its flags before it starts anything.

    lidar_and_rsu   the roadside LiDAR detector's tracks plus the car's own roof LiDAR       (perception rsu)
    camera_and_rsu  SIMULATED roadside camera tracks plus the car's own roof LiDAR           (perception camera)
    lidar_only      only the car's own roof LiDAR, with no roadside unit and so no safety gate (perception sight)
"""
CONFIGS = {"lidar_and_rsu": "rsu", "camera_and_rsu": "camera", "lidar_only": "sight"}
CONFIG_HELP = {
    "lidar_and_rsu": "the roadside LiDAR detector's tracks plus the car's own roof LiDAR, with the safety gate on the detector",
    "camera_and_rsu": "SIMULATED roadside camera tracks (ground truth plus noise) plus the car's own roof LiDAR, with the gate on them",
    "lidar_only": "only the car's own roof LiDAR: no roadside unit, so no safety gate",
}
SUITE_TESTS = {"traffic_light": 1, "left_turn": 2, "blind_corner": 3, "cut_in": 4, "yellow_light": 5}
# red_light_runner is blind_corner's speeding car with its timing varied between repeats, as run_rsu_comparison.py runs it
HAZARDS = ("red_light_runner", "hidden_pedestrian", "oncoming_turner", "clear_road")
SCENARIOS = (*SUITE_TESTS, *HAZARDS)
# The window that goes with each config, besides the ML car's own split screen
VIEWS = {"rsu": ["scripts/lidar_view.py"], "camera": ["scripts/rsu_camera_view.py"], "sight": ["scripts/lidar_view.py", "--feed", "roof"]}


def processor_args(perception, model, view=True):
    """central_gpu_processor.py's arguments for a config. The safety gate runs on what the config's roadside unit sees,
    and in sight, where there's no roadside unit, on ground truth, which that car ignores."""
    detector = {"rsu": ["--detector", "onnx", "--model", str(model)], "camera": ["--detector", "camera"]}.get(perception, ["--detector", "groundtruth"])
    return [*detector, *(["--lidar-view"] if view and perception in ("rsu", "camera") else [])]


def add_config_flags(parser):
    """The configs as flags that each set perception. Returns the group, so other ways of setting it can join it."""
    group = parser.add_mutually_exclusive_group()
    for flag, mode in CONFIGS.items():
        group.add_argument(f"--{flag}", dest="perception", action="store_const", const=mode, help=CONFIG_HELP[flag])
    return group


def scenarios_from(names):
    """--scenario's names, with all expanded, in the study's order and without repeats."""
    chosen = set(SCENARIOS if "all" in names else names)
    return [name for name in SCENARIOS if name in chosen]
