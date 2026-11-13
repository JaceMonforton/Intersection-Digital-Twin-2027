"""
The configs and their windows, shared by run_simulation.py, run_bench.py and the suite. No imports beyond the
standard library, so the launcher can check its flags before it starts anything.

    lidar_and_rsu   the car's own roof LiDAR detector, fused with the roadside LiDAR detector's tracks (perception fused)
    camera_and_rsu  the car's own roof LiDAR detector, fused with the roadside camera detector's   (perception fused_camera)
    lidar_only      only the car's own roof LiDAR detector, with no roadside unit                   (perception onboard)

Every config perceives with trained detectors, as a real car would (see onboard_perception.py), and drives with the
planner and the car's own safety monitor.
"""
CONFIGS = {"lidar_and_rsu": "fused", "camera_and_rsu": "fused_camera", "lidar_only": "onboard"}
# The window that goes with each config, besides the ML car's own split screen
# shortcut: lidar_only has no window yet; add one that reads the roof detector's sweeps
VIEWS = {"fused": ["scripts/lidar_view.py"], "fused_camera": ["scripts/rsu_camera_view.py"]}


def processor_args(perception, model, view=True):
    """central_gpu_processor.py's arguments for a config. The safety gate runs on what the config's roadside unit
    sees, and with no roadside unit on ground truth, which that car ignores."""
    detector = {"fused": ["--detector", "onnx", "--model", str(model)], "fused_camera": ["--detector", "camera"]}
    return [*detector.get(perception, ["--detector", "groundtruth"]), *(["--lidar-view"] if view and perception in VIEWS else [])]
