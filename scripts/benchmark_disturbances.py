import sys
from pathlib import Path

# Add project root to Python module search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import math
import numpy as np

from simulation.simulator import Simulator
from tracking.pipeline import TrackingPipeline
from control.controller import CameraController


def load_config(config_path: str = "config/config.json") -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def run_benchmark_trial(config: dict, jitter_level: float, noise_level: float, steps: int = 180) -> dict:
    """
    Runs a single simulation trial under specific jitter and noise levels.
    """
    trial_config = json.loads(json.dumps(config))
    trial_config["disturbance"]["camera_jitter"] = jitter_level
    trial_config["disturbance"]["noise_level"] = noise_level
    if noise_level > 0:
        trial_config["disturbance"]["noise_type"] = "gaussian"

    simulator = Simulator(trial_config)
    simulator.target.set_velocity(2.0, 1.0)

    pipeline = TrackingPipeline()

    controller = CameraController(
        frame_width=trial_config["camera"]["width"],
        frame_height=trial_config["camera"]["height"],
        max_pan_speed=trial_config["control"]["max_pan_speed"],
        max_tilt_speed=trial_config["control"]["max_tilt_speed"],
        gain=trial_config["control"]["gain"]
    )

    center_x = trial_config["camera"]["width"] / 2
    center_y = trial_config["camera"]["height"] / 2

    center_errors = []
    detection_failures = 0
    out_of_view_frames = 0

    for _ in range(steps):
        simulator.update()
        frame = simulator.get_frame()
        ground_truth = simulator.get_ground_truth()

        target_x = ground_truth["x"]
        target_y = ground_truth["y"]

        inside_camera = (
            0 <= target_x < trial_config["camera"]["width"]
            and 0 <= target_y < trial_config["camera"]["height"]
        )

        if not inside_camera:
            out_of_view_frames += 1

        result = pipeline.process(frame)

        if not result["detected"]:
            if inside_camera:
                detection_failures += 1
            continue

        estimated = result["position"]

        command = controller.compute_command(
            estimated["x"],
            estimated["y"]
        )

        simulator.move_camera(
            command["pan_speed"],
            command["tilt_speed"]
        )

        new_ground_truth = simulator.get_ground_truth()

        error_x = new_ground_truth["x"] - center_x
        error_y = new_ground_truth["y"] - center_y
        center_error = math.sqrt(error_x ** 2 + error_y ** 2)

        center_errors.append(center_error)

    mean_err = np.mean(center_errors) if center_errors else float("nan")
    max_err = np.max(center_errors) if center_errors else float("nan")

    return {
        "jitter_level": jitter_level,
        "noise_level": noise_level,
        "total_frames": steps,
        "tracked_frames": len(center_errors),
        "mean_error_px": round(float(mean_err), 2),
        "max_error_px": round(float(max_err), 2),
        "out_of_view_frames": out_of_view_frames,
        "detection_failures": detection_failures
    }


def main():
    config = load_config()
    
    jitter_levels = [0.0, 1.0, 3.0, 5.0, 8.0]
    noise_levels = [0.0, 2.0, 5.0]

    results = []

    print(f"\n{'='*80}")
    print(f"{'DAY 4: SYSTEMATIC DISTURBANCE BENCHMARK':^80}")
    print(f"{'='*80}")
    print(f"{'Jitter (px)':<12} | {'Noise (px)':<12} | {'Mean Error':<14} | {'Max Error':<14} | {'OOV Frames':<10} | {'Failures':<10}")
    print("-" * 80)

    for jitter in jitter_levels:
        for noise in noise_levels:
            res = run_benchmark_trial(config, jitter_level=jitter, noise_level=noise)
            results.append(res)
            print(
                f"{res['jitter_level']:<12.1f} | "
                f"{res['noise_level']:<12.1f} | "
                f"{res['mean_error_px']:<14.2f} | "
                f"{res['max_error_px']:<14.2f} | "
                f"{res['out_of_view_frames']:<10} | "
                f"{res['detection_failures']:<10}"
            )

    print(f"{'='*80}\n")

    out_file = Path("benchmark_disturbances_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results successfully exported to {out_file.resolve()}\n")


if __name__ == "__main__":
    main()