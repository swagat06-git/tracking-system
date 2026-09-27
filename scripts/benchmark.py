import csv
import json
import os
import time

from simulation.simulator import Simulator
from tracking.system import TrackingSystem


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

CONFIG_PATH = "config/config.json"

# Number of simulation frames to benchmark.
BENCHMARK_FRAMES = 900       # 30 seconds at 30 FPS

# Frames used before measurement to warm up the ML model.
WARMUP_FRAMES = 10

OUTPUT_DIR = "outputs/benchmark"


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def load_config():
    with open(CONFIG_PATH, "r") as file:
        return json.load(file)


def calculate_error(estimated, ground_truth):
    if estimated is None or ground_truth is None:
        return None

    dx = estimated["x"] - ground_truth["x"]
    dy = estimated["y"] - ground_truth["y"]

    error = (dx * dx + dy * dy) ** 0.5

    return {
        "dx": dx,
        "dy": dy,
        "error": error,
    }


# ---------------------------------------------------------
# BENCHMARK
# ---------------------------------------------------------

def run_benchmark():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = load_config()

    simulator = Simulator(config)

    tracking_system = TrackingSystem(
        config=config,
        acceleration_noise=500.0,
        gain=4.0,
        velocity_scale=0.5,
    )

    target_fps = config["camera"]["fps"]

    print()
    print("=" * 60)
    print("ATLAS PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"Target FPS       : {target_fps}")
    print(f"Benchmark frames : {BENCHMARK_FRAMES}")
    print(f"Duration         : {BENCHMARK_FRAMES / target_fps:.2f} sec")
    print("=" * 60)
    print()

    # -----------------------------------------------------
    # WARMUP
    # -----------------------------------------------------

    print("Warming up ML pipeline...")

    for _ in range(WARMUP_FRAMES):
        simulator.update()
        frame = simulator.get_frame()
        tracking_system.process(frame)

    print("Warmup complete.")
    print()

    # -----------------------------------------------------
    # MEASUREMENT
    # -----------------------------------------------------

    records = []

    processing_times = []

    detected_frames = 0
    tracking_frames = 0
    lost_frames = 0

    acquisition_frame = None
    acquisition_time = None

    errors = []

    benchmark_start = time.perf_counter()

    for frame_number in range(BENCHMARK_FRAMES):

        # ---------------------------------------------
        # Update simulation
        # ---------------------------------------------

        simulator.update()

        # Ground truth BEFORE camera movement
        ground_truth = simulator.get_ground_truth()

        # ---------------------------------------------
        # Capture camera frame
        # ---------------------------------------------

        frame = simulator.get_frame()

        # ---------------------------------------------
        # Run tracking system
        # ---------------------------------------------

        processing_start = time.perf_counter()

        result = tracking_system.process(frame)

        processing_end = time.perf_counter()

        processing_time = processing_end - processing_start

        processing_times.append(processing_time)

        # ---------------------------------------------
        # Tracking state
        # ---------------------------------------------

        detected = bool(result["detected"])
        tracking = bool(result["tracking"])

        if detected:
            detected_frames += 1

        if tracking:
            tracking_frames += 1
        else:
            lost_frames += 1

        # ---------------------------------------------
        # Acquisition time
        # ---------------------------------------------

        if tracking and acquisition_frame is None:
            acquisition_frame = frame_number

            acquisition_time = frame_number / target_fps

        # ---------------------------------------------
        # Position error
        # ---------------------------------------------

        estimated_position = result.get("position")

        error_data = calculate_error(
            estimated_position,
            ground_truth,
        )

        if error_data is not None:
            errors.append(error_data["error"])

        # ---------------------------------------------
        # Move virtual camera
        # ---------------------------------------------

        command = result.get("command")

        if command is not None:
            simulator.move_camera(
                command["pan_speed"],
                command["tilt_speed"],
            )

        # ---------------------------------------------
        # Save frame record
        # ---------------------------------------------

        records.append(
            {
                "frame": frame_number,
                "ground_truth_x": ground_truth["x"],
                "ground_truth_y": ground_truth["y"],
                "estimated_x": (
                    estimated_position["x"]
                    if estimated_position is not None
                    else None
                ),
                "estimated_y": (
                    estimated_position["y"]
                    if estimated_position is not None
                    else None
                ),
                "error_pixels": (
                    error_data["error"]
                    if error_data is not None
                    else None
                ),
                "detected": detected,
                "tracking": tracking,
                "processing_time_ms": processing_time * 1000,
            }
        )

    benchmark_end = time.perf_counter()

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    total_runtime = benchmark_end - benchmark_start

    simulation_duration = BENCHMARK_FRAMES / target_fps

    measured_fps = BENCHMARK_FRAMES / total_runtime

    average_processing_ms = (
        sum(processing_times) / len(processing_times) * 1000
        if processing_times
        else 0
    )

    max_processing_ms = (
        max(processing_times) * 1000
        if processing_times
        else 0
    )

    average_error = (
        sum(errors) / len(errors)
        if errors
        else None
    )

    max_error = (
        max(errors)
        if errors
        else None
    )

    rmse = (
        (
            sum(error ** 2 for error in errors)
            / len(errors)
        ) ** 0.5
        if errors
        else None
    )

    if acquisition_frame is not None:
        acquisition_time = acquisition_frame / target_fps

    lock_retention = (
        tracking_frames / BENCHMARK_FRAMES * 100
    )

    target_loss = (
        lost_frames / BENCHMARK_FRAMES * 100
    )

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    results = {
        "simulation": {
            "duration_seconds": simulation_duration,
            "frames": BENCHMARK_FRAMES,
            "configured_fps": target_fps,
            "measured_fps": measured_fps,
        },

        "tracking": {
            "detected_frames": detected_frames,
            "tracking_frames": tracking_frames,
            "lost_frames": lost_frames,
            "lock_retention_percent": lock_retention,
            "target_loss_percent": target_loss,
        },

        "accuracy": {
            "average_error_pixels": average_error,
            "maximum_error_pixels": max_error,
            "rmse_pixels": rmse,
        },

        "acquisition": {
            "acquisition_time_seconds": acquisition_time,
        },

        "processing": {
            "average_processing_time_ms": average_processing_ms,
            "maximum_processing_time_ms": max_processing_ms,
        },
    }

    # -----------------------------------------------------
    # SAVE JSON
    # -----------------------------------------------------

    json_path = os.path.join(
        OUTPUT_DIR,
        "benchmark_results.json",
    )

    with open(json_path, "w") as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # SAVE CSV
    # -----------------------------------------------------

    csv_path = os.path.join(
        OUTPUT_DIR,
        "tracking_log.csv",
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=records[0].keys(),
        )

        writer.writeheader()
        writer.writerows(records)

    # -----------------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("ATLAS BENCHMARK RESULTS")
    print("=" * 60)

    print()
    print("PERFORMANCE")
    print("-" * 60)
    print(f"Configured FPS       : {target_fps:.2f}")
    print(f"Measured FPS         : {measured_fps:.2f}")
    print(
        f"Avg processing time  : "
        f"{average_processing_ms:.2f} ms"
    )
    print(
        f"Max processing time  : "
        f"{max_processing_ms:.2f} ms"
    )

    print()
    print("TRACKING")
    print("-" * 60)
    print(
        f"Acquisition time     : "
        f"{acquisition_time:.3f} sec"
        if acquisition_time is not None
        else "Acquisition time     : NOT ACQUIRED"
    )

    print(
        f"Lock retention       : "
        f"{lock_retention:.2f}%"
    )

    print(
        f"Target loss          : "
        f"{target_loss:.2f}%"
    )

    print()
    print("ACCURACY")
    print("-" * 60)

    print(
        f"Average error        : "
        f"{average_error:.3f} px"
        if average_error is not None
        else "Average error        : N/A"
    )

    print(
        f"Maximum error        : "
        f"{max_error:.3f} px"
        if max_error is not None
        else "Maximum error        : N/A"
    )

    print(
        f"RMSE                 : "
        f"{rmse:.3f} px"
        if rmse is not None
        else "RMSE                 : N/A"
    )

    print()
    print("FILES")
    print("-" * 60)
    print(f"JSON : {json_path}")
    print(f"CSV  : {csv_path}")

    print()
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print()


if __name__ == "__main__":
    run_benchmark()
