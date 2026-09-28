import json
import os
import time

from simulation.simulator import Simulator
from tracking.system import TrackingSystem
from tracking.performance_logger import PerformanceLogger


CONFIG_PATH = "config/config.json"

BENCHMARK_FRAMES = 900
WARMUP_FRAMES = 10

OUTPUT_DIR = "outputs/benchmark"


def load_config():
    with open(CONFIG_PATH, "r") as file:
        return json.load(file)


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

    logger = PerformanceLogger(
        output_dir=OUTPUT_DIR
    )

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
    # START LOGGER
    # -----------------------------------------------------

    logger.start()

    benchmark_start = time.perf_counter()

    # -----------------------------------------------------
    # MEASUREMENT
    # -----------------------------------------------------

    for frame_number in range(BENCHMARK_FRAMES):

        simulator.update()

        # get_frame() MUST happen before get_ground_truth()
        # because camera jitter is generated inside get_frame().
        frame = simulator.get_frame()

        ground_truth = simulator.get_ground_truth()

        processing_start = time.perf_counter()

        result = tracking_system.process(frame)

        processing_end = time.perf_counter()

        processing_time = (
            processing_end - processing_start
        )

        timestamp_seconds = (
            frame_number / target_fps
        )

        logger.record_frame(
            frame_number=frame_number,
            timestamp_seconds=timestamp_seconds,
            result=result,
            processing_time_seconds=processing_time,
            ground_truth=ground_truth,
        )

        # Move virtual camera after processing.
        command = result.get("command")

        if command is not None:

            simulator.move_camera(
                command["pan_speed"],
                command["tilt_speed"],
            )

    benchmark_end = time.perf_counter()

    # -----------------------------------------------------
    # BENCHMARK FPS
    # -----------------------------------------------------

    total_runtime = benchmark_end - benchmark_start

    measured_fps = (
        BENCHMARK_FRAMES / total_runtime
        if total_runtime > 0
        else 0
    )

    # -----------------------------------------------------
    # SAVE PERFORMANCE REPORT
    # -----------------------------------------------------

    logger_result = logger.save(
        name="simulator_benchmark",
        video_fps=target_fps,
    )

    summary = logger_result["summary"]

    # Add benchmark-level FPS to the existing JSON structure.
    summary["benchmark"] = {
        "measured_fps": measured_fps,
        "benchmark_runtime_seconds": total_runtime,
    }

    # Save updated JSON.
    with open(
        logger_result["json"],
        "w",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("ATLAS BENCHMARK RESULTS")
    print("=" * 60)

    print()
    print("PERFORMANCE")
    print("-" * 60)

    print(
        f"Configured FPS       : "
        f"{target_fps:.2f}"
    )

    print(
        f"Measured FPS         : "
        f"{measured_fps:.2f}"
    )

    print(
        f"Avg processing time  : "
        f"{summary['processing']['average_processing_ms']:.2f} ms"
    )

    print(
        f"Max processing time  : "
        f"{summary['processing']['max_processing_ms']:.2f} ms"
    )

    print()
    print("TRACKING")
    print("-" * 60)

    acquisition_time = (
        summary["acquisition"]["acquisition_time_seconds"]
    )

    print(
        f"Acquisition time     : "
        f"{acquisition_time:.3f} sec"
        if acquisition_time is not None
        else "Acquisition time     : NOT ACQUIRED"
    )

    print(
        f"Detection rate       : "
        f"{summary['detection']['detection_rate_percent']:.2f}%"
    )

    print(
        f"Lock retention       : "
        f"{summary['tracking']['lock_retention_percent']:.2f}%"
    )

    print(
        f"Target loss          : "
        f"{summary['tracking']['target_loss_percent']:.2f}%"
    )

    print()
    print("ACCURACY")
    print("-" * 60)

    average_error = (
        summary["error"]["average_error_pixels"]
    )

    max_error = (
        summary["error"]["max_error_pixels"]
    )

    rmse = (
        summary["error"]["rmse_pixels"]
    )

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

    print(f"JSON : {logger_result['json']}")
    print(f"CSV  : {logger_result['csv']}")

    print()
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print()


if __name__ == "__main__":
    run_benchmark()