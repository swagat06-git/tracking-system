import math
import time
from copy import deepcopy
from typing import Any

from simulation.simulator import Simulator
from tracking.system import TrackingSystem


ALLOWED_MOTIONS = {"linear", "circular", "figure8", "random"}
ALLOWED_ATMOSPHERES = {"clear", "haze", "fog", "rain", "low_light"}
ALLOWED_NOISE = {"none", "gaussian", "salt_pepper", "poisson"}

SCENARIO_FRAMES = 300


def run_scenario(
    base_config: dict[str, Any],
    motion: str,
    atmosphere: str,
    noise_type: str = "none",
    noise_level: float = 0.0,
) -> dict[str, Any]:
    if motion not in ALLOWED_MOTIONS:
        raise ValueError(
            f"Unsupported motion. Choose one of: {', '.join(sorted(ALLOWED_MOTIONS))}."
        )

    if atmosphere not in ALLOWED_ATMOSPHERES:
        raise ValueError(
            f"Unsupported atmosphere. Choose one of: {', '.join(sorted(ALLOWED_ATMOSPHERES))}."
        )

    if noise_type not in ALLOWED_NOISE:
        raise ValueError(
            f"Unsupported noise type. Choose one of: {', '.join(sorted(ALLOWED_NOISE))}."
        )

    if noise_level < 0:
        raise ValueError("Noise level cannot be negative.")

    config = deepcopy(base_config)
    config["motion"]["type"] = motion
    config["disturbance"]["atmosphere"] = atmosphere
    config["disturbance"]["noise_type"] = noise_type
    config["disturbance"]["noise_level"] = noise_level

    simulator = Simulator(config)
    tracker = TrackingSystem(
        config=config,
        acceleration_noise=500.0,
        gain=4.0,
        velocity_scale=0.5,
    )

    detected_frames = 0
    tracking_frames = 0
    first_detection_frame: int | None = None
    processing_times: list[float] = []
    errors: list[float] = []

    started = time.perf_counter()

    for frame_number in range(1, SCENARIO_FRAMES + 1):
        simulator.update()
        frame = simulator.get_frame()
        ground_truth = simulator.get_ground_truth()

        process_started = time.perf_counter()
        result = tracker.process(frame)
        processing_times.append(time.perf_counter() - process_started)

        detected = bool(result["detected"])
        tracking = bool(result["tracking"])

        if detected:
            detected_frames += 1
            if first_detection_frame is None:
                first_detection_frame = frame_number

        if tracking:
            tracking_frames += 1

        position = result.get("position")
        if position is not None:
            errors.append(
                math.hypot(
                    position["x"] - ground_truth["x"],
                    position["y"] - ground_truth["y"],
                )
            )

        command = result.get("command") or {}
        simulator.move_camera(
            float(command.get("pan_speed", 0.0) or 0.0),
            float(command.get("tilt_speed", 0.0) or 0.0),
        )

    runtime = time.perf_counter() - started
    fps = config["camera"]["fps"]
    duration = SCENARIO_FRAMES / fps
    average_processing = (
        sum(processing_times) / len(processing_times)
        if processing_times
        else 0.0
    )
    measured_fps = (
        1.0 / average_processing
        if average_processing > 0
        else 0.0
    )
    tracking_rate = tracking_frames / SCENARIO_FRAMES * 100.0

    return {
        "status": "completed",
        "generated_at": time.time(),
        "scenario": {
            "motion": motion,
            "atmosphere": atmosphere,
            "noise_type": noise_type,
            "noise_level": noise_level,
            "frames": SCENARIO_FRAMES,
            "duration_seconds": duration,
            "input_fps": fps,
        },
        "benchmark": {
            "benchmark_runtime_seconds": runtime,
            "measured_processing_fps": measured_fps,
            "average_processing_ms": average_processing * 1000.0,
            "max_processing_ms": (
                max(processing_times) * 1000.0
                if processing_times
                else 0.0
            ),
        },
        "tracking": {
            "detected_frames": detected_frames,
            "tracking_frames": tracking_frames,
            "detection_rate_percent": (
                detected_frames / SCENARIO_FRAMES * 100.0
            ),
            "lock_retention_percent": tracking_rate,
            "target_loss_percent": 100.0 - tracking_rate,
            "first_detection_frame": first_detection_frame,
            "acquisition_time_seconds": (
                first_detection_frame / fps
                if first_detection_frame is not None
                else None
            ),
        },
        "accuracy": {
            "frames_with_error": len(errors),
            "average_centroid_error_pixels": (
                sum(errors) / len(errors)
                if errors
                else None
            ),
            "maximum_centroid_error_pixels": (
                max(errors)
                if errors
                else None
            ),
            "rmse_pixels": (
                math.sqrt(sum(error * error for error in errors) / len(errors))
                if errors
                else None
            ),
        },
    }
