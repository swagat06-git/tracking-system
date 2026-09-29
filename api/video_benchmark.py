import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2

from tracking.system import TrackingSystem


MAX_VIDEO_SECONDS = 30.0
MAX_VIDEO_SIZE_BYTES = 100 * 1024 * 1024
MIN_FPS = 1.0
MAX_FPS = 31.0
EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 480


def _load_ground_truth(path: str) -> dict[int, dict[str, float]]:
    ground_truth: dict[int, dict[str, float]] = {}

    import csv

    with open(path, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            frame = int(row["frame"])
            ground_truth[frame] = {
                "x": float(row["ground_truth_x"]),
                "y": float(row["ground_truth_y"]),
            }

    return ground_truth


def _process_video(
    video_path: str,
    config: dict[str, Any],
    ground_truth_path: str | None = None,
) -> dict[str, Any]:
    source = cv2.VideoCapture(video_path)

    if not source.isOpened():
        raise ValueError("The uploaded file could not be opened as a video.")

    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(source.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(source.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if fps < MIN_FPS:
        source.release()
        raise ValueError("The uploaded video has no valid frame rate.")

    if fps > MAX_FPS:
        source.release()
        raise ValueError(
            "Video must be recorded at about 30 FPS. "
            f"Detected {fps:.2f} FPS."
        )

    duration = frame_count / fps if frame_count > 0 else 0.0

    if duration > MAX_VIDEO_SECONDS + 0.05:
        source.release()
        raise ValueError(
            f"Video must be 30 seconds or shorter. "
            f"Detected {duration:.2f} seconds."
        )

    if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
        source.release()
        raise ValueError(
            "Video must be 640×480 for the current ATLAS detector. "
            f"Received {width}×{height}."
        )

    ground_truth = (
        _load_ground_truth(ground_truth_path)
        if ground_truth_path
        else {}
    )

    # A fresh pipeline is used for every uploaded video so the live
    # dashboard's detector/Kalman state can never leak into this test.
    tracker = TrackingSystem(
        config=config,
        acceleration_noise=500.0,
        gain=4.0,
        velocity_scale=0.5,
    )

    total_frames = 0
    detected_frames = 0
    tracking_frames = 0
    first_detection_frame: int | None = None
    processing_times: list[float] = []
    errors: list[float] = []

    benchmark_start = time.perf_counter()

    try:
        while True:
            ok, frame = source.read()
            if not ok:
                break

            total_frames += 1

            process_start = time.perf_counter()
            result = tracker.process(frame)
            processing_times.append(
                time.perf_counter() - process_start
            )

            detected = bool(result["detected"])
            tracking = bool(result["tracking"])

            if detected:
                detected_frames += 1
                if first_detection_frame is None:
                    first_detection_frame = total_frames

            if tracking:
                tracking_frames += 1

            position = result.get("position")
            gt = ground_truth.get(total_frames)

            if position is not None and gt is not None:
                dx = position["x"] - gt["x"]
                dy = position["y"] - gt["y"]
                errors.append(math.hypot(dx, dy))
    finally:
        source.release()

    runtime = time.perf_counter() - benchmark_start

    if total_frames == 0:
        raise ValueError("The uploaded video contains no readable frames.")

    actual_duration = (
        total_frames / fps
        if fps > 0
        else 0.0
    )

    average_processing = (
        sum(processing_times) / len(processing_times)
        if processing_times
        else 0.0
    )

    detection_rate = detected_frames / total_frames * 100.0
    tracking_rate = tracking_frames / total_frames * 100.0

    result: dict[str, Any] = {
        "video": {
            "filename": Path(video_path).name,
            "width": width,
            "height": height,
            "fps": fps,
            "frames": total_frames,
            "duration_seconds": actual_duration,
        },
        "benchmark": {
            "benchmark_runtime_seconds": runtime,
            "measured_processing_fps": (
                1.0 / average_processing
                if average_processing > 0
                else 0.0
            ),
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
            "detection_rate_percent": detection_rate,
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
                math.sqrt(
                    sum(error * error for error in errors)
                    / len(errors)
                )
                if errors
                else None
            ),
            "ground_truth_available": bool(ground_truth),
        },
    }

    return {
        "status": "completed",
        "generated_at": time.time(),
        "result": result,
        "notes": [
            "This test processes the uploaded frames directly; it does not modify the existing simulator benchmark.",
            "True centroid error is calculated only when matching ground-truth data is supplied.",
        ],
    }


async def save_upload(upload_file, destination: str) -> int:
    total_bytes = 0

    with open(destination, "wb") as output:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break

            total_bytes += len(chunk)

            if total_bytes > MAX_VIDEO_SIZE_BYTES:
                raise ValueError(
                    "Video file is too large. Maximum upload size is 100 MB."
                )

            output.write(chunk)

    return total_bytes


async def run_uploaded_video(upload_file, config: dict[str, Any]) -> dict[str, Any]:
    suffix = Path(upload_file.filename or "").suffix.lower()

    if suffix != ".mp4":
        raise ValueError("Only MP4 video uploads are supported.")

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix="atlas_video_",
            suffix=".mp4",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name

        await save_upload(upload_file, temporary_path)

        # Importing here keeps the upload feature isolated from the normal
        # startup path and avoids changing current dashboard behavior.
        from starlette.concurrency import run_in_threadpool

        return await run_in_threadpool(
            _process_video,
            temporary_path,
            config,
            None,
        )
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
        await upload_file.close()
