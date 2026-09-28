import csv
import json
import math
import os
import time


class PerformanceLogger:
    def __init__(self, output_dir="outputs/performance"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.records = []
        self.processing_times = []

        self.total_frames = 0
        self.detected_frames = 0
        self.tracking_frames = 0

        self.first_detection_frame = None
        self.acquisition_time_seconds = None

        self.error_values = []

        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def record_frame(
        self,
        frame_number,
        timestamp_seconds,
        result,
        processing_time_seconds,
        ground_truth=None,
    ):
        self.total_frames += 1

        detected = bool(result.get("detected", False))
        tracking = bool(result.get("tracking", False))

        if detected:
            self.detected_frames += 1

            if self.first_detection_frame is None:
                self.first_detection_frame = frame_number

                self.acquisition_time_seconds = (
                    timestamp_seconds
                )

        if tracking:
            self.tracking_frames += 1

        processing_ms = processing_time_seconds * 1000.0
        self.processing_times.append(
            processing_time_seconds
        )

        position = result.get("position")
        velocity = result.get("velocity")
        command = result.get("command")

        error_pixels = None

        if (
            ground_truth is not None
            and position is not None
        ):
            dx = (
                position["x"]
                - ground_truth["x"]
            )
            dy = (
                position["y"]
                - ground_truth["y"]
            )

            error_pixels = math.sqrt(
                dx * dx + dy * dy
            )

            self.error_values.append(
                error_pixels
            )

        self.records.append({
            "frame": frame_number,
            "timestamp_seconds": timestamp_seconds,
            "detected": detected,
            "tracking": tracking,
            "x": (
                position["x"]
                if position is not None
                else None
            ),
            "y": (
                position["y"]
                if position is not None
                else None
            ),
            "velocity_x": (
                velocity["vx"]
                if velocity is not None
                else None
            ),
            "velocity_y": (
                velocity["vy"]
                if velocity is not None
                else None
            ),
            "pan_speed": (
                command["pan_speed"]
                if command is not None
                else None
            ),
            "tilt_speed": (
                command["tilt_speed"]
                if command is not None
                else None
            ),
            "processing_ms": processing_ms,
            "ground_truth_x": (
                ground_truth["x"]
                if ground_truth is not None
                else None
            ),
            "ground_truth_y": (
                ground_truth["y"]
                if ground_truth is not None
                else None
            ),
            "error_pixels": error_pixels,
        })

    def finish(self):
        self.end_time = time.perf_counter()

    def summary(self, video_fps=None):
        if self.total_frames == 0:
            return {}

        if self.start_time is not None and self.end_time is not None:
            runtime_seconds = (
                self.end_time - self.start_time
            )
        else:
            runtime_seconds = 0.0

        average_processing_ms = 0.0
        max_processing_ms = 0.0
        processing_fps = 0.0

        if self.processing_times:
            average_processing_seconds = (
                sum(self.processing_times)
                / len(self.processing_times)
            )

            average_processing_ms = (
                average_processing_seconds * 1000.0
            )

            max_processing_ms = (
                max(self.processing_times)
                * 1000.0
            )

            if average_processing_seconds > 0:
                processing_fps = (
                    1.0
                    / average_processing_seconds
                )

        detection_rate = (
            self.detected_frames
            / self.total_frames
            * 100.0
        )

        lock_retention = (
            self.tracking_frames
            / self.total_frames
            * 100.0
        )

        target_loss = 100.0 - lock_retention

        average_error = None
        max_error = None
        rmse = None

        if self.error_values:
            average_error = (
                sum(self.error_values)
                / len(self.error_values)
            )

            max_error = max(self.error_values)

            rmse = math.sqrt(
                sum(
                    error * error
                    for error in self.error_values
                )
                / len(self.error_values)
            )

        duration_seconds = None

        if self.records and video_fps:
            duration_seconds = (
                self.total_frames
                / video_fps
            )
        elif self.records:
            duration_seconds = (
                self.records[-1]["timestamp_seconds"]
                + 1.0 / 30.0
            )

        return {
            "duration_seconds": duration_seconds,
            "total_frames": self.total_frames,
            "video_fps": video_fps,

            "processing": {
                "processing_fps": processing_fps,
                "average_processing_ms": (
                    average_processing_ms
                ),
                "max_processing_ms": (
                    max_processing_ms
                ),
                "benchmark_runtime_seconds": (
                    runtime_seconds
                ),
            },

            "detection": {
                "detected_frames": (
                    self.detected_frames
                ),
                "detection_rate_percent": (
                    detection_rate
                ),
            },

            "tracking": {
                "tracking_frames": (
                    self.tracking_frames
                ),
                "lock_retention_percent": (
                    lock_retention
                ),
                "target_loss_percent": (
                    target_loss
                ),
            },

            "acquisition": {
                "first_detection_frame": (
                    self.first_detection_frame
                ),
                "acquisition_time_seconds": (
                    self.acquisition_time_seconds
                ),
            },

            "error": {
                "samples": len(self.error_values),
                "average_error_pixels": (
                    average_error
                ),
                "max_error_pixels": (
                    max_error
                ),
                "rmse_pixels": rmse,
            },
        }

    def save(self, name, video_fps=None):
        self.finish()

        summary = self.summary(video_fps)

        csv_path = os.path.join(
            self.output_dir,
            f"{name}_frames.csv"
        )

        json_path = os.path.join(
            self.output_dir,
            f"{name}_summary.json"
        )

        fieldnames = [
            "frame",
            "timestamp_seconds",
            "detected",
            "tracking",
            "x",
            "y",
            "velocity_x",
            "velocity_y",
            "pan_speed",
            "tilt_speed",
            "processing_ms",
            "ground_truth_x",
            "ground_truth_y",
            "error_pixels",
        ]

        with open(
            csv_path,
            "w",
            newline=""
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(self.records)

        with open(
            json_path,
            "w"
        ) as file:
            json.dump(
                summary,
                file,
                indent=2
            )

        return {
            "csv": csv_path,
            "json": json_path,
            "summary": summary,
        }