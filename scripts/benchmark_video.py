import csv
import json
import math
import os
import sys
import time

from simulation.video_source import VideoFileSource
from tracking.system import TrackingSystem


def load_ground_truth(path):
    ground_truth = {}

    with open(path, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            frame = int(row["frame"])

            ground_truth[frame] = {
                "x": float(row["ground_truth_x"]),
                "y": float(row["ground_truth_y"]),
            }

    return ground_truth


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python -m scripts.benchmark_video <video.mp4>")
        return

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    # Load configuration
    with open("config/config.json", "r") as file:
        config = json.load(file)

    # Open video
    source = VideoFileSource(video_path)

    # Locate matching ground-truth CSV
    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    ground_truth_path = os.path.join(
        os.path.dirname(video_path),
        f"{video_name}_ground_truth.csv"
    )

    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(
            "Ground-truth CSV not found:\n"
            f"{ground_truth_path}\n\n"
            "Generate the synthetic video first using:\n"
            "python -m scripts.generate_synthetic_video"
        )

    ground_truth = load_ground_truth(
        ground_truth_path
    )

    # Use the existing tracking system unchanged
    tracking_system = TrackingSystem(
        config=config,
        acceleration_noise=500.0,
        gain=4.0,
        velocity_scale=0.5
    )

    output_dir = "outputs/benchmark_video"
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(
        output_dir,
        f"{video_name}_frames.csv"
    )

    json_path = os.path.join(
        output_dir,
        f"{video_name}_summary.json"
    )

    total_frames = 0
    detected_frames = 0
    tracking_frames = 0

    processing_times = []

    acquisition_time = None
    first_detection_frame = None

    errors = []

    frame_records = []

    benchmark_start = time.perf_counter()

    while True:
        frame = source.get_frame()

        if frame is None:
            break

        total_frames += 1

        # Measure ONLY tracking-system processing time.
        process_start = time.perf_counter()

        result = tracking_system.process(frame)
        

        process_time = (
            time.perf_counter() - process_start
        )

        processing_times.append(process_time)

        detected = bool(result["detected"])
        tracking = bool(result["tracking"])

        if detected:
            detected_frames += 1

            if first_detection_frame is None:
                first_detection_frame = total_frames

                acquisition_time = (
                    total_frames / source.fps
                )

        if tracking:
            tracking_frames += 1

        position = result.get("position")
        velocity = result.get("velocity")
        command = result.get("command")

        # Calculate centroiding error.
        error = None

        gt = ground_truth.get(total_frames)
        if (
            gt is not None
            and position is not None
        ):
            dx = position["x"] - gt["x"]
            dy = position["y"] - gt["y"]

            error = math.sqrt(
                dx * dx + dy * dy
            )

            errors.append(error)

        frame_records.append({
            "frame": total_frames,
            "timestamp_seconds": (
                (total_frames - 1) / source.fps
            ),
            "ground_truth_x": (
                gt["x"]
                if gt is not None
                else None
            ),
            "ground_truth_y": (
                gt["y"]
                if gt is not None
                else None
            ),
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
            "centroid_error": error,
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
            "processing_ms": process_time * 1000.0,
        })

    benchmark_elapsed = (
        time.perf_counter() - benchmark_start
    )

    source.release()

    # Calculate tracking rates
    if total_frames > 0:
        detection_rate = (
            detected_frames / total_frames * 100.0
        )

        tracking_rate = (
            tracking_frames / total_frames * 100.0
        )
    else:
        detection_rate = 0.0
        tracking_rate = 0.0

    # Processing metrics
    if processing_times:
        average_processing_ms = (
            sum(processing_times)
            / len(processing_times)
            * 1000.0
        )

        max_processing_ms = (
            max(processing_times)
            * 1000.0
        )

        measured_processing_fps = (
            1.0
            / (
                sum(processing_times)
                / len(processing_times)
            )
        )
    else:
        average_processing_ms = 0.0
        max_processing_ms = 0.0
        measured_processing_fps = 0.0

    # Accuracy metrics
    if errors:
        average_error = (
            sum(errors) / len(errors)
        )

        maximum_error = max(errors)

        rmse = math.sqrt(
            sum(error * error for error in errors)
            / len(errors)
        )
    else:
        average_error = None
        maximum_error = None
        rmse = None

    video_duration = (
        total_frames / source.fps
        if source.fps > 0
        else 0.0
    )

    lock_retention = tracking_rate

    target_loss = (
        100.0 - lock_retention
    )

    summary = {
        "video": {
            "path": video_path,
            "ground_truth_path": ground_truth_path,
            "width": source.width,
            "height": source.height,
            "fps": source.fps,
            "frames": total_frames,
            "duration_seconds": video_duration,
        },
        "benchmark": {
            "benchmark_runtime_seconds": benchmark_elapsed,
            "measured_processing_fps": measured_processing_fps,
            "average_processing_ms": average_processing_ms,
            "max_processing_ms": max_processing_ms,
        },
        "tracking": {
            "detected_frames": detected_frames,
            "tracking_frames": tracking_frames,
            "detection_rate_percent": detection_rate,
            "lock_retention_percent": lock_retention,
            "target_loss_percent": target_loss,
            "first_detection_frame": first_detection_frame,
            "acquisition_time_seconds": acquisition_time,
        },
        "accuracy": {
            "frames_with_error": len(errors),
            "average_centroid_error_pixels": average_error,
            "maximum_centroid_error_pixels": maximum_error,
            "rmse_pixels": rmse,
        },
    }

    # Write frame-level CSV
    with open(
        csv_path,
        "w",
        newline=""
    ) as file:

        fieldnames = (
            frame_records[0].keys()
            if frame_records
            else [
                "frame",
                "timestamp_seconds",
                "ground_truth_x",
                "ground_truth_y",
                "detected",
                "tracking",
                "x",
                "y",
                "centroid_error",
                "velocity_x",
                "velocity_y",
                "pan_speed",
                "tilt_speed",
                "processing_ms",
            ]
        )

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(frame_records)

    # Write summary JSON
    with open(
        json_path,
        "w"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )

    print()
    print("========================================")
    print("       ATLAS MP4 BENCHMARK")
    print("========================================")
    print()
    print(f"Video frames       : {total_frames}")
    print(f"Video FPS          : {source.fps:.2f}")
    print()
    print("PERFORMANCE")
    print("----------------------------------------")
    print(
        f"Processing FPS     : "
        f"{measured_processing_fps:.2f}"
    )
    print(
        f"Avg processing     : "
        f"{average_processing_ms:.2f} ms"
    )
    print(
        f"Max processing     : "
        f"{max_processing_ms:.2f} ms"
    )
    print()
    print("TRACKING")
    print("----------------------------------------")
    print(
        f"Acquisition time   : "
        f"{acquisition_time:.3f} sec"
        if acquisition_time is not None
        else "Acquisition time   : N/A"
    )
    print(
        f"Detection rate     : "
        f"{detection_rate:.2f}%"
    )
    print(
        f"Lock retention     : "
        f"{lock_retention:.2f}%"
    )
    print(
        f"Target loss        : "
        f"{target_loss:.2f}%"
    )
    print()
    print("ACCURACY")
    print("----------------------------------------")

    if average_error is not None:
        print(
            f"Average error      : "
            f"{average_error:.3f} px"
        )
        print(
            f"Maximum error      : "
            f"{maximum_error:.3f} px"
        )
        print(
            f"RMSE               : "
            f"{rmse:.3f} px"
        )
    else:
        print("No valid ground-truth errors calculated.")

    print()
    print("FILES")
    print("----------------------------------------")
    print(f"JSON : {json_path}")
    print(f"CSV  : {csv_path}")
    print()
    print("========================================")
    print("       BENCHMARK COMPLETE")
    print("========================================")
    print()


if __name__ == "__main__":
    main()