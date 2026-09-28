import csv
import json
import os
import sys
import time

from simulation.video_source import VideoFileSource
from tracking.system import TrackingSystem


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

    # Use the existing tracking system unchanged
    tracking_system = TrackingSystem(
        config=config,
        acceleration_noise=500.0,
        gain=4.0,
        velocity_scale=0.5
    )

    output_dir = "outputs/benchmark_video"
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

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

        frame_records.append({
            "frame": total_frames,
            "timestamp_seconds": (
                (total_frames - 1) / source.fps
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

    # Calculate metrics
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
        "limitations": [
            "This MP4 benchmark does not have independent "
            "ground-truth target coordinates.",
            "Centroid error and RMSE therefore cannot be "
            "computed from this video alone.",
            "The simulator benchmark remains the source "
            "of ground-truth error measurements."
        ]
    }

    # Write frame-level CSV
    with open(
        csv_path,
        "w",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=frame_records[0].keys()
            if frame_records
            else [
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
            ]
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
    print(f"Video:              {video_path}")
    print(
        f"Resolution:         "
        f"{source.width}x{source.height}"
    )
    print(f"Video FPS:          {source.fps:.2f}")
    print(f"Frames processed:   {total_frames}")
    print(f"Duration:           {video_duration:.2f}s")
    print()
    print(
        f"Detection rate:     "
        f"{detection_rate:.2f}%"
    )
    print(
        f"Lock retention:     "
        f"{lock_retention:.2f}%"
    )
    print(
        f"Target loss:        "
        f"{target_loss:.2f}%"
    )
    print()
    print(
        f"Acquisition time:   "
        f"{acquisition_time:.3f}s"
        if acquisition_time is not None
        else "Acquisition time:   Not detected"
    )
    print()
    print(
        f"Processing FPS:     "
        f"{measured_processing_fps:.2f}"
    )
    print(
        f"Avg processing:     "
        f"{average_processing_ms:.3f} ms"
    )
    print(
        f"Max processing:     "
        f"{max_processing_ms:.3f} ms"
    )
    print()
    print(f"Frame log:          {csv_path}")
    print(f"Summary:            {json_path}")
    print()
    print("Ground-truth error: NOT AVAILABLE")
    print("========================================")


if __name__ == "__main__":
    main()