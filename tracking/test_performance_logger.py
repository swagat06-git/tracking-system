from tracking.performance_logger import PerformanceLogger


def main():
    logger = PerformanceLogger(
        output_dir="outputs/test_logger"
    )

    logger.start()

    for frame in range(1, 31):
        result = {
            "detected": True,
            "tracking": True,
            "position": {
                "x": 320.0 + frame,
                "y": 240.0 + frame,
            },
            "velocity": {
                "vx": 30.0,
                "vy": 30.0,
            },
            "command": {
                "pan_speed": 2.0,
                "tilt_speed": 2.0,
            },
        }

        ground_truth = {
            "x": 320.0 + frame + 2.0,
            "y": 240.0 + frame + 2.0,
        }

        logger.record_frame(
            frame_number=frame,
            timestamp_seconds=(frame - 1) / 30.0,
            result=result,
            processing_time_seconds=0.004,
            ground_truth=ground_truth,
        )

    result = logger.save(
        "logger_test",
        video_fps=30.0
    )

    print("\nPerformance logger test")
    print("-----------------------")
    print(f"CSV:  {result['csv']}")
    print(f"JSON: {result['json']}")
    print()

    summary = result["summary"]

    print(
        "Average error:",
        summary["error"]["average_error_pixels"]
    )

    print(
        "RMSE:",
        summary["error"]["rmse_pixels"]
    )

    print(
        "Processing FPS:",
        summary["processing"]["processing_fps"]
    )


if __name__ == "__main__":
    main()