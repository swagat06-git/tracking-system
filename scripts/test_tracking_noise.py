import math
import json
import random

from simulation.simulator import Simulator
from tracking.pipeline import TrackingPipeline


def main():

    # -----------------------------------------
    # Load configuration
    # -----------------------------------------

    with open("config/config.json", "r") as file:
        config = json.load(file)

    # -----------------------------------------
    # Create simulator and tracking pipeline
    # -----------------------------------------

    simulator = Simulator(config)
    pipeline = TrackingPipeline()

    total_frames = 60

    # Artificial detector measurement noise
    noise_std = 5.0

    errors = []

    detected_frames = 0
    out_of_view_frames = 0
    detection_failure_frames = 0

    # -----------------------------------------
    # Evaluation
    # -----------------------------------------

    print("Tracking Noise Test")
    print("=" * 60)

    print(
        f"Measurement noise std : "
        f"{noise_std:.1f} pixels"
    )

    for frame_number in range(total_frames):

        # Advance simulator
        simulator.update()

        # Get camera frame
        frame = simulator.get_frame()

        # Get ground truth
        ground_truth = simulator.get_ground_truth()

        # -----------------------------------------
        # Check whether target is inside camera
        # -----------------------------------------

        target_x = ground_truth["x"]
        target_y = ground_truth["y"]

        camera_width = config["camera"]["width"]
        camera_height = config["camera"]["height"]

        inside_camera = (
            0 <= target_x < camera_width
            and
            0 <= target_y < camera_height
        )

        # -----------------------------------------
        # Detect target
        # -----------------------------------------

        detection = pipeline.detector.detect(frame)

        if not detection["detected"]:

            if inside_camera:
                detection_failure_frames += 1
            else:
                out_of_view_frames += 1

            continue

        detected_frames += 1

        # -----------------------------------------
        # Add artificial measurement noise
        # -----------------------------------------

        noisy_x = (
            detection["x"]
            + random.gauss(0, noise_std)
        )

        noisy_y = (
            detection["y"]
            + random.gauss(0, noise_std)
        )

        # -----------------------------------------
        # Kalman prediction + correction
        # -----------------------------------------

        pipeline.tracker.predict()

        estimate = pipeline.tracker.update(
            noisy_x,
            noisy_y
        )

        # -----------------------------------------
        # Kalman tracking error
        # -----------------------------------------

        error = math.sqrt(
            (estimate["x"] - target_x) ** 2
            +
            (estimate["y"] - target_y) ** 2
        )

        errors.append(error)

        # -----------------------------------------
        # Print progress
        # -----------------------------------------

        if frame_number % 10 == 0:

            print(
                f"Frame {frame_number:02d} | "
                f"Truth: "
                f"({target_x:.1f}, {target_y:.1f}) | "
                f"Estimate: "
                f"({estimate['x']:.1f}, "
                f"{estimate['y']:.1f}) | "
                f"Error: {error:.2f}px"
            )

    # -----------------------------------------
    # Statistics
    # -----------------------------------------

    if errors:
        mean_error = sum(errors) / len(errors)
        max_error = max(errors)
    else:
        mean_error = float("inf")
        max_error = float("inf")

    # -----------------------------------------
    # Results
    # -----------------------------------------

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        f"Total frames           : "
        f"{total_frames}"
    )

    print(
        f"Detected frames        : "
        f"{detected_frames}"
    )

    print(
        f"Out-of-view frames     : "
        f"{out_of_view_frames}"
    )

    print(
        f"Detection failures     : "
        f"{detection_failure_frames}"
    )

    print(
        f"Kalman estimate error  : "
        f"{mean_error:.2f} pixels"
    )

    print(
        f"Maximum Kalman error   : "
        f"{max_error:.2f} pixels"
    )

    # -----------------------------------------
    # Validation
    # -----------------------------------------

    max_mean_error = 15.0
    max_max_error = 30.0

    mean_pass = mean_error <= max_mean_error
    maximum_pass = max_error <= max_max_error
    detection_pass = detection_failure_frames == 0

    print("\n" + "=" * 60)

    if mean_pass and maximum_pass and detection_pass:

        print(
            "PASS: Tracking remains stable "
            "under measurement noise."
        )

    else:

        print(
            "FAIL: Tracking performance under "
            "measurement noise does not meet "
            "baseline requirements."
        )

        if not mean_pass:
            print(
                f"- Mean error exceeds "
                f"{max_mean_error:.1f} pixels."
            )

        if not maximum_pass:
            print(
                f"- Maximum error exceeds "
                f"{max_max_error:.1f} pixels."
            )

        if not detection_pass:
            print(
                f"- {detection_failure_frames} "
                f"detection failures occurred."
            )


if __name__ == "__main__":
    main()