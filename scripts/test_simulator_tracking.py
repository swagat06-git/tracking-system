import os
import sys
import json
import math

# Project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)

from simulation.simulator import Simulator
from tracking.pipeline import TrackingPipeline


def main():

    # -----------------------------
    # Load configuration
    # -----------------------------

    config_path = os.path.join(
        PROJECT_ROOT,
        "config",
        "config.json"
    )

    with open(config_path, "r") as file:
        config = json.load(file)

    # -----------------------------
    # Create simulator + pipeline
    # -----------------------------

    simulator = Simulator(config)
    pipeline = TrackingPipeline()

    camera_x, camera_y = simulator.camera.get_position()

    camera_width = config["camera"]["width"]
    camera_height = config["camera"]["height"]

    frames = 60
    errors = []

    detected_frames = 0
    out_of_view_frames = 0

    print("Simulator -> Tracking Pipeline Integration Test")
    print("=" * 70)

    # -----------------------------
    # Integration loop
    # -----------------------------

    for frame_number in range(frames):

        # Advance simulator
        simulator.update()

        # Get camera frame
        frame = simulator.get_frame()

        # Get world-coordinate ground truth
        truth = simulator.get_ground_truth()

        # Convert world coordinates to camera coordinates
        truth_x = truth["x"] - camera_x
        truth_y = truth["y"] - camera_y

        # Check whether target should be inside camera view
        target_visible = (
            0 <= truth_x < camera_width
            and
            0 <= truth_y < camera_height
        )

        # Process frame through tracking pipeline
        result = pipeline.process(frame)

        if not result["detected"]:

            if target_visible:
                print(
                    f"Frame {frame_number:02d} | "
                    "Detection FAILED while target was visible"
                )
            else:
                print(
                    f"Frame {frame_number:02d} | "
                    "Target out of camera view"
                )

            out_of_view_frames += 1
            continue

        detected_frames += 1

        # Calculate tracking error
        estimated = result["position"]

        error = math.sqrt(
            (estimated["x"] - truth_x) ** 2
            + (estimated["y"] - truth_y) ** 2
        )

        errors.append(error)

        if frame_number % 10 == 0:
            print(
                f"Frame {frame_number:02d} | "
                f"Truth: ({truth_x:.1f}, {truth_y:.1f}) | "
                f"Estimate: "
                f"({estimated['x']:.1f}, "
                f"{estimated['y']:.1f}) | "
                f"Error: {error:.2f}px"
            )

    # -----------------------------
    # Results
    # -----------------------------

    print("\n" + "=" * 70)
    print("INTEGRATION RESULTS")
    print("=" * 70)

    print(
        f"Detected frames      : "
        f"{detected_frames}/{frames}"
    )

    print(
        f"Out-of-view frames   : "
        f"{out_of_view_frames}/{frames}"
    )

    if errors:

        mean_error = sum(errors) / len(errors)
        max_error = max(errors)

        print(
            f"Mean position error  : "
            f"{mean_error:.2f} pixels"
        )

        print(
            f"Maximum position error: "
            f"{max_error:.2f} pixels"
        )

    # -----------------------------
    # Validation
    # -----------------------------

    if detected_frames > 0 and errors:

        mean_error = sum(errors) / len(errors)

        if mean_error <= 15.0:
            print(
                "\nPASS: Simulator and tracking pipeline "
                "integrated successfully."
            )
            print(
                "Simulator -> Camera -> Detector -> Kalman"
            )
        else:
            print(
                "\nFAIL: Tracking error is too high."
            )

    else:
        print(
            "\nFAIL: No valid detections."
        )


if __name__ == "__main__":
    main()