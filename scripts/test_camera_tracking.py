import math
import json

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

    total_frames = 120

    errors = []

    detected_frames = 0
    out_of_view_frames = 0
    detection_failure_frames = 0

    # -----------------------------------------
    # Evaluation
    # -----------------------------------------

    print("Camera + Tracking Integration Test")
    print("=" * 70)

    for frame_number in range(total_frames):

        # -----------------------------------------
        # Move target
        # -----------------------------------------

        simulator.update()

        # -----------------------------------------
        # Move camera
        #
        # Alternate pan/tilt directions so that
        # the camera position changes during the test.
        # -----------------------------------------

        if frame_number < 40:
            pan_speed = 5.0
            tilt_speed = 0.0

        elif frame_number < 80:
            pan_speed = 0.0
            tilt_speed = 5.0

        else:
            pan_speed = -5.0
            tilt_speed = -5.0

        simulator.move_camera(
            pan_speed,
            tilt_speed
        )

        # -----------------------------------------
        # Get camera frame and ground truth
        # -----------------------------------------

        frame = simulator.get_frame()
        ground_truth = simulator.get_ground_truth()

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
        # Run tracking pipeline
        # -----------------------------------------

        result = pipeline.process(frame)

        # -----------------------------------------
        # Handle detection failure
        # -----------------------------------------

        if not result["detected"]:

            if inside_camera:
                detection_failure_frames += 1
            else:
                out_of_view_frames += 1

            continue

        detected_frames += 1

        # -----------------------------------------
        # Calculate tracking error
        # -----------------------------------------

        estimated = result["position"]

        error = math.sqrt(
            (estimated["x"] - target_x) ** 2
            +
            (estimated["y"] - target_y) ** 2
        )

        errors.append(error)

        # -----------------------------------------
        # Print progress
        # -----------------------------------------

        if frame_number % 10 == 0:

            camera_x, camera_y = simulator.camera.get_position()

            print(
                f"Frame {frame_number:03d} | "
                f"Camera: "
                f"({camera_x:.1f}, {camera_y:.1f}) | "
                f"Truth: "
                f"({target_x:.1f}, {target_y:.1f}) | "
                f"Estimate: "
                f"({estimated['x']:.1f}, "
                f"{estimated['y']:.1f}) | "
                f"Error: {error:.2f}px"
            )

    # -----------------------------------------
    # Calculate statistics
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

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"Total frames           : {total_frames}")
    print(f"Detected frames        : {detected_frames}")
    print(f"Out-of-view frames     : {out_of_view_frames}")
    print(f"Detection failures     : {detection_failure_frames}")

    print(
        f"Mean position error    : "
        f"{mean_error:.2f} pixels"
    )

    print(
        f"Maximum position error : "
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

    print("\n" + "=" * 70)

    if mean_pass and maximum_pass and detection_pass:

        print(
            "PASS: Moving camera and tracking "
            "pipeline integrated successfully."
        )

    else:

        print(
            "FAIL: Camera + tracking integration "
            "does not meet baseline requirements."
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
                f"detection failures occurred "
                f"while target was visible."
            )


if __name__ == "__main__":
    main()