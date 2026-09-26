import json
import math
import argparse

from simulation.simulator import Simulator
from tracking.pipeline import TrackingPipeline
from control.controller import CameraController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gain", type=float, default=5.0)
    args = parser.parse_args()

    # -------------------------------------------------
    # Load configuration
    # -------------------------------------------------

    with open("config/config.json", "r") as file:
        config = json.load(file)

    # -------------------------------------------------
    # Create simulator
    # -------------------------------------------------

    simulator = Simulator(config)

    # Use a target speed that the camera can realistically
    # follow with the configured control limits.
    simulator.target.set_velocity(2.0, 1.0)

    pipeline = TrackingPipeline()

    controller = CameraController(
        frame_width=config["camera"]["width"],
        frame_height=config["camera"]["height"],
        max_pan_speed=config["control"]["max_pan_speed"],
        max_tilt_speed=config["control"]["max_tilt_speed"],
        gain=args.gain
    )

    total_frames = 180

    center_x = config["camera"]["width"] / 2
    center_y = config["camera"]["height"] / 2

    center_errors = []

    detection_failures = 0
    out_of_view_frames = 0

    # -------------------------------------------------
    # Test header
    # -------------------------------------------------

    print("Closed-Loop Camera Tracking Test")
    print("=" * 70)

    print(
        f"Target velocity        : "
        f"({simulator.target.vx:.1f}, "
        f"{simulator.target.vy:.1f}) px/s"
    )

    print(
        f"Camera max speed       : "
        f"({config['control']['max_pan_speed']:.1f}, "
        f"{config['control']['max_tilt_speed']:.1f}) px/s"
    )

    # -------------------------------------------------
    # Closed-loop simulation
    # -------------------------------------------------

    for frame_number in range(total_frames):

        # Move target
        simulator.update()

        # Capture current camera frame
        frame = simulator.get_frame()

        # Ground truth in image coordinates
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
        # -------------------------------------------------
        # Visibility
        # -------------------------------------------------

        # Ground truth determines whether the target is
        # physically inside the camera view.
        if not inside_camera:
            out_of_view_frames += 1

        
        # -------------------------------------------------
        # Detection + Kalman tracking
        # -------------------------------------------------

        result = pipeline.process(frame)

        if not result["detected"]:

            if inside_camera:
                detection_failures += 1
            else:
                out_of_view_frames += 1

            continue

        # -------------------------------------------------
        # Controller
        # -------------------------------------------------

        estimated = result["position"]

        command = controller.compute_command(
            estimated["x"],
            estimated["y"]
        )

        # -------------------------------------------------
        # Move camera
        # -------------------------------------------------

        simulator.move_camera(
            command["pan_speed"],
            command["tilt_speed"]
        )

        # -------------------------------------------------
        # Measure target distance from camera center
        # -------------------------------------------------

        new_ground_truth = simulator.get_ground_truth()

        error_x = new_ground_truth["x"] - center_x
        error_y = new_ground_truth["y"] - center_y

        center_error = math.sqrt(
            error_x ** 2 +
            error_y ** 2
        )

        center_errors.append(center_error)

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        if frame_number % 20 == 0:

            camera_x, camera_y = simulator.camera.get_position()

            print(
                f"Frame {frame_number:03d} | "
                f"Camera: ({camera_x:.1f}, {camera_y:.1f}) | "
                f"Target: "
                f"({new_ground_truth['x']:.1f}, "
                f"{new_ground_truth['y']:.1f}) | "
                f"Center Error: {center_error:.2f}px"
            )

    # -------------------------------------------------
    # Calculate statistics
    # -------------------------------------------------

    if center_errors:
        mean_center_error = (
            sum(center_errors) / len(center_errors)
        )
        max_center_error = max(center_errors)

        first_error = center_errors[0]
        last_error = center_errors[-1]

    else:
        mean_center_error = float("inf")
        max_center_error = float("inf")
        first_error = float("inf")
        last_error = float("inf")

    # -------------------------------------------------
    # Results
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"Total frames               : {total_frames}")
    print(f"Tracked frames             : {len(center_errors)}")
    print(f"Out-of-view frames         : {out_of_view_frames}")
    print(f"Detection failures         : {detection_failures}")

    print(
        f"Initial center error       : "
        f"{first_error:.2f} pixels"
    )

    print(
        f"Final center error         : "
        f"{last_error:.2f} pixels"
    )

    print(
        f"Mean camera-center error   : "
        f"{mean_center_error:.2f} pixels"
    )

    print(
        f"Maximum center error       : "
        f"{max_center_error:.2f} pixels"
    )

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    # The target is intentionally slow enough for the
    # camera to follow. We therefore expect:
    #
    # 1. No detection failures while visible.
    # 2. The target should remain visible.
    # 3. Final error should be reasonably controlled.

    max_allowed_mean_error = 100.0
    max_allowed_final_error = 100.0

    detection_pass = detection_failures == 0
    visibility_pass = out_of_view_frames == 0
    mean_error_pass = mean_center_error <= max_allowed_mean_error
    final_error_pass = last_error <= max_allowed_final_error

    print("\n" + "=" * 70)

    if (
        detection_pass
        and visibility_pass
        and mean_error_pass
        and final_error_pass
    ):
        print(
            "PASS: Closed-loop camera tracking "
            "meets baseline requirements."
        )
    else:
        print(
            "FAIL: Closed-loop camera tracking "
            "does not meet baseline requirements."
        )

        if not detection_pass:
            print(
                f"- {detection_failures} detection "
                f"failures occurred."
            )

        if not visibility_pass:
            print(
                f"- Target left the camera view for "
                f"{out_of_view_frames} frames."
            )

        if not mean_error_pass:
            print(
                f"- Mean center error exceeds "
                f"{max_allowed_mean_error:.1f} pixels."
            )

        if not final_error_pass:
            print(
                f"- Final center error exceeds "
                f"{max_allowed_final_error:.1f} pixels."
            )


if __name__ == "__main__":
    main()