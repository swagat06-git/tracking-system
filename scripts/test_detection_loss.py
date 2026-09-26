import math
import json

from simulation.simulator import Simulator
from tracking.pipeline import TrackingPipeline


def main():

    with open("config/config.json", "r") as file:
        config = json.load(file)

    simulator = Simulator(config)
    pipeline = TrackingPipeline(max_missed_frames=10)

    total_frames = 60

    # Frames during which we intentionally hide the target
    loss_start = 15
    loss_end = 20

    tracking_errors = []

    detected_frames = 0
    predicted_frames = 0
    tracking_lost_frames = 0

    camera_width = config["camera"]["width"]
    camera_height = config["camera"]["height"]

    print("Detection Loss Test")
    print("=" * 70)

    for frame_number in range(total_frames):

        simulator.update()

        frame = simulator.get_frame()

        ground_truth = simulator.get_ground_truth()

        # -------------------------------------------------
        # Check whether target is inside camera view
        # -------------------------------------------------

        target_x = ground_truth["x"]
        target_y = ground_truth["y"]

        inside_camera = (
            0 <= target_x < camera_width
            and
            0 <= target_y < camera_height
        )

        # -------------------------------------------------
        # Simulate temporary detection loss
        # -------------------------------------------------

        if loss_start <= frame_number < loss_end:
            frame = None

        result = pipeline.process(frame)

        if result["detected"]:
            detected_frames += 1

        if (
            not result["detected"]
            and result["tracking"]
        ):
            predicted_frames += 1

        # Count tracking loss only when the target should
        # actually be visible in the camera.
        if (
            not result["tracking"]
            and inside_camera
        ):
            tracking_lost_frames += 1

        # -------------------------------------------------
        # Evaluate valid estimates
        # -------------------------------------------------

        if result["position"] is not None:

            error = math.sqrt(
                (result["position"]["x"] - ground_truth["x"]) ** 2
                +
                (result["position"]["y"] - ground_truth["y"]) ** 2
            )

            tracking_errors.append(error)

        if frame_number in (
            loss_start - 1,
            loss_start,
            loss_end - 1,
            loss_end
        ):
            print(
                f"Frame {frame_number:02d} | "
                f"Detected: {result['detected']} | "
                f"Tracking: {result['tracking']} | "
                f"Position: {result['position']}"
            )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"Total frames             : {total_frames}")
    print(f"Detected frames          : {detected_frames}")
    print(f"Predicted frames         : {predicted_frames}")
    print(f"Tracking lost frames     : {tracking_lost_frames}")

    if tracking_errors:
        mean_error = sum(tracking_errors) / len(tracking_errors)
        max_error = max(tracking_errors)

        print(
            f"Mean tracking error     : "
            f"{mean_error:.2f} pixels"
        )

        print(
            f"Maximum tracking error  : "
            f"{max_error:.2f} pixels"
        )

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    print("\n" + "=" * 70)

    temporary_prediction_pass = predicted_frames >= (
        loss_end - loss_start
    )

    recovery_pass = (
        tracking_lost_frames == 0
        and detected_frames > 0
    )

    if temporary_prediction_pass and recovery_pass:
        print(
            "PASS: Tracker successfully handled "
            "temporary detection loss."
        )
    else:
        print(
            "FAIL: Tracker did not handle "
            "temporary detection loss correctly."
        )


if __name__ == "__main__":
    main()