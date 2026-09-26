import json
import numpy as np
import matplotlib.pyplot as plt

from simulation.simulator import Simulator
from tracking.system import TrackingSystem


def run_evaluation(
    config,
    num_frames=100,
    acceleration_noise=500.0,
    gain=4.0,
    velocity_scale=0.5,
    motion_type=None
):
    """
    Run the complete closed-loop tracking system.

    High-level pipeline:

        Simulator
            ↓
        TrackingSystem
            ↓
        ML Detector
            ↓
        Kalman Filter
            ↓
        Controller
            ↓
        Camera Movement
    """

    simulator = Simulator(config)

    if motion_type is not None:
        simulator.motion.motion_type = motion_type

    tracking_system = TrackingSystem(
        config=config,
        acceleration_noise=acceleration_noise,
        gain=gain,
        velocity_scale=velocity_scale
    )

    detector_errors = []
    kalman_errors = []
    frame_numbers = []

    valid_frames = 0
    out_of_view = 0
    failures = 0

    camera_width = config["camera"]["width"]
    camera_height = config["camera"]["height"]

    center_x = camera_width / 2.0
    center_y = camera_height / 2.0

    # --------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------

    for frame_number in range(num_frames):

        simulator.update()

        frame = simulator.get_frame()

        ground_truth = simulator.get_ground_truth()

        gt_x = ground_truth["x"]
        gt_y = ground_truth["y"]

        # --------------------------------------------------
        # Check whether target is inside camera view
        # --------------------------------------------------

        if (
            gt_x < 0
            or gt_x >= camera_width
            or gt_y < 0
            or gt_y >= camera_height
        ):
            out_of_view += 1
            continue

        # --------------------------------------------------
        # TRACKING SYSTEM
        # --------------------------------------------------

        result = tracking_system.process(frame)

        if not result["detected"]:
            failures += 1
            continue

        detector_position = result["detector_position"]
        kalman_position = result["position"]

        if detector_position is None or kalman_position is None:
            failures += 1
            continue

        predicted_x = detector_position["x"]
        predicted_y = detector_position["y"]

        filtered_x = kalman_position["x"]
        filtered_y = kalman_position["y"]

        # --------------------------------------------------
        # ML DETECTOR ERROR
        # --------------------------------------------------

        detector_error = np.sqrt(
            (predicted_x - gt_x) ** 2
            +
            (predicted_y - gt_y) ** 2
        )

        detector_errors.append(float(detector_error))

        # --------------------------------------------------
        # KALMAN FILTER ERROR
        # --------------------------------------------------

        kalman_error = np.sqrt(
            (filtered_x - gt_x) ** 2
            +
            (filtered_y - gt_y) ** 2
        )

        kalman_errors.append(float(kalman_error))

        frame_numbers.append(frame_number)

        valid_frames += 1

        # --------------------------------------------------
        # CAMERA MOVEMENT
        # --------------------------------------------------

        command = result["command"]

        simulator.move_camera(
            command["pan_speed"],
            command["tilt_speed"]
        )

    # --------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------

    results = {
        "total_frames": num_frames,
        "valid_frames": valid_frames,
        "out_of_view": out_of_view,
        "failures": failures,

        "mean_detector_error": (
            float(np.mean(detector_errors))
            if detector_errors
            else None
        ),

        "max_detector_error": (
            float(np.max(detector_errors))
            if detector_errors
            else None
        ),

        "min_detector_error": (
            float(np.min(detector_errors))
            if detector_errors
            else None
        ),

        "mean_kalman_error": (
            float(np.mean(kalman_errors))
            if kalman_errors
            else None
        ),

        "max_kalman_error": (
            float(np.max(kalman_errors))
            if kalman_errors
            else None
        ),

        "min_kalman_error": (
            float(np.min(kalman_errors))
            if kalman_errors
            else None
        ),

        "frame_numbers": frame_numbers,
        "detector_errors": detector_errors,
        "kalman_errors": kalman_errors
    }

    return results


def print_results(
    results,
    config,
    acceleration_noise,
    gain,
    velocity_scale,
    motion_type
):
    """
    Print evaluation results in a clean format.
    """

    disturbance = config["disturbance"]

    print("\n" + "=" * 64)
    print("DAY 6 — END-TO-END TRACKING EVALUATION")
    print("=" * 64)

    print("\nSYSTEM")
    print("-" * 64)
    print(f"Motion type         : {motion_type}")
    print("Pipeline            : Simulator → TrackingSystem")
    print(f"Frames              : {results['total_frames']}")

    print("\nDISTURBANCE")
    print("-" * 64)
    print(f"Noise type          : {disturbance['noise_type']}")
    print(f"Noise level         : {disturbance['noise_level']}")
    print(f"Camera jitter       : ±{disturbance['camera_jitter']} px")

    print("\nPARAMETERS")
    print("-" * 64)
    print(f"Kalman acceleration : {acceleration_noise}")
    print(f"Controller gain     : {gain}")
    print(f"Velocity scale      : {velocity_scale}")

    print("\nTRACKING STATUS")
    print("-" * 64)
    print(f"Valid frames        : {results['valid_frames']}")
    print(f"Out of view         : {results['out_of_view']}")
    print(f"Detection failures  : {results['failures']}")

    print("\nML DETECTOR")
    print("-" * 64)

    if results["mean_detector_error"] is not None:
        print(
            f"Mean error          : "
            f"{results['mean_detector_error']:.2f} px"
        )

        print(
            f"Maximum error       : "
            f"{results['max_detector_error']:.2f} px"
        )

        print(
            f"Minimum error       : "
            f"{results['min_detector_error']:.2f} px"
        )

    print("\nKALMAN FILTER")
    print("-" * 64)

    if results["mean_kalman_error"] is not None:
        print(
            f"Mean error          : "
            f"{results['mean_kalman_error']:.2f} px"
        )

        print(
            f"Maximum error       : "
            f"{results['max_kalman_error']:.2f} px"
        )

        print(
            f"Minimum error       : "
            f"{results['min_kalman_error']:.2f} px"
        )

    print("\n" + "=" * 64)

    if (
        results["valid_frames"] == results["total_frames"]
        and results["out_of_view"] == 0
        and results["failures"] == 0
    ):
        print("STATUS              : PASS")
    else:
        print("STATUS              : ATTENTION")

    print("=" * 64)


def plot_tracking_error(results):
    """
    Plot detector and Kalman tracking error over time.
    """

    plt.figure(figsize=(10, 6))

    plt.plot(
        results["frame_numbers"],
        results["detector_errors"],
        label="ML Detector Error"
    )

    plt.plot(
        results["frame_numbers"],
        results["kalman_errors"],
        label="Kalman Filter Error"
    )

    plt.xlabel("Frame Number")
    plt.ylabel("Position Error (pixels)")

    plt.title("Tracking Error Over Time")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "tracking_error_over_time.png",
        dpi=200
    )

    plt.show()


def main():

    # --------------------------------------------------
    # Load configuration
    # --------------------------------------------------

    with open("config/config.json", "r") as file:
        config = json.load(file)

    # --------------------------------------------------
    # Evaluation parameters
    # --------------------------------------------------

    num_frames = 100

    acceleration_noise = 500.0
    gain = 4.0
    velocity_scale = 0.5

    motion_type = "linear"

    # --------------------------------------------------
    # Run evaluation
    # --------------------------------------------------

    results = run_evaluation(
        config=config,
        num_frames=num_frames,
        acceleration_noise=acceleration_noise,
        gain=gain,
        velocity_scale=velocity_scale,
        motion_type=motion_type
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print_results(
        results,
        config,
        acceleration_noise,
        gain,
        velocity_scale,
        motion_type
    )

    # --------------------------------------------------
    # Plot tracking error
    # --------------------------------------------------

    plot_tracking_error(results)


if __name__ == "__main__":
    main()