import json
import numpy as np

from simulation.simulator import Simulator
from tracking.ml_detector import MLDetector
from tracking.kalman import KalmanTracker
from control.controller import CameraController


def main():

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------

    with open("config/config.json", "r") as f:
        config = json.load(f)

    # ---------------------------------------------------------
    # Create simulator and detector
    # ---------------------------------------------------------

    sim = Simulator(config)
    detector = MLDetector(config)

    fps = config["camera"]["fps"]

    # ---------------------------------------------------------
    # Kalman filter
    # ---------------------------------------------------------

    kalman = KalmanTracker(
        dt=1.0 / fps,
        acceleration_noise=300.0
    )

    # ---------------------------------------------------------
    # Velocity-aware controller
    # ---------------------------------------------------------

    controller = CameraController(
        frame_width=config["camera"]["width"],
        frame_height=config["camera"]["height"],
        max_pan_speed=config["control"]["max_pan_speed"],
        max_tilt_speed=config["control"]["max_tilt_speed"],
        gain=2.0,
        velocity_scale=0.5
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    detector_errors = []
    kalman_errors = []

    valid_frames = 0
    out_of_view = 0
    failures = 0

    # ---------------------------------------------------------
    # Main closed-loop simulation
    # ---------------------------------------------------------

    for frame_idx in range(50):

        # Update target motion
        sim.update()

        # Capture frame
        frame = sim.get_frame()

        # Ground truth
        gt = sim.get_ground_truth()

        gt_x = gt["x"]
        gt_y = gt["y"]

        # -----------------------------------------------------
        # Check target visibility
        # -----------------------------------------------------

        if (
            gt_x < 0
            or gt_x >= config["camera"]["width"]
            or gt_y < 0
            or gt_y >= config["camera"]["height"]
        ):
            out_of_view += 1
            continue

        # -----------------------------------------------------
        # ML detection
        # -----------------------------------------------------

        pred_offset = detector.detect(frame)

        if pred_offset is None:
            failures += 1
            continue

        valid_frames += 1

        center_x = config["camera"]["width"] / 2.0
        center_y = config["camera"]["height"] / 2.0

        pred_x = center_x + pred_offset[0]
        pred_y = center_y + pred_offset[1]

        # -----------------------------------------------------
        # Detector error
        # -----------------------------------------------------

        detector_error = np.sqrt(
            (pred_x - gt_x) ** 2 +
            (pred_y - gt_y) ** 2
        )

        detector_errors.append(float(detector_error))

        # -----------------------------------------------------
        # Kalman filter
        # -----------------------------------------------------

        if not kalman.initialized:

            kalman.initialize(
                pred_x,
                pred_y
            )

        else:

            kalman.predict()

            kalman.update(
                pred_x,
                pred_y
            )

        kalman_position = kalman.get_position()
        kalman_velocity = kalman.get_velocity()

        filtered_x = kalman_position["x"]
        filtered_y = kalman_position["y"]

        estimated_vx = kalman_velocity["vx"]
        estimated_vy = kalman_velocity["vy"]

        # -----------------------------------------------------
        # Kalman position error
        # -----------------------------------------------------

        kalman_error = np.sqrt(
            (filtered_x - gt_x) ** 2 +
            (filtered_y - gt_y) ** 2
        )

        kalman_errors.append(float(kalman_error))

        # -----------------------------------------------------
        # Velocity-aware control
        #
        # IMPORTANT:
        # We use ONLY the Kalman velocity.
        #
        # We do NOT add the previous camera command.
        # -----------------------------------------------------

        command = controller.compute_velocity_command(
            filtered_x,
            filtered_y,
            estimated_vx,
            estimated_vy
        )

        # -----------------------------------------------------
        # Move camera
        # -----------------------------------------------------

        sim.move_camera(
            command["pan_speed"],
            command["tilt_speed"]
        )

        # -----------------------------------------------------
        # Print frame information
        # -----------------------------------------------------

        print(
            f"Frame {frame_idx:02d} | "
            f"GT=({gt_x:.2f}, {gt_y:.2f}) | "
            f"ML=({pred_x:.2f}, {pred_y:.2f}) | "
            f"KF=({filtered_x:.2f}, {filtered_y:.2f}) | "
            f"VKF=({estimated_vx:.2f}, "
            f"{estimated_vy:.2f}) | "
            f"ML Err={detector_error:.2f} px | "
            f"KF Err={kalman_error:.2f} px | "
            f"Pan={command['pan_speed']:.2f} | "
            f"Tilt={command['tilt_speed']:.2f}"
        )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("KALMAN-VELOCITY CLOSED-LOOP RESULTS")
    print("=" * 60)

    print(f"Valid frames       : {valid_frames}")
    print(f"Out of view        : {out_of_view}")
    print(f"Failures           : {failures}")

    if detector_errors:

        print(
            f"Mean detector error: "
            f"{np.mean(detector_errors):.2f} px"
        )

        print(
            f"Max detector error : "
            f"{np.max(detector_errors):.2f} px"
        )

        print(
            f"Min detector error : "
            f"{np.min(detector_errors):.2f} px"
        )

    if kalman_errors:

        print(
            f"Mean Kalman error  : "
            f"{np.mean(kalman_errors):.2f} px"
        )

        print(
            f"Max Kalman error   : "
            f"{np.max(kalman_errors):.2f} px"
        )

        print(
            f"Min Kalman error   : "
            f"{np.min(kalman_errors):.2f} px"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()