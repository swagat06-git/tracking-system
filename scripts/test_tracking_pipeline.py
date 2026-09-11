import random
import math

from simulation.target import Target
from simulation.motion import Motion
from simulation.scene import Scene
from vision.detector import TargetDetector
from tracking.kalman import KalmanTracker


def main():
    width = 640
    height = 480
    dt = 1 / 30
    frames = 60

    # Expected simulator velocity
    expected_vx = 120.0
    expected_vy = 60.0

    # Maximum acceptable final errors
    max_position_error = 5.0
    max_velocity_error = 15.0

    # -----------------------------
    # Simulation
    # -----------------------------

    target = Target(
        x=100,
        y=100,
        size=10,
        brightness=255
    )

    target.set_velocity(expected_vx, expected_vy)

    motion = Motion(
        motion_type="linear",
        width=width,
        height=height
    )

    scene = Scene(width, height)

    # -----------------------------
    # Detector + Kalman
    # -----------------------------

    detector = TargetDetector(
        threshold=200,
        min_area=5,
        max_area=5000
    )

    tracker = KalmanTracker(dt=dt)

    valid_frames = 0

    print("Integrated Tracking Pipeline")
    print("=" * 80)

    # -----------------------------
    # Main pipeline
    # -----------------------------

    for frame_number in range(frames):

        # Move true target
        motion.update(target, dt)

        true_x, true_y = target.get_position()

        # Render target
        frame = scene.create_canvas()
        frame = scene.draw_target(frame, target)

        # Detect target
        detection = detector.detect(frame)

        if not detection["detected"]:
            print(f"Frame {frame_number:02d} | Detection FAILED")
            continue

        # Simulate detector noise
        measured_x = detection["x"] + random.gauss(0, 3)
        measured_y = detection["y"] + random.gauss(0, 3)

        # Kalman prediction
        tracker.predict()

        # Kalman correction
        estimate = tracker.update(
            measured_x,
            measured_y
        )

        valid_frames += 1

        if frame_number % 10 == 0:
            print(
                f"Frame {frame_number:02d} | "
                f"True: ({true_x:.1f}, {true_y:.1f}) | "
                f"Measured: ({measured_x:.1f}, {measured_y:.1f}) | "
                f"Kalman: ({estimate['x']:.1f}, {estimate['y']:.1f})"
            )

    # -----------------------------
    # Validation
    # -----------------------------

    if valid_frames == 0:
        print("\nFAIL: No detections.")
        return

    final_x, final_y = target.get_position()

    estimated_position = tracker.get_position()
    estimated_velocity = tracker.get_velocity()

    position_error = math.sqrt(
        (estimated_position["x"] - final_x) ** 2
        + (estimated_position["y"] - final_y) ** 2
    )

    velocity_error = math.sqrt(
        (estimated_velocity["vx"] - expected_vx) ** 2
        + (estimated_velocity["vy"] - expected_vy) ** 2
    )

    print("\n" + "=" * 80)
    print("INTEGRATION RESULTS")
    print("=" * 80)

    print(f"Valid detections       : {valid_frames}/{frames}")

    print(
        f"True final position    : "
        f"({final_x:.2f}, {final_y:.2f})"
    )

    print(
        f"Estimated final position: "
        f"({estimated_position['x']:.2f}, "
        f"{estimated_position['y']:.2f})"
    )

    print(
        f"Final position error    : "
        f"{position_error:.2f} pixels"
    )

    print(
        f"\nExpected velocity       : "
        f"({expected_vx:.2f}, {expected_vy:.2f})"
    )

    print(
        f"Estimated velocity      : "
        f"({estimated_velocity['vx']:.2f}, "
        f"{estimated_velocity['vy']:.2f})"
    )

    print(
        f"Velocity error          : "
        f"{velocity_error:.2f} pixels/s"
    )

    # -----------------------------
    # Final pass/fail
    # -----------------------------

    position_pass = position_error <= max_position_error
    velocity_pass = velocity_error <= max_velocity_error
    detection_pass = valid_frames == frames

    print("\n" + "=" * 80)

    if detection_pass and position_pass and velocity_pass:
        print("PASS: Integrated tracking pipeline is working.")
        print("Simulator -> Detector -> Kalman")
    else:
        print("FAIL: Integrated tracking validation failed.")

        if not detection_pass:
            print("- Detector missed one or more frames.")

        if not position_pass:
            print(
                f"- Final position error exceeds "
                f"{max_position_error:.1f} pixels."
            )

        if not velocity_pass:
            print(
                f"- Velocity error exceeds "
                f"{max_velocity_error:.1f} pixels/s."
            )


if __name__ == "__main__":
    main()
