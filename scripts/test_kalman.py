import numpy as np

from tracking.kalman import KalmanTracker


def calculate_rmse(errors):
    """Calculate root mean square error."""
    return float(np.sqrt(np.mean(np.array(errors) ** 2)))


def main():

    # Simulation parameters
    num_frames = 60
    dt = 1 / 30

    # Known target motion
    start_x = 100.0
    start_y = 100.0

    velocity_x = 90.0   # pixels/second
    velocity_y = 30.0   # pixels/second

    # Create Kalman tracker
    tracker = KalmanTracker(dt=dt)

    # Store errors
    measurement_errors = []
    kalman_errors = []

    # Reproducible random noise
    rng = np.random.default_rng(42)

    print("Kalman Filter - Quantitative Noise Test")
    print("=" * 80)

    for frame in range(num_frames):

        # -------------------------------------------------
        # 1. Ground-truth position
        # -------------------------------------------------

        time = frame * dt

        true_x = start_x + velocity_x * time
        true_y = start_y + velocity_y * time

        # -------------------------------------------------
        # 2. Simulated detector measurement
        # -------------------------------------------------

        noise_x = rng.normal(0, 5)
        noise_y = rng.normal(0, 5)

        measured_x = true_x + noise_x
        measured_y = true_y + noise_y

        # -------------------------------------------------
        # 3. Kalman prediction
        # -------------------------------------------------

        prediction = tracker.predict()

        # -------------------------------------------------
        # 4. Kalman correction
        # -------------------------------------------------

        estimate = tracker.update(
            measured_x,
            measured_y
        )

        # -------------------------------------------------
        # 5. Calculate measurement error
        # -------------------------------------------------

        measurement_error = np.sqrt(
            (measured_x - true_x) ** 2
            + (measured_y - true_y) ** 2
        )

        # -------------------------------------------------
        # 6. Calculate Kalman error
        # -------------------------------------------------

        kalman_error = np.sqrt(
            (estimate["x"] - true_x) ** 2
            + (estimate["y"] - true_y) ** 2
        )

        measurement_errors.append(measurement_error)
        kalman_errors.append(kalman_error)

        # Print every 10th frame
        if frame % 10 == 0:
            print(
                f"Frame {frame:02d} | "
                f"True: ({true_x:.1f}, {true_y:.1f}) | "
                f"Measured: ({measured_x:.1f}, {measured_y:.1f}) | "
                f"Estimate: ({estimate['x']:.1f}, {estimate['y']:.1f})"
            )

    # -----------------------------------------------------
    # Final results
    # -----------------------------------------------------

    measurement_rmse = calculate_rmse(measurement_errors)
    kalman_rmse = calculate_rmse(kalman_errors)

    improvement = (
        (measurement_rmse - kalman_rmse)
        / measurement_rmse
    ) * 100

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(
        f"Raw measurement RMSE : "
        f"{measurement_rmse:.2f} pixels"
    )

    print(
        f"Kalman estimate RMSE  : "
        f"{kalman_rmse:.2f} pixels"
    )

    print(
        f"Error improvement     : "
        f"{improvement:.2f}%"
    )

    print()

    if kalman_rmse < measurement_rmse:
        print("PASS: Kalman filter reduced tracking error.")
    else:
        print("WARNING: Kalman filter did not reduce tracking error.")

    print()
    print("Estimated final velocity:")
    print(tracker.get_velocity())


if __name__ == "__main__":
    main()