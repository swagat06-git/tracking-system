import numpy as np


class KalmanTracker:
    """
    2D Kalman filter for target tracking.

    State:
        [x, y, vx, vy]

    x, y   -> target position in pixels
    vx, vy -> target velocity in pixels/second
    """

    def __init__(self, dt=1 / 30, acceleration_noise=50.0):
        self.dt = dt

        # State vector: [x, y, vx, vy]
        self.state = np.zeros((4, 1), dtype=np.float64)

        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=np.float64)

        # Measurement matrix.
        # The detector provides x and y only.
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)

        # Initial state uncertainty
        self.P = np.eye(4, dtype=np.float64) * 100.0

        # Process noise.
        # Models small unpredictable changes in target motion.
        q = acceleration_noise ** 2

        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4

        self.Q = q * np.array([
            [dt4 / 4, 0,        dt3 / 2, 0],
            [0,        dt4 / 4, 0,        dt3 / 2],
            [dt3 / 2, 0,        dt2,      0],
            [0,        dt3 / 2, 0,        dt2]
        ], dtype=np.float64)

        # Measurement noise.
        # This represents uncertainty in detector coordinates.
        measurement_noise = 5.0

        self.R = np.eye(2, dtype=np.float64) * (
            measurement_noise ** 2
        )

        self.initialized = False

    def initialize(self, x, y):
        """
        Initialize the tracker using the first detection.
        Initial velocity is assumed to be zero.
        """

        self.state = np.array([
            [float(x)],
            [float(y)],
            [0.0],
            [0.0]
        ], dtype=np.float64)

        self.initialized = True

    def predict(self):
        """
        Predict the target's next state.

        Returns:
            Dictionary containing predicted x and y.
            Returns None if the tracker has not been initialized.
        """

        if not self.initialized:
            return None

        self.state = self.F @ self.state

        self.P = (
            self.F @ self.P @ self.F.T
            + self.Q
        )

        return self.get_position()

    def update(self, x, y):
        """
        Correct the predicted state using a new detection.

        Args:
            x: Detected x coordinate.
            y: Detected y coordinate.

        Returns:
            Corrected target position.
        """

        if not self.initialized:
            self.initialize(x, y)
            return self.get_position()

        measurement = np.array([
            [float(x)],
            [float(y)]
        ], dtype=np.float64)

        # Measurement residual
        innovation = (
            measurement
            - self.H @ self.state
        )

        # Innovation covariance
        S = (
            self.H @ self.P @ self.H.T
            + self.R
        )

        # Kalman gain
        K = (
            self.P
            @ self.H.T
            @ np.linalg.inv(S)
        )

        # Correct the state
        self.state = (
            self.state
            + K @ innovation
        )

        # Correct covariance
        I = np.eye(4)

        self.P = (
            I - K @ self.H
        ) @ self.P

        return self.get_position()

    def get_position(self):
        """
        Return the current estimated target position.
        """

        if not self.initialized:
            return None

        return {
            "x": float(self.state[0, 0]),
            "y": float(self.state[1, 0])
        }

    def get_velocity(self):
        """
        Return the current estimated target velocity.
        """

        if not self.initialized:
            return None

        return {
            "vx": float(self.state[2, 0]),
            "vy": float(self.state[3, 0])
        }