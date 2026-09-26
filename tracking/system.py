import numpy as np

from tracking.ml_detector import MLDetector
from tracking.kalman import KalmanTracker
from control.controller import CameraController


class TrackingSystem:
    """
    High-level tracking API.

    Internally:

        Frame
          ↓
        ML Detector
          ↓
        Kalman Filter
          ↓
        Velocity-Aware Controller
          ↓
        Tracking Result

    External code only needs to call:

        process(frame)
    """

    def __init__(
        self,
        config,
        acceleration_noise=500.0,
        gain=4.0,
        velocity_scale=0.5,
        max_missed_frames=10
    ):
        self.config = config

        camera_width = config["camera"]["width"]
        camera_height = config["camera"]["height"]
        fps = config["camera"]["fps"]

        # --------------------------------------------------
        # ML Detector
        # --------------------------------------------------

        self.detector = MLDetector(config)

        # --------------------------------------------------
        # Kalman Filter
        # --------------------------------------------------

        self.tracker = KalmanTracker(
            dt=1.0 / fps,
            acceleration_noise=acceleration_noise
        )

        # --------------------------------------------------
        # Camera Controller
        # --------------------------------------------------

        self.controller = CameraController(
            frame_width=camera_width,
            frame_height=camera_height,
            max_pan_speed=config["control"]["max_pan_speed"],
            max_tilt_speed=config["control"]["max_tilt_speed"],
            gain=gain,
            velocity_scale=velocity_scale
        )

        self.max_missed_frames = max_missed_frames
        self.missed_frames = 0

        self.center_x = camera_width / 2.0
        self.center_y = camera_height / 2.0

    def process(self, frame):
        """
        Process one camera frame.

        Returns:

            detected
            tracking
            detector_position
            position
            velocity
            confidence
            command
        """

        prediction_offset = self.detector.detect(frame)

        # --------------------------------------------------
        # DETECTION SUCCESS
        # --------------------------------------------------

        if prediction_offset is not None:

            predicted_x = (
                self.center_x + prediction_offset[0]
            )

            predicted_y = (
                self.center_y + prediction_offset[1]
            )

            detector_position = {
                "x": float(predicted_x),
                "y": float(predicted_y)
            }

            self.missed_frames = 0

            # Initialize Kalman filter on first detection.
            if not self.tracker.initialized:

                self.tracker.initialize(
                    predicted_x,
                    predicted_y
                )

            else:

                self.tracker.predict()

                self.tracker.update(
                    predicted_x,
                    predicted_y
                )

            position = self.tracker.get_position()
            velocity = self.tracker.get_velocity()

            command = self.controller.compute_velocity_command(
                position["x"],
                position["y"],
                velocity["vx"],
                velocity["vy"]
            )

            return {
                "detected": True,
                "tracking": True,

                "detector_position": detector_position,

                "position": position,

                "velocity": velocity,

                "confidence": None,

                "command": {
                    "pan_speed": command["pan_speed"],
                    "tilt_speed": command["tilt_speed"]
                }
            }

        # --------------------------------------------------
        # DETECTION LOST — USE KALMAN PREDICTION
        # --------------------------------------------------

        if self.tracker.initialized:

            self.missed_frames += 1

            if self.missed_frames <= self.max_missed_frames:

                self.tracker.predict()

                position = self.tracker.get_position()
                velocity = self.tracker.get_velocity()

                command = self.controller.compute_velocity_command(
                    position["x"],
                    position["y"],
                    velocity["vx"],
                    velocity["vy"]
                )

                return {
                    "detected": False,
                    "tracking": True,

                    "detector_position": None,

                    "position": position,

                    "velocity": velocity,

                    "confidence": None,

                    "command": {
                        "pan_speed": command["pan_speed"],
                        "tilt_speed": command["tilt_speed"]
                    }
                }

        # --------------------------------------------------
        # TRACKING LOST
        # --------------------------------------------------

        return {
            "detected": False,
            "tracking": False,

            "detector_position": None,

            "position": None,

            "velocity": None,

            "confidence": None,

            "command": {
                "pan_speed": 0.0,
                "tilt_speed": 0.0
            }
        }