from tracking.kalman import KalmanTracker
from vision.detector import TargetDetector


class TrackingPipeline:
    """
    Connects target detection and Kalman tracking.

    The pipeline supports temporary detection loss by using
    Kalman prediction for a limited number of missed frames.
    """

    def __init__(
        self,
        detector=None,
        tracker=None,
        max_missed_frames=10
    ):
        self.detector = detector or TargetDetector()
        self.tracker = tracker or KalmanTracker()

        # Maximum number of consecutive frames that can be
        # handled using prediction without a detection.
        self.max_missed_frames = max_missed_frames

        self.missed_frames = 0

    def process(self, frame):
        """
        Process one image frame.

        If detection is available:
            Predict + correct using the detection.

        If detection is temporarily unavailable:
            Continue using Kalman prediction.

        If detection remains unavailable for too long:
            Mark tracking as lost.
        """

        detection = self.detector.detect(frame)

        # -----------------------------------------
        # Detection available
        # -----------------------------------------

        if detection["detected"]:

            self.missed_frames = 0

            self.tracker.predict()

            estimate = self.tracker.update(
                detection["x"],
                detection["y"]
            )

            return {
                "detected": True,
                "tracking": True,
                "position": estimate,
                "velocity": self.tracker.get_velocity(),
                "confidence": detection["confidence"]
            }

        # -----------------------------------------
        # Detection unavailable
        # -----------------------------------------

        if self.tracker.initialized:

            self.missed_frames += 1

            # Continue prediction while within the
            # allowed temporary-loss window.
            if self.missed_frames <= self.max_missed_frames:

                estimate = self.tracker.predict()

                return {
                    "detected": False,
                    "tracking": True,
                    "position": estimate,
                    "velocity": self.tracker.get_velocity(),
                    "confidence": 0.0
                }

        # -----------------------------------------
        # Tracking lost
        # -----------------------------------------

        return {
            "detected": False,
            "tracking": False,
            "position": None,
            "velocity": None,
            "confidence": 0.0
        }