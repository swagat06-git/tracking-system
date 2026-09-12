from tracking.kalman import KalmanTracker
from vision.detector import TargetDetector


class TrackingPipeline:
    """
    Connects target detection and Kalman tracking.
    """

    def __init__(self, detector=None, tracker=None):
        self.detector = detector or TargetDetector()
        self.tracker = tracker or KalmanTracker()

    def process(self, frame):
        """
        Process one image frame.

        Returns detection, position, velocity and confidence.
        """

        detection = self.detector.detect(frame)

        if not detection["detected"]:
            return {
                "detected": False,
                "position": None,
                "velocity": None,
                "confidence": 0.0
            }
        
        self.tracker.predict()
        estimate = self.tracker.update(
            detection["x"],
            detection["y"]
        )

        return {
            "detected": True,
            "position": estimate,
            "velocity": self.tracker.get_velocity(),
            "confidence": detection["confidence"]
        }