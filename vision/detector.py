import cv2
import numpy as np


class TargetDetector:
    def __init__(self, threshold=200, min_area=5, max_area=5000):
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, frame):
        """
        Detect a bright target in an image.

        Args:
            frame: OpenCV image (grayscale or BGR)

        Returns:
            Dictionary containing:
                x          -> target x-coordinate
                y          -> target y-coordinate
                confidence -> detection confidence
                detected   -> whether a target was found
        """

        # Safety check
        if frame is None:
            return {
                "x": None,
                "y": None,
                "confidence": 0.0,
                "detected": False
            }

        # Convert BGR image to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Threshold bright pixels
        _, binary = cv2.threshold(
            gray,
            self.threshold,
            255,
            cv2.THRESH_BINARY
        )

        # Find connected bright regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary
        )

        best_candidate = None
        best_area = 0

        # Ignore label 0 because it represents the background
        for i in range(1, num_labels):

            area = stats[i, cv2.CC_STAT_AREA]

            # Reject regions that are too small or too large
            if self.min_area <= area <= self.max_area:

                # Choose the largest valid region
                if area > best_area:
                    best_area = area
                    best_candidate = centroids[i]

        # No valid target found
        if best_candidate is None:
            return {
                "x": None,
                "y": None,
                "confidence": 0.0,
                "detected": False
            }

        # Target centroid
        x, y = best_candidate

        # Simple confidence score
        confidence = min(best_area / self.max_area, 1.0)

        return {
            "x": float(x),
            "y": float(y),
            "confidence": float(confidence),
            "detected": True
        }