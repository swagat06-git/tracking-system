import cv2
import numpy as np


class Atmosphere:

    def __init__(self, mode="clear"):
        self.mode = mode.lower()

    def apply(self, frame):
        if frame is None:
            return None

        if self.mode == "clear":
            return frame

        if self.mode == "haze":
            return self._haze(frame)

        if self.mode == "fog":
            return self._fog(frame)

        if self.mode == "rain":
            return self._rain(frame)

        if self.mode == "low_light":
            return self._low_light(frame)

        raise ValueError(
            f"Unknown atmosphere mode: {self.mode}"
        )

    def _haze(self, frame):
        # Reduce contrast and slightly increase brightness.
        result = cv2.convertScaleAbs(
            frame,
            alpha=0.90,
            beta=10
        )

        return result

    def _fog(self, frame):
        # Blur the scene and blend it toward a bright gray/white background.
        blurred = cv2.GaussianBlur(
            frame,
            (9, 9),
            0
        )

        fog_layer = np.full_like(
            blurred,
            220
        )

        result = cv2.addWeighted(
            blurred,
            0.65,
            fog_layer,
            0.35,
            0
        )

        return result

    def _rain(self, frame):
        result = frame.copy()

        height, width = result.shape[:2]

        number_of_streaks = max(
            20,
            int(width * height * 0.00005)
        )

        for _ in range(number_of_streaks):

            x = np.random.randint(0, width)
            y = np.random.randint(0, height)

            length = np.random.randint(8, 20)

            x2 = min(width - 1, x + 2)
            y2 = min(height - 1, y + length)

            cv2.line(
                result,
                (x, y),
                (x2, y2),
                180,
                1
            )

        return result

    def _low_light(self, frame):
        # Reduce brightness and contrast.
        result = cv2.convertScaleAbs(
            frame,
            alpha=0.65,
            beta=0
        )

        # Add a small amount of sensor noise.
        noise = np.random.normal(
            0,
            2,
            result.shape
        )

        result = np.clip(
            result.astype(np.float32) + noise,
            0,
            255
        ).astype(np.uint8)

        return result