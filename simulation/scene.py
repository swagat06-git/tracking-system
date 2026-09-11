import numpy as np
import cv2


class Scene:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def create_canvas(self):
        """Create an empty grayscale canvas."""
        canvas = np.zeros(
            (self.height, self.width),
            dtype=np.uint8
        )

        return canvas

    def draw_target(self, canvas, target):
        """Draw the target on the canvas."""

        x, y = target.get_position()

        cv2.circle(
            canvas,
            (int(x), int(y)),
            target.size,
            target.brightness,
            -1
        )

        return canvas