import cv2
import numpy as np

from vision.detector import TargetDetector


def main():
    # Create a black 640 x 480 image
    frame = np.zeros((480, 640), dtype=np.uint8)

    # Create a bright circular target at (320, 240)
    cv2.circle(
        frame,
        (320, 240),
        10,
        255,
        -1
    )

    # Create detector
    detector = TargetDetector(
        threshold=200,
        min_area=5,
        max_area=5000
    )

    # Detect target
    result = detector.detect(frame)

    # Print result
    print("Detection Result:")
    print(result)

    # Display image
    cv2.imshow("Detector Test", frame)

    print("\nPress any key inside the image window to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()