import numpy as np
import cv2

from tracking.pipeline import TrackingPipeline


def main():
    pipeline = TrackingPipeline()

    print("Tracking Pipeline Motion Test")
    print("=" * 60)

    all_detected = True

    for frame_number in range(30):

        frame = np.zeros((480, 640), dtype=np.uint8)

        # Moving target
        x = 100 + frame_number * 4
        y = 100 + frame_number * 2

        cv2.circle(
            frame,
            (x, y),
            10,
            255,
            -1
        )

        result = pipeline.process(frame)

        if not result["detected"]:
            all_detected = False

        if frame_number % 5 == 0:
            print(
                f"Frame {frame_number:02d} | "
                f"Detected: {result['detected']} | "
                f"Position: {result['position']} | "
                f"Velocity: {result['velocity']}"
            )

    print("\n" + "=" * 60)

    if all_detected:
        print("PASS: All 30 frames were processed successfully.")
    else:
        print("FAIL: One or more frames were not detected.")


if __name__ == "__main__":
    main()