import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import cv2
import numpy as np

from tracking.ml_detector import MLDetector
import json


def load_config():
    with open("config/config.json", "r") as f:
        return json.load(f)


def main():

    config = load_config()

    detector = MLDetector(config)

    df = pd.read_csv("dataset/labels.csv")

    # Only evaluate visible targets
    df = df[df["is_visible"] == 1].reset_index(drop=True)

    errors = []

    print("\n" + "=" * 70)
    print("STANDALONE ML DETECTOR EVALUATION")
    print("=" * 70)

    # Evaluate 100 random samples
    samples = df.sample(
        n=min(100, len(df)),
        random_state=42
    )

    for _, row in samples.iterrows():

        image_path = Path("dataset/images") / row["image_filename"]

        frame = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        prediction = detector.detect(frame)

        if prediction is None:
            continue

        pred_x = 320.0 + float(prediction[0])
        pred_y = 240.0 + float(prediction[1])

        gt_x = float(row["center_x"])
        gt_y = float(row["center_y"])

        error = np.sqrt(
            (pred_x - gt_x) ** 2 +
            (pred_y - gt_y) ** 2
        )

        errors.append(error)

    errors = np.array(errors)

    print(f"Samples evaluated : {len(errors)}")
    print(f"Mean error        : {errors.mean():.2f} px")
    print(f"Median error      : {np.median(errors):.2f} px")
    print(f"Maximum error     : {errors.max():.2f} px")
    print(f"Minimum error     : {errors.min():.2f} px")

    print("=" * 70)


if __name__ == "__main__":
    main()