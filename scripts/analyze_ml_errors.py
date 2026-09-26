import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tracking.ml_detector import MLDetector


def load_config():
    with open("config/config.json", "r") as f:
        return json.load(f)


def main():
    config = load_config()
    detector = MLDetector(config)

    df = pd.read_csv("dataset/labels.csv")
    df = df[df["is_visible"] == 1].reset_index(drop=True).reset_index(drop=True)

    results = []

    print(f"Evaluating {len(df)} non-edge images...")

    for _, row in df.iterrows():

        image_path = Path("dataset/images") / row["image_filename"]

        frame = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        prediction = detector.detect(frame)

        pred_x = 320.0 + float(prediction[0])
        pred_y = 240.0 + float(prediction[1])

        gt_x = float(row["center_x"])
        gt_y = float(row["center_y"])

        error = np.sqrt(
            (pred_x - gt_x) ** 2 +
            (pred_y - gt_y) ** 2
        )

        results.append({
            "error": error,
            "image": row["image_filename"],
            "gt_x": gt_x,
            "gt_y": gt_y,
            "pred_x": pred_x,
            "pred_y": pred_y
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "error",
        ascending=False
    )

    print()
    print("=" * 90)
    print("WORST 10 ML DETECTOR PREDICTIONS")
    print("=" * 90)

    print(
        results_df.head(10).to_string(index=False)
    )

    print("=" * 90)


if __name__ == "__main__":
    main()