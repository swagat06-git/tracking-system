import pandas as pd
import numpy as np
import torch
from pathlib import Path
from scripts.train_ml_detector import TargetDetectorCNN

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = TargetDetectorCNN().to(device)

model.load_state_dict(
    torch.load(
        "models/target_detector_cnn.pth",
        map_location=device
    )
)

model.eval()

df = pd.read_csv("dataset/labels.csv")
df = df[df["is_visible"] == 1].reset_index(drop=True)

results = []

with torch.no_grad():

    for _, row in df.sample(100, random_state=42).iterrows():

        img = np.load(
            Path("dataset/images") / row["image_filename"]
        )

        tensor = (
            torch.from_numpy(
                img.astype(np.float32)
            )
            .unsqueeze(0)
            .unsqueeze(0)
            / 255.0
        ).to(device)

        output = model(tensor)[0].cpu().numpy()

        pred_x = output[0] * 640
        pred_y = output[1] * 480

        true_x = row["center_x"]
        true_y = row["center_y"]

        dx = pred_x - true_x
        dy = pred_y - true_y

        error = np.sqrt(dx * dx + dy * dy)

        results.append({
            "file": row["image_filename"],
            "true_x": true_x,
            "true_y": true_y,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "error": error,
            "noise": row["noise_type"],
            "noise_level": row["noise_level"]
        })

results.sort(key=lambda x: x["error"], reverse=True)

print()
print("10 WORST CNN PREDICTIONS")
print("========================")

for i, r in enumerate(results[:10], 1):

    print(
        f"{i:02d}. "
        f"{r['file']} | "
        f"Noise={r['noise']}({r['noise_level']}) | "
        f"GT=({r['true_x']:.1f},{r['true_y']:.1f}) | "
        f"Pred=({r['pred_x']:.1f},{r['pred_y']:.1f}) | "
        f"Error={r['error']:.2f}px"
    )