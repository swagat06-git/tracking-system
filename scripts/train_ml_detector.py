import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# MPS-Compatible Lightweight CNN Architecture
class TargetDetectorCNN(nn.Module):
    def __init__(self):
        super(TargetDetectorCNN, self).__init__()
        # Input shape: 1 x 480 x 640
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> 16 x 240 x 320
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                   # -> 16 x 120 x 160
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> 32 x 60 x 80
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                   # -> 32 x 30 x 40
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> 64 x 15 x 20
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                                    # -> 64 x 7 x 10
        )
        # 64 channels * 7 height * 10 width = 4480
        self.regressor = nn.Sequential(
            nn.Linear(64 * 7 * 10, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Output: (norm_center_x, norm_center_y)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)


# PyTorch Dataset Loader
class TargetDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df["is_visible"] == 1].reset_index(drop=True)
        self.img_dir = Path(img_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self.img_dir / row["image_filename"]

    # Load NumPy image
        import numpy as np
        img = np.load(img_path)

    # Convert to float32 and normalize to [0, 1]
        tensor_img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0) / 255.0

        target = torch.tensor(
            [row["norm_center_x"], row["norm_center_y"]],
            dtype=torch.float32
        )

        return tensor_img, target


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nTraining PyTorch Target Detector on device: {device}")

    dataset_path = Path("dataset")
    csv_file = dataset_path / "labels.csv"
    img_dir = dataset_path / "images"

    dataset = TargetDataset(csv_file, img_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = TargetDetectorCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 15
    print("-" * 60)
    print(f"{'Epoch':<8} | {'Train Loss (MSE)':<18} | {'Val Loss (MSE)':<18}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(val_dataset)

        print(f"{epoch:<8} | {train_loss:<18.6f} | {val_loss:<18.6f}")

    # Save model weights
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_save_path = models_dir / "target_detector_cnn.pth"
    torch.save(model.state_dict(), model_save_path)

    print("-" * 60)
    print(f"SUCCESS: Model saved to {model_save_path.resolve()}\n")


if __name__ == "__main__":
    train()