import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path


class TargetDetectorCNN(nn.Module):
    def __init__(self):
        super(TargetDetectorCNN, self).__init__()
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


class MLDetector:
    def __init__(self, config: dict, model_path: str = "models/target_detector_cnn.pth"):
        self.config = config
        self.img_width = config["camera"]["width"]
        self.img_height = config["camera"]["height"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        self.model = TargetDetectorCNN().to(self.device)
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Trained model file not found at {model_path}. Please run train_ml_detector.py first.")
        
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.model.eval()

    def detect(self, frame: np.ndarray):
        """
        Takes an image frame (numpy array) and returns predicted pixel coordinates [x, y]
        relative to the image center, matching the pipeline's detection interface.
        """
        if frame is None:
            return None

        # Ensure grayscale format
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Normalize and convert to PyTorch tensor [1, 1, H, W]
        tensor_img = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        tensor_img = tensor_img.to(self.device)

        with torch.no_grad():
            output = self.model(tensor_img)
            norm_x, norm_y = output[0].cpu().numpy()

        # Denormalize coordinates to pixel values
        pixel_x = norm_x * self.img_width
        pixel_y = norm_y * self.img_height

        # Convert to center-offset coordinates used by tracking pipeline
        center_x = pixel_x - (self.img_width / 2.0)
        center_y = pixel_y - (self.img_height / 2.0)

        return np.array([center_x, center_y], dtype=np.float32)