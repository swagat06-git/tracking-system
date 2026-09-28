import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path


class TargetDetectorCNN(nn.Module):
    def __init__(self):
        super(TargetDetectorCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.regressor = nn.Sequential(
            nn.Linear(64 * 7 * 10, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)


class MLDetector:

    def __init__(
        self,
        config: dict,
        model_path: str = "models/target_detector_cnn.pth"
    ):
        self.config = config

        self.img_width = config["camera"]["width"]
        self.img_height = config["camera"]["height"]

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model = TargetDetectorCNN().to(self.device)

        model_file = Path(model_path)

        if not model_file.exists():
            raise FileNotFoundError(
                f"Trained model file not found at {model_path}. "
                "Please run train_ml_detector.py first."
            )

        self.model.load_state_dict(
            torch.load(
                model_file,
                map_location=self.device
            )
        )

        self.model.eval()

        self.target_size = config["target"]["size"]
        self.noise_type = config["disturbance"]["noise_type"]
        self.camera_jitter = config["disturbance"]["camera_jitter"]

        # -----------------------------------------------------
        # Detection history
        # -----------------------------------------------------

        self.last_detection = None

        # -----------------------------------------------------
        # Maximum believable target displacement.
        #
        # With camera jitter, the apparent target position can
        # change by up to twice the jitter amount between frames.
        # -----------------------------------------------------

        self.max_jump = max(
            30.0,
            30.0 + 2.0 * self.camera_jitter
        )

    # =========================================================
    # CNN DETECTOR
    # =========================================================

    def _cnn_detect(self, gray):

        tensor_img = (
            torch.tensor(
                gray,
                dtype=torch.float32
            )
            .unsqueeze(0)
            .unsqueeze(0)
            / 255.0
        )

        tensor_img = tensor_img.to(self.device)

        with torch.no_grad():
            output = self.model(tensor_img)

            norm_x, norm_y = (
                output[0]
                .cpu()
                .numpy()
            )

        pixel_x = norm_x * self.img_width
        pixel_y = norm_y * self.img_height

        center_x = (
            pixel_x
            - self.img_width / 2.0
        )

        center_y = (
            pixel_y
            - self.img_height / 2.0
        )

        return np.array(
            [center_x, center_y],
            dtype=np.float32
        )

    # =========================================================
    # BRIGHT TARGET DETECTOR
    # =========================================================

    def _bright_target_detect(self, gray):

        _, binary = cv2.threshold(
            gray,
            220,
            255,
            cv2.THRESH_BINARY
        )

    # Small cleanup for isolated bright noise.
        kernel = np.ones((3, 3), np.uint8)

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel
        )

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )

        if num_labels <= 1:
            return None

        candidates = []

        for label in range(1, num_labels):

            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]

        # -------------------------------------------------
        # Target-size filtering
        # Allow a wider range because rendering can enlarge
        # the nominal target.
        # -------------------------------------------------

            min_size = max(5, int(self.target_size * 0.5))
            max_size = int(self.target_size * 3.0)

            if width < min_size or width > max_size:
                continue

            if height < min_size or height > max_size:
                continue

        # -------------------------------------------------
        # Target should approximately be square.
        # -------------------------------------------------

            aspect_ratio = width / max(height, 1)

            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue

        # -------------------------------------------------
        # Prefer compact bright objects.
        # -------------------------------------------------

            expected_area = self.target_size * self.target_size

            area_error = abs(area - expected_area)

            size_error = (
                abs(width - self.target_size)
                + abs(height - self.target_size)
            )

            shape_score = (
                size_error * 2.0
                + area_error * 0.25
            )

            centroid = centroids[label]

            pixel_x = float(centroid[0])
            pixel_y = float(centroid[1])

            offset_x = pixel_x - self.img_width / 2.0
            offset_y = pixel_y - self.img_height / 2.0

            candidate = np.array(
                [offset_x, offset_y],
                dtype=np.float32
            )

            candidates.append(
                (shape_score, candidate)
            )

        if not candidates:
            return None

    # -----------------------------------------------------
    # Temporal consistency
    # -----------------------------------------------------

        if self.last_detection is not None:

            best_candidate = None
            best_score = float("inf")

            for shape_score, candidate in candidates:

                jump = float(
                    np.linalg.norm(
                        candidate - self.last_detection
                    )
                )

                if jump > self.max_jump:
                    continue

                combined_score = (
                    shape_score
                    + jump * 5.0
                )

                if combined_score < best_score:
                    best_score = combined_score
                    best_candidate = candidate

            if best_candidate is not None:
                return best_candidate

            return None

    # -----------------------------------------------------
    # Initial acquisition
    # -----------------------------------------------------

        candidates.sort(
            key=lambda item: item[0]
        )

        return candidates[0][1]

    # =========================================================
    # PUBLIC DETECTOR
    # =========================================================

    def detect(self, frame: np.ndarray):

        if frame is None:
            return None

    # -----------------------------------------------------
    # Convert to grayscale
    # -----------------------------------------------------

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

    # -----------------------------------------------------
    # Gaussian noise preprocessing
    # -----------------------------------------------------

        if self.noise_type == "gaussian":
            gray = cv2.GaussianBlur(
                gray,
                (5, 5),
                0
            )

    # -----------------------------------------------------
    # Salt & Pepper mode
    # -----------------------------------------------------

        if self.noise_type == "salt_pepper":

            detection = self._salt_pepper_detect(gray)

            if detection is not None:
                self.last_detection = detection.copy()
                return detection

        # Allow Kalman to handle temporary misses
            if self.last_detection is not None:
                return None

        # CNN fallback for initial acquisition
                detection = self._cnn_detect(gray)

            if detection is not None:
                self.last_detection = detection.copy()

            return detection

    # -----------------------------------------------------
    # Primary bright-target detection
    # -----------------------------------------------------

        detection = self._bright_target_detect(gray)

        if detection is not None:
            self.last_detection = detection.copy()
            return detection

    # -----------------------------------------------------
    # CNN fallback
    # -----------------------------------------------------

        

        detection = self._cnn_detect(gray)

        if detection is not None:
            self.last_detection = detection.copy()

            return detection