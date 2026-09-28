import csv
import random
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

NUM_IMAGES = 5000

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

TARGET_SIZE = 10
TARGET_BRIGHTNESS = 255

DATASET_DIR = Path("dataset")
IMAGE_DIR = DATASET_DIR / "images"
LABEL_FILE = DATASET_DIR / "labels.csv"

random.seed(42)
np.random.seed(42)


# ============================================================
# IMAGE GENERATION
# ============================================================

def create_clean_frame(target_x, target_y):
    """
    Create a black 640x480 image with a white square target.
    """
    frame = np.zeros(
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        dtype=np.uint8
    )

    half = TARGET_SIZE // 2

    x1 = int(round(target_x - half))
    y1 = int(round(target_y - half))
    x2 = x1 + TARGET_SIZE
    y2 = y1 + TARGET_SIZE

    # Draw only if target intersects the image.
    if x2 > 0 and x1 < IMAGE_WIDTH and y2 > 0 and y1 < IMAGE_HEIGHT:
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2 - 1, y2 - 1),
            TARGET_BRIGHTNESS,
            -1
        )

    return frame


def add_gaussian_noise(frame, noise_level):
    """
    Add Gaussian noise with the requested standard deviation.
    """
    noise = np.random.normal(
        loc=0.0,
        scale=noise_level,
        size=frame.shape
    )

    noisy = frame.astype(np.float32) + noise

    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(frame, probability=0.10):
    """
    Add salt-and-pepper noise.

    probability = fraction of pixels affected approximately.
    Half become white and half become black.
    """
    noisy = frame.copy()

    random_matrix = np.random.random(frame.shape)

    salt_mask = random_matrix < (probability / 2.0)
    pepper_mask = random_matrix > (1.0 - probability / 2.0)

    noisy[salt_mask] = 255
    noisy[pepper_mask] = 0

    return noisy


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset():

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old dataset images.
    old_images = list(IMAGE_DIR.glob("*.npy"))

    print(f"Removing {len(old_images)} old dataset images...")

    for image_file in old_images:
        image_file.unlink()

    # Dataset distribution:
    #
    # 15% clean
    # 15% Gaussian 5-10
    # 15% Gaussian 10-15
    # 15% Gaussian 15-20
    # 40% Salt & Pepper 10%
    #
    # Total = 100%

    clean_count = int(NUM_IMAGES * 0.15)
    gaussian_low_count = int(NUM_IMAGES * 0.15)
    gaussian_mid_count = int(NUM_IMAGES * 0.15)
    gaussian_high_count = int(NUM_IMAGES * 0.15)

    salt_pepper_count = (
        NUM_IMAGES
        - clean_count
        - gaussian_low_count
        - gaussian_mid_count
        - gaussian_high_count
    )

    conditions = []

    # Clean
    conditions.extend(
        [("clean", 0.0)] * clean_count
    )

    # Gaussian 5-10
    for _ in range(gaussian_low_count):
        conditions.append(
            ("gaussian", random.uniform(5.0, 10.0))
        )

    # Gaussian 10-15
    for _ in range(gaussian_mid_count):
        conditions.append(
            ("gaussian", random.uniform(10.0, 15.0))
        )

    # Gaussian 15-20
    for _ in range(gaussian_high_count):
        conditions.append(
            ("gaussian", random.uniform(15.0, 20.0))
        )

    # Salt and pepper 10%
    conditions.extend(
        [("salt_pepper", 0.10)] * salt_pepper_count
    )

    random.shuffle(conditions)

    with open(LABEL_FILE, "w", newline="") as csv_file:

        writer = csv.writer(csv_file)

        # IMPORTANT:
        # These column names must match train_ml_detector.py.
        writer.writerow([
            "image_filename",
            "center_x",
            "center_y",
            "norm_center_x",
            "norm_center_y",
            "is_visible",
            "noise_type",
            "noise_level"
        ])

        for index, (noise_type, noise_level) in enumerate(conditions):

            # ------------------------------------------------
            # Keep targets visible.
            # ------------------------------------------------
            #
            # The target is intentionally sampled with enough
            # margin so the complete target remains inside the
            # 640x480 camera frame.
            #
            target_x = random.uniform(
                TARGET_SIZE,
                IMAGE_WIDTH - TARGET_SIZE
            )

            target_y = random.uniform(
                TARGET_SIZE,
                IMAGE_HEIGHT - TARGET_SIZE
            )

            # Create clean target image.
            frame = create_clean_frame(
                target_x,
                target_y
            )

            # Apply disturbance.
            if noise_type == "gaussian":

                frame = add_gaussian_noise(
                    frame,
                    noise_level
                )

            elif noise_type == "salt_pepper":

                frame = add_salt_pepper_noise(
                    frame,
                    probability=0.10
                )

            # ------------------------------------------------
            # Save image.
            # ------------------------------------------------

            filename = f"frame_{index:05d}.npy"

            np.save(
                IMAGE_DIR / filename,
                frame
            )

            # ------------------------------------------------
            # Normalized target coordinates.
            # ------------------------------------------------

            norm_x = target_x / IMAGE_WIDTH
            norm_y = target_y / IMAGE_HEIGHT

            writer.writerow([
                filename,
                target_x,
                target_y,
                norm_x,
                norm_y,
                1,
                noise_type,
                noise_level
            ])

    print()
    print("=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Images generated : {NUM_IMAGES}")
    print(f"Image size       : {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
    print(f"Clean            : {clean_count}")
    print(f"Gaussian 5-10    : {gaussian_low_count}")
    print(f"Gaussian 10-15   : {gaussian_mid_count}")
    print(f"Gaussian 15-20   : {gaussian_high_count}")
    print(f"Salt & Pepper    : {salt_pepper_count}")
    print(f"Labels           : {LABEL_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    generate_dataset()