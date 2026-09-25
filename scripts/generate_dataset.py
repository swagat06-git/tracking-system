import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import csv
import random
import shutil
import cv2

from simulation.simulator import Simulator


def load_config(config_path: str = "config/config.json") -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def generate_dataset(
    num_episodes: int = 50,
    frames_per_episode: int = 100,
    output_dir: str = "dataset"
):
    """
    Generate a spatially diverse target-detection dataset.

    Important:
    - Target position is randomized for EVERY frame.
    - Target always remains safely inside the camera view.
    - Noise and camera jitter are disabled initially.
    - Ground-truth coordinates are in image/camera coordinates.
    """

    dataset_path = Path(output_dir)
    images_path = dataset_path / "images"

    # ---------------------------------------------------------
    # PREPARE DATASET DIRECTORY
    # ---------------------------------------------------------

    images_path.mkdir(parents=True, exist_ok=True)

    # Remove old generated images so stale files cannot remain
    old_images = list(images_path.glob("*.png"))

    if old_images:
        print(f"Removing {len(old_images)} old dataset images...")

        for image_path in old_images:
            image_path.unlink()

    csv_file_path = dataset_path / "labels.csv"

    # ---------------------------------------------------------
    # LOAD CONFIGURATION
    # ---------------------------------------------------------

    config = load_config()

    target_size = config["target"]["size"]

    img_width = config["camera"]["width"]
    img_height = config["camera"]["height"]

    # ---------------------------------------------------------
    # DATASET SETTINGS
    # ---------------------------------------------------------

    total_frames = num_episodes * frames_per_episode

    # Fixed seed makes dataset generation reproducible.
    random.seed(42)

    print()
    print("=" * 70)
    print(f"{'DATASET GENERATION':^70}")
    print("=" * 70)

    print(f"Episodes              : {num_episodes}")
    print(f"Frames per episode    : {frames_per_episode}")
    print(f"Total frames          : {total_frames}")
    print(f"Image resolution      : {img_width} x {img_height}")
    print(f"Target size           : {target_size}")
    print(f"Saving to             : {dataset_path.resolve()}")
    print("-" * 70)

    # ---------------------------------------------------------
    # CREATE CSV
    # ---------------------------------------------------------

    fieldnames = [
        "image_filename",
        "center_x",
        "center_y",
        "width",
        "height",
        "norm_center_x",
        "norm_center_y",
        "norm_width",
        "norm_height",
        "is_visible"
    ]

    total_frame_count = 0

    with open(csv_file_path, mode="w", newline="") as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        # -----------------------------------------------------
        # EPISODES
        # -----------------------------------------------------

        for episode in range(num_episodes):

            # -------------------------------------------------
            # Create a fresh simulator
            # -------------------------------------------------

            ep_config = json.loads(json.dumps(config))

            # Keep disturbances OFF for the first clean model.
            ep_config["disturbance"]["camera_jitter"] = 0.0
            ep_config["disturbance"]["noise_level"] = 0.0

            sim = Simulator(ep_config)

            # Camera position remains fixed during dataset
            # generation.
            camera_x = sim.camera.x
            camera_y = sim.camera.y

            # Keep the entire target comfortably inside the
            # camera frame.
            margin = 40

            min_x = camera_x + margin
            max_x = camera_x + img_width - margin

            min_y = camera_y + margin
            max_y = camera_y + img_height - margin

            # -------------------------------------------------
            # Generate frames
            # -------------------------------------------------

            for _ in range(frames_per_episode):

                # -------------------------------------------------
                # RANDOMIZE TARGET POSITION FOR EVERY FRAME
                # -------------------------------------------------

                target_x_world = random.uniform(
                    min_x,
                    max_x
                )

                target_y_world = random.uniform(
                    min_y,
                    max_y
                )

                sim.target.set_position(
                    target_x_world,
                    target_y_world
                )

                # No motion between frames.
                # Each frame is an independent spatial sample.
                sim.target.set_velocity(0.0, 0.0)

                # -------------------------------------------------
                # RENDER IMAGE
                # -------------------------------------------------

                frame = sim.get_frame()

                # -------------------------------------------------
                # GET GROUND TRUTH
                # -------------------------------------------------

                gt = sim.get_ground_truth()

                target_x = float(gt["x"])
                target_y = float(gt["y"])

                # -------------------------------------------------
                # VISIBILITY
                # -------------------------------------------------

                is_visible = (
                    0 <= target_x < img_width
                    and
                    0 <= target_y < img_height
                )

                # -------------------------------------------------
                # SAVE IMAGE
                # -------------------------------------------------

                frame_filename = (
                    f"frame_{total_frame_count:05d}.png"
                )

                img_save_path = images_path / frame_filename

                cv2.imwrite(
                    str(img_save_path),
                    frame
                )

                # -------------------------------------------------
                # BOUNDING BOX
                # -------------------------------------------------

                bbox_w = target_size
                bbox_h = target_size

                # Normalize center coordinates to [0, 1]
                norm_cx = target_x / img_width
                norm_cy = target_y / img_height

                norm_w = bbox_w / img_width
                norm_h = bbox_h / img_height

                # -------------------------------------------------
                # WRITE LABEL
                # -------------------------------------------------

                writer.writerow({
                    "image_filename": frame_filename,

                    "center_x": round(target_x, 2),
                    "center_y": round(target_y, 2),

                    "width": bbox_w,
                    "height": bbox_h,

                    "norm_center_x": round(norm_cx, 4),
                    "norm_center_y": round(norm_cy, 4),

                    "norm_width": round(norm_w, 4),
                    "norm_height": round(norm_h, 4),

                    "is_visible": 1 if is_visible else 0
                })

                total_frame_count += 1

            print(
                f"Episode {episode + 1:02d}/{num_episodes:02d} "
                f"complete | "
                f"Total frames: {total_frame_count}"
            )

    # ---------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------

    print("=" * 70)

    print(
        f"SUCCESS: Generated {total_frame_count} labeled frames."
    )

    print(
        f"Dataset CSV : {csv_file_path.resolve()}"
    )

    print(
        f"Images       : {images_path.resolve()}"
    )

    print("=" * 70)


if __name__ == "__main__":
    generate_dataset(
        num_episodes=50,
        frames_per_episode=100
    )