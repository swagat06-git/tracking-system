import csv
import json
import os
import cv2

from simulation.simulator import Simulator


def main():
    # Load the same configuration used by the simulator
    with open("config/config.json", "r") as file:
        config = json.load(file)

    simulator = Simulator(config)

    fps = config["camera"]["fps"]
    width = config["camera"]["width"]
    height = config["camera"]["height"]

    duration_seconds = 30
    total_frames = int(fps * duration_seconds)

    output_dir = "videos"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "atlas_synthetic_30s.mp4"
    )

    ground_truth_path = os.path.join(
        output_dir,
        "atlas_synthetic_30s_ground_truth.csv"
    )

    # MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
        False
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not create MP4 video. "
            "Check that OpenCV has MP4 writing support."
        )

    print("Generating synthetic ATLAS video...")
    print(f"Resolution: {width}x{height}")
    print(f"FPS:        {fps}")
    print(f"Duration:   {duration_seconds} seconds")
    print(f"Frames:     {total_frames}")
    print(f"Video:      {output_path}")
    print(f"Ground truth: {ground_truth_path}")
    print()

    with open(
        ground_truth_path,
        "w",
        newline=""
    ) as gt_file:

        gt_writer = csv.writer(gt_file)

        gt_writer.writerow([
            "frame",
            "timestamp_seconds",
            "ground_truth_x",
            "ground_truth_y"
        ])

        for frame_number in range(total_frames):

            # Advance target motion
            simulator.update()

            # Generate the exact camera frame
            frame = simulator.get_frame()

            # Get the target position in camera/image coordinates.
            # This is the ground truth corresponding to this frame.
            ground_truth = simulator.get_ground_truth()

            # Simulator produces a grayscale frame.
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )

            writer.write(frame)

            gt_writer.writerow([
                frame_number + 1,
                (frame_number) / fps,
                ground_truth["x"],
                ground_truth["y"]
            ])

            if (frame_number + 1) % int(fps) == 0:
                elapsed = (frame_number + 1) / fps

                print(
                    f"Generated {frame_number + 1}/{total_frames} "
                    f"frames ({elapsed:.0f}s)"
                )

    writer.release()

    print()
    print("Video generation complete.")
    print(f"Saved to: {output_path}")
    print(f"Ground truth saved to: {ground_truth_path}")

    # Verify the generated video
    cap = cv2.VideoCapture(output_path)

    if not cap.isOpened():
        raise RuntimeError(
            "MP4 was created but could not be reopened."
        )

    generated_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    generated_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    generated_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    generated_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    cap.release()

    print()
    print("Verification")
    print("------------")
    print(f"Frames:      {generated_frames}")
    print(f"FPS:         {generated_fps:.2f}")
    print(f"Resolution:  {generated_width}x{generated_height}")
    print(
        f"Duration:    "
        f"{generated_frames / generated_fps:.2f}s"
    )

    # Verify ground-truth row count
    with open(
        ground_truth_path,
        "r",
        newline=""
    ) as gt_file:

        ground_truth_rows = sum(
            1 for _ in gt_file
        ) - 1

    print(
        f"GT rows:     {ground_truth_rows}"
    )


if __name__ == "__main__":
    main()