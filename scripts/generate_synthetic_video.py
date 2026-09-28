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
    print(f"Output:     {output_path}")
    print()

    for frame_number in range(total_frames):
        # Advance target motion
        simulator.update()

        # Generate the exact camera frame from the simulator
        frame = simulator.get_frame()

        # Simulator produces a grayscale frame.
        # VideoWriter is configured for grayscale.
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

        writer.write(frame)

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

    # Verify the generated file
    cap = cv2.VideoCapture(output_path)

    if not cap.isOpened():
        raise RuntimeError(
            "MP4 was created but could not be reopened."
        )

    generated_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    generated_fps = cap.get(cv2.CAP_PROP_FPS)
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


if __name__ == "__main__":
    main()