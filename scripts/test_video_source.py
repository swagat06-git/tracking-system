import sys

from simulation.video_source import VideoFileSource


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python -m scripts.test_video_source <video.mp4>")
        return

    video_path = sys.argv[1]

    source = VideoFileSource(video_path)

    print("\nVideo information")
    print("-----------------")

    info = source.get_info()

    print(f"Width:       {info['width']}")
    print(f"Height:      {info['height']}")
    print(f"FPS:         {info['fps']:.2f}")
    print(f"Frames:      {info['frame_count']}")
    print(f"Duration:    {info['duration_seconds']:.2f} seconds")

    print("\nReading frames...")

    count = 0

    while True:
        frame = source.get_frame()

        if frame is None:
            break

        count += 1

        if count <= 5:
            print(
                f"Frame {count}: "
                f"{frame.shape[1]}x{frame.shape[0]}"
            )

    source.release()

    print(f"\nSuccessfully read {count} frames.")


if __name__ == "__main__":
    main()