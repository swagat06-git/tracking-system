import cv2


class VideoFileSource:
    """
    Reads frames from an MP4/video file and exposes them through
    the same get_frame() style interface used by the simulator.
    """

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.fps <= 0:
            self.fps = 30.0

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame(self):
        """
        Return the next video frame.

        Returns:
            frame: BGR OpenCV frame
            None: when video ends
        """
        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def reset(self):
        """Restart the video from the first frame."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        """Release the video resource."""
        self.cap.release()

    def get_info(self):
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_seconds": (
                self.frame_count / self.fps
                if self.fps > 0
                else 0
            ),
        }