class CameraController:
    """
    Converts target position error into camera
    pan and tilt commands.
    """

    def __init__(
        self,
        frame_width=640,
        frame_height=480,
        max_pan_speed=5.0,
        max_tilt_speed=5.0,
        gain=5.0
    ):

        self.frame_width = frame_width
        self.frame_height = frame_height

        # Camera center
        self.center_x = frame_width / 2
        self.center_y = frame_height / 2

        # Maximum allowed speeds
        self.max_pan_speed = max_pan_speed
        self.max_tilt_speed = max_tilt_speed

        # Controller sensitivity
        self.gain = gain

    def compute_command(self, target_x, target_y):
        """
        Calculate pan and tilt commands based on
        target position in image coordinates.
        """

        # Calculate error from camera center
        error_x = target_x - self.center_x
        error_y = target_y - self.center_y

        # Proportional control
        pan_command = self.gain * error_x
        tilt_command = self.gain * error_y

        # Limit pan speed
        pan_command = max(
            -self.max_pan_speed,
            min(pan_command, self.max_pan_speed)
        )

        # Limit tilt speed
        tilt_command = max(
            -self.max_tilt_speed,
            min(tilt_command, self.max_tilt_speed)
        )

        return {
            "pan_speed": pan_command,
            "tilt_speed": tilt_command,
            "error_x": error_x,
            "error_y": error_y
        }