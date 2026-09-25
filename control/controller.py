class CameraController:
    def __init__(
        self,
        frame_width=640,
        frame_height=480,
        max_pan_speed=5.0,
        max_tilt_speed=5.0,
        gain=2.0,
        velocity_scale=0.5
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.center_x = frame_width / 2.0
        self.center_y = frame_height / 2.0

        self.max_pan_speed = max_pan_speed
        self.max_tilt_speed = max_tilt_speed

        self.gain = gain
        self.velocity_scale = velocity_scale

    def compute_command(self, target_x, target_y):
        """
        Position-only proportional controller.
        """

        error_x = target_x - self.center_x
        error_y = target_y - self.center_y

        pan_command = self.gain * error_x
        tilt_command = self.gain * error_y

        pan_command = max(
            -self.max_pan_speed,
            min(pan_command, self.max_pan_speed)
        )

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

    def compute_velocity_command(
        self,
        target_x,
        target_y,
        target_vx,
        target_vy
    ):
        """
        Velocity-aware controller.

        Control law:

            camera_speed_x =
                velocity_scale * target_vx
                + gain * position_error_x

            camera_speed_y =
                velocity_scale * target_vy
                + gain * position_error_y

        target_vx and target_vy are expected to come
        directly from the Kalman filter.
        """

        error_x = target_x - self.center_x
        error_y = target_y - self.center_y

        feedforward_x = self.velocity_scale * target_vx
        feedforward_y = self.velocity_scale * target_vy

        feedback_x = self.gain * error_x
        feedback_y = self.gain * error_y

        pan_command = feedforward_x + feedback_x
        tilt_command = feedforward_y + feedback_y

        pan_command = max(
            -self.max_pan_speed,
            min(pan_command, self.max_pan_speed)
        )

        tilt_command = max(
            -self.max_tilt_speed,
            min(tilt_command, self.max_tilt_speed)
        )

        return {
            "pan_speed": pan_command,
            "tilt_speed": tilt_command,
            "error_x": error_x,
            "error_y": error_y,
            "target_vx": target_vx,
            "target_vy": target_vy,
            "feedforward_x": feedforward_x,
            "feedforward_y": feedforward_y,
            "feedback_x": feedback_x,
            "feedback_y": feedback_y
        }