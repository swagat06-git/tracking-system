import random


class Camera:
    def __init__(
        self,
        width,
        height,
        world_width,
        world_height,
        max_pan_speed=5.0,
        max_tilt_speed=5.0,
        camera_jitter=0
    ):
        self.width = width
        self.height = height

        self.world_width = world_width
        self.world_height = world_height

        # Maximum allowed camera position
        self.max_x = world_width - width
        self.max_y = world_height - height

        # Maximum movement speeds
        self.max_pan_speed = max_pan_speed
        self.max_tilt_speed = max_tilt_speed

        # Maximum temporary camera jitter in pixels
        self.camera_jitter = camera_jitter

        # Start camera at the center of the world
        self.x = (world_width - width) / 2
        self.y = (world_height - height) / 2

    def get_frame(self, world_frame):
        """Return the camera's view of the world with optional jitter."""

        # Generate temporary random jitter
        jitter_x = random.uniform(
            -self.camera_jitter,
            self.camera_jitter
        )

        jitter_y = random.uniform(
            -self.camera_jitter,
            self.camera_jitter
        )

        # Temporary rendered camera position
        rendered_x = self.x + jitter_x
        rendered_y = self.y + jitter_y

        # Keep rendered position inside world boundaries
        rendered_x = max(
            0,
            min(rendered_x, self.max_x)
        )

        rendered_y = max(
            0,
            min(rendered_y, self.max_y)
        )

        # Convert position to integers for NumPy slicing
        x = int(rendered_x)
        y = int(rendered_y)

        frame = world_frame[
            y:y + self.height,
            x:x + self.width
        ]

        return frame

    def get_position(self):
        """Return the commanded camera position in world coordinates."""

        return self.x, self.y

    def move(self, pan_speed, tilt_speed, dt):
        """
        Move the camera.

        Positive pan  = right
        Negative pan  = left

        Positive tilt = down
        Negative tilt = up
        """

        # Limit pan speed
        pan_speed = max(
            -self.max_pan_speed,
            min(pan_speed, self.max_pan_speed)
        )

        # Limit tilt speed
        tilt_speed = max(
            -self.max_tilt_speed,
            min(tilt_speed, self.max_tilt_speed)
        )

        # Update commanded position
        self.x += pan_speed * dt
        self.y += tilt_speed * dt

        # Keep camera inside world boundaries
        self.x = max(0, min(self.x, self.max_x))
        self.y = max(0, min(self.y, self.max_y))