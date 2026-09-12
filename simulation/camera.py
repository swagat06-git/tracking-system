class Camera:
    def __init__(
        self,
        width,
        height,
        world_width,
        world_height,
        max_pan_speed=5.0,
        max_tilt_speed=5.0
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

        # Start camera at the center of the world
        self.x = (world_width - width) / 2
        self.y = (world_height - height) / 2

    def get_frame(self, world_frame):
        """Return the camera's view of the world."""

        # Convert position to integers for NumPy slicing
        x = int(self.x)
        y = int(self.y)

        frame = world_frame[
            y:y + self.height,
            x:x + self.width
        ]

        return frame

    def get_position(self):
        """Return the camera position in world coordinates."""

        return self.x, self.y

    def move(self, pan_speed, tilt_speed, dt):
        """
        Move the camera.

        pan_speed:
            Positive = right
            Negative = left

        tilt_speed:
            Positive = down
            Negative = up

        dt:
            Time step in seconds
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

        # Update position
        self.x += pan_speed * dt
        self.y += tilt_speed * dt

        # Keep camera inside world boundaries
        self.x = max(0, min(self.x, self.max_x))
        self.y = max(0, min(self.y, self.max_y))