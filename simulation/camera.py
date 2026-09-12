class Camera:
    def __init__(self, width, height, world_width, world_height):
        self.width = width
        self.height = height

        self.world_width = world_width
        self.world_height = world_height

        # Start camera at the center of the world
        self.x = (world_width - width) // 2
        self.y = (world_height - height) // 2

    def get_frame(self, world_frame):
        """Return the camera's 640x480 view of the world."""

        frame = world_frame[
            self.y:self.y + self.height,
            self.x:self.x + self.width
        ]

        return frame

    def get_position(self):
        """Return the camera position in world coordinates."""

        return self.x, self.y