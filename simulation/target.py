class Target:
    def __init__(self, x, y, size, brightness):
        # Position
        self.x = float(x)
        self.y = float(y)

        # Velocity
        self.vx = 0.0
        self.vy = 0.0

        # Appearance
        self.size = size
        self.brightness = brightness

    def get_position(self):
        """Return the true target position."""
        return self.x, self.y

    def set_position(self, x, y):
        """Update the target position."""
        self.x = float(x)
        self.y = float(y)

    def set_velocity(self, vx, vy):
        """Set the target velocity."""
        self.vx = float(vx)
        self.vy = float(vy)

    def get_velocity(self):
        """Return the current target velocity."""
        return self.vx, self.vy