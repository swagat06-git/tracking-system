import math
import random


class Motion:
    def __init__(self, motion_type, width, height):
        self.motion_type = motion_type
        self.width = width
        self.height = height

        self.time = 0.0

        # Parameters for circular motion
        self.center_x = width / 2
        self.center_y = height / 2
        self.radius = 400

        # Parameters for figure-8 motion
        self.figure8_scale = 400

    def update(self, target, dt):
        """Update the target position based on motion type."""

        if self.motion_type == "linear":
            self.linear_motion(target, dt)

        elif self.motion_type == "circular":
            self.circular_motion(target, dt)

        elif self.motion_type == "figure8":
            self.figure8_motion(target, dt)

        elif self.motion_type == "random":
            self.random_motion(target, dt)

        self.time += dt

    # -------------------------------------------------
    # 1. LINEAR MOTION
    # -------------------------------------------------

    def linear_motion(self, target, dt):
        x, y = target.get_position()
        vx, vy = target.get_velocity()

        new_x = x + vx * dt
        new_y = y + vy * dt

        # Keep target inside the world
        new_x = max(target.size, min(self.width - target.size, new_x))
        new_y = max(target.size, min(self.height - target.size, new_y))

        target.set_position(new_x, new_y)

    # -------------------------------------------------
    # 2. CIRCULAR MOTION
    # -------------------------------------------------

    def circular_motion(self, target, dt):
        angular_speed = 0.5

        x = self.center_x + self.radius * math.cos(
            angular_speed * self.time
        )

        y = self.center_y + self.radius * math.sin(
            angular_speed * self.time
        )

        target.set_position(x, y)

    # -------------------------------------------------
    # 3. FIGURE-8 MOTION
    # -------------------------------------------------

    def figure8_motion(self, target, dt):
        angular_speed = 0.5

        x = self.center_x + self.figure8_scale * math.sin(
            angular_speed * self.time
        )

        y = self.center_y + (
            self.figure8_scale
            * math.sin(2 * angular_speed * self.time)
            / 2
        )

        target.set_position(x, y)

    # -------------------------------------------------
    # 4. RANDOM MOTION
    # -------------------------------------------------

    def random_motion(self, target, dt):
        x, y = target.get_position()

        speed = 200

        dx = random.uniform(-1, 1) * speed * dt
        dy = random.uniform(-1, 1) * speed * dt

        new_x = x + dx
        new_y = y + dy

        # Keep target inside the canvas
        new_x = max(target.size, min(self.width - target.size, new_x))
        new_y = max(target.size, min(self.height - target.size, new_y))

        target.set_position(new_x, new_y)
        