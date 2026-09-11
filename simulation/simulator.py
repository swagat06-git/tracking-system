from simulation.scene import Scene
from simulation.target import Target
from simulation.motion import Motion


class Simulator:
    def __init__(self, config):

        # Read configuration
        self.width = config["canvas"]["width"]
        self.height = config["canvas"]["height"]

        self.target_size = config["target"]["size"]
        self.target_brightness = config["target"]["brightness"]

        self.fps = config["camera"]["fps"]
        self.dt = 1 / self.fps

        self.motion_type = config["motion"]["type"]

        # Create scene
        self.scene = Scene(
            self.width,
            self.height
        )

        # Create target
        self.target = Target(
            x=self.width / 2,
            y=self.height / 2,
            size=self.target_size,
            brightness=self.target_brightness
        )

        # Initial velocity
        self.target.set_velocity(
            200,
            100
        )

        # Create motion system
        self.motion = Motion(
            self.motion_type,
            self.width,
            self.height
        )

    def update(self):
        """Advance simulation by one frame."""

        self.motion.update(
            self.target,
            self.dt
        )

    def get_world_frame(self):
        """Return the complete 2000×2000 simulation world."""

        canvas = self.scene.create_canvas()

        self.scene.draw_target(
            canvas,
            self.target
        )

        return canvas

    def get_ground_truth(self):
        """Return the true target position."""

        x, y = self.target.get_position()

        return {
            "x": float(x),
            "y": float(y)
        }