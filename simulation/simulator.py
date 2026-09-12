from simulation.scene import Scene
from simulation.target import Target
from simulation.motion import Motion
from simulation.camera import Camera


class Simulator:
    def __init__(self, config):
        self.width = config["canvas"]["width"]
        self.height = config["canvas"]["height"]

        self.target_size = config["target"]["size"]
        self.target_brightness = config["target"]["brightness"]

        self.camera_width = config["camera"]["width"]
        self.camera_height = config["camera"]["height"]

        self.fps = config["camera"]["fps"]
        self.dt = 1 / self.fps

        self.motion_type = config["motion"]["type"]

        self.scene = Scene(
            self.width,
            self.height
        )

        self.target = Target(
            x=self.width / 2,
            y=self.height / 2,
            size=self.target_size,
            brightness=self.target_brightness
        )

        self.target.set_velocity(
            200,
            100
        )

        self.motion = Motion(
            self.motion_type,
            self.width,
            self.height
        )

        self.camera = Camera(
            width=self.camera_width,
            height=self.camera_height,
            world_width=self.width,
            world_height=self.height
        )

    def update(self):
        self.motion.update(
            self.target,
            self.dt
        )

    def get_world_frame(self):
        canvas = self.scene.create_canvas()

        self.scene.draw_target(
            canvas,
            self.target
        )

        return canvas

    def get_frame(self):
        world_frame = self.get_world_frame()

        frame = self.camera.get_frame(world_frame)

        return frame

    def get_ground_truth(self):
        x, y = self.target.get_position()

        return {
            "x": float(x),
            "y": float(y)
        }