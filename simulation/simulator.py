from simulation.scene import Scene
from simulation.target import Target
from simulation.motion import Motion
from simulation.camera import Camera


class Simulator:

    def __init__(self, config):

        # -----------------------------------------
        # READ CONFIGURATION
        # -----------------------------------------

        self.width = config["canvas"]["width"]
        self.height = config["canvas"]["height"]

        self.target_size = config["target"]["size"]
        self.target_brightness = config["target"]["brightness"]

        self.camera_width = config["camera"]["width"]
        self.camera_height = config["camera"]["height"]

        self.fps = config["camera"]["fps"]
        self.dt = 1 / self.fps

        self.motion_type = config["motion"]["type"]

        # -----------------------------------------
        # CREATE SCENE
        # -----------------------------------------

        self.scene = Scene(
            self.width,
            self.height
        )

        # -----------------------------------------
        # CREATE TARGET
        # -----------------------------------------

        self.target = Target(
            x=self.width / 2,
            y=self.height / 2,
            size=self.target_size,
            brightness=self.target_brightness
        )

        # -----------------------------------------
        # SET INITIAL VELOCITY
        # -----------------------------------------

        self.target.set_velocity(
            200,
            100
        )

        # -----------------------------------------
        # CREATE MOTION SYSTEM
        # -----------------------------------------

        self.motion = Motion(
            self.motion_type,
            self.width,
            self.height
        )

        # -----------------------------------------
        # CREATE VIRTUAL CAMERA
        # -----------------------------------------

        self.camera = Camera(
            width=self.camera_width,
            height=self.camera_height,
            world_width=self.width,
            world_height=self.height,
            max_pan_speed=config["control"]["max_pan_speed"],
            max_tilt_speed=config["control"]["max_tilt_speed"]
        )

    # -----------------------------------------
    # UPDATE SIMULATION
    # -----------------------------------------

    def update(self):
        """Advance the simulation by one frame."""

        self.motion.update(
            self.target,
            self.dt
        )
    def move_camera(self, pan_speed, tilt_speed):
        """Move the camera using pan and tilt speeds."""

        self.camera.move(
            pan_speed,
            tilt_speed,
            self.dt
        )
    # -----------------------------------------
    # GET COMPLETE WORLD FRAME
    # -----------------------------------------

    def get_world_frame(self):
        """Return the complete simulation world."""

        canvas = self.scene.create_canvas()

        self.scene.draw_target(
            canvas,
            self.target
        )

        return canvas

    # -----------------------------------------
    # GET CAMERA FRAME
    # -----------------------------------------

    def get_frame(self):
        """Return the current 640x480 camera frame."""

        world_frame = self.get_world_frame()

        frame = self.camera.get_frame(
            world_frame
        )

        return frame

    # -----------------------------------------
    # GET CAMERA/IMAGE GROUND TRUTH
    # -----------------------------------------

    def get_ground_truth(self):
        """Return target position in camera/image coordinates."""

        target_x, target_y = self.target.get_position()

        camera_x, camera_y = self.camera.get_position()

        image_x = target_x - camera_x
        image_y = target_y - camera_y

        return {
            "x": float(image_x),
            "y": float(image_y)
        }

    # -----------------------------------------
    # GET WORLD GROUND TRUTH
    # -----------------------------------------

    def get_world_ground_truth(self):
        """Return target position in world coordinates."""

        x, y = self.target.get_position()

        return {
            "x": float(x),
            "y": float(y)
        }