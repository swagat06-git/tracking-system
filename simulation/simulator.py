
import numpy as np

from simulation.scene import Scene
from simulation.target import Target
from simulation.motion import Motion
from simulation.camera import Camera
from simulation.atmosphere import Atmosphere


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
        # DISTURBANCE CONFIGURATION
        # -----------------------------------------

        self.noise_type = config["disturbance"]["noise_type"]
        self.noise_level = config["disturbance"]["noise_level"]
        self.camera_jitter = config["disturbance"]["camera_jitter"]
        self.atmosphere_mode = config["disturbance"].get(
            "atmosphere",
            "clear"
        )

        self.atmosphere = Atmosphere(
            self.atmosphere_mode
        )

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
            max_tilt_speed=config["control"]["max_tilt_speed"],
            camera_jitter=self.camera_jitter
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

    # -----------------------------------------
    # MOVE CAMERA
    # -----------------------------------------

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
        world_frame = self.get_world_frame()
        frame = self.camera.get_frame(world_frame)

        # -----------------------------------------
        # GAUSSIAN NOISE
        # -----------------------------------------

        if self.noise_type == "gaussian" and self.noise_level > 0:
            noise = np.random.normal(
                loc=0,
                scale=self.noise_level,
                size=frame.shape
            )

            noisy_frame = np.clip(
                frame.astype(np.float32) + noise,
                0,
                255
            )

            frame = noisy_frame.astype(np.uint8)

        # -----------------------------------------
        # POISSON NOISE
        # -----------------------------------------

        elif self.noise_type == "poisson" and self.noise_level > 0:
            scale = 1.0 / self.noise_level

            scaled_frame = (
                frame.astype(np.float32) * scale
            )

            poisson_frame = np.random.poisson(
                scaled_frame
            ).astype(np.float32)

            poisson_frame = poisson_frame / scale

            frame = np.clip(
                poisson_frame,
                0,
                255
            ).astype(np.uint8)

        # -----------------------------------------
        # SALT & PEPPER NOISE
        # -----------------------------------------

        elif self.noise_type == "salt_pepper" and self.noise_level > 0:
            noisy_frame = frame.copy()

            probability = self.noise_level

            random_matrix = np.random.random(
                frame.shape[:2]
            )

            salt_mask = random_matrix < (probability / 2)
            pepper_mask = random_matrix > (1 - probability / 2)

            noisy_frame[salt_mask] = 255
            noisy_frame[pepper_mask] = 0

            frame = noisy_frame
        frame = self.atmosphere.apply(frame)
        return frame

    # -----------------------------------------
    # GET CAMERA/IMAGE GROUND TRUTH
    # -----------------------------------------

    def get_ground_truth(self):
        """
        Return target position in camera/image coordinates.
        """

        target_x, target_y = self.target.get_position()

        camera_x, camera_y = self.camera.get_rendered_position()

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

