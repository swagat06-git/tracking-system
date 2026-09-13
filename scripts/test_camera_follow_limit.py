import json

from simulation.simulator import Simulator
from tracking.pipeline import TrackingPipeline
from control.controller import CameraController


def main():

    # -------------------------------------------------
    # Load configuration
    # -------------------------------------------------

    with open("config/config.json", "r") as file:
        config = json.load(file)

    # -------------------------------------------------
    # Create system
    # -------------------------------------------------

    simulator = Simulator(config)

    # Deliberately use a target faster than the camera.
    simulator.target.set_velocity(200.0, 100.0)

    pipeline = TrackingPipeline()

    controller = CameraController(
        frame_width=config["camera"]["width"],
        frame_height=config["camera"]["height"],
        max_pan_speed=config["control"]["max_pan_speed"],
        max_tilt_speed=config["control"]["max_tilt_speed"]
    )

    total_frames = 180

    detection_frames = 0
    out_of_view_frames = 0
    detection_failures = 0

    first_out_of_view_frame = None

    print("Camera Follow Limit Test")
    print("=" * 70)

    print(
        f"Target velocity  : "
        f"({simulator.target.vx:.1f}, "
        f"{simulator.target.vy:.1f}) px/s"
    )

    print(
        f"Camera max speed : "
        f"({config['control']['max_pan_speed']:.1f}, "
        f"{config['control']['max_tilt_speed']:.1f}) px/s"
    )

    # -------------------------------------------------
    # Simulation
    # -------------------------------------------------

    for frame_number in range(total_frames):

        simulator.update()

        frame = simulator.get_frame()

        ground_truth = simulator.get_ground_truth()

        target_x = ground_truth["x"]
        target_y = ground_truth["y"]

        camera_width = config["camera"]["width"]
        camera_height = config["camera"]["height"]

        inside_camera = (
            0 <= target_x < camera_width
            and
            0 <= target_y < camera_height
        )

        # -------------------------------------------------
        # Tracking
        # -------------------------------------------------

        result = pipeline.process(frame)

        if not result["detected"]:

            if inside_camera:
                detection_failures += 1
            else:
                out_of_view_frames += 1

                if first_out_of_view_frame is None:
                    first_out_of_view_frame = frame_number

            continue

        detection_frames += 1

        # -------------------------------------------------
        # Camera control
        # -------------------------------------------------

        estimated = result["position"]

        command = controller.compute_command(
            estimated["x"],
            estimated["y"]
        )

        simulator.move_camera(
            command["pan_speed"],
            command["tilt_speed"]
        )

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        if frame_number % 20 == 0:

            camera_x, camera_y = simulator.camera.get_position()

            print(
                f"Frame {frame_number:03d} | "
                f"Camera: ({camera_x:.1f}, {camera_y:.1f}) | "
                f"Target: ({target_x:.1f}, {target_y:.1f}) | "
                f"Visible: {inside_camera}"
            )

    # -------------------------------------------------
    # Results
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"Total frames             : {total_frames}")
    print(f"Detected frames          : {detection_frames}")
    print(f"Out-of-view frames       : {out_of_view_frames}")
    print(f"Detection failures       : {detection_failures}")

    if first_out_of_view_frame is not None:
        print(
            f"First out-of-view frame  : "
            f"{first_out_of_view_frame}"
        )
    else:
        print("First out-of-view frame  : None")

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    print("\n" + "=" * 70)

    # This is an intentional stress test.
    #
    # Because the target is much faster than the camera,
    # going out of view is expected.
    #
    # The important requirements are:
    #
    # 1. No detection failures while the target is visible.
    # 2. Out-of-view is correctly distinguished from detection failure.
    # 3. Camera remains inside world boundaries.

    camera_x, camera_y = simulator.camera.get_position()

    max_camera_x = (
        config["canvas"]["width"]
        - config["camera"]["width"]
    )

    max_camera_y = (
        config["canvas"]["height"]
        - config["camera"]["height"]
    )

    camera_inside_world = (
        0 <= camera_x <= max_camera_x
        and
        0 <= camera_y <= max_camera_y
    )

    if (
        detection_failures == 0
        and out_of_view_frames > 0
        and camera_inside_world
    ):
        print(
            "PASS: Camera follow limit handled correctly."
        )
        print(
            "- Target outran the camera as expected."
        )
        print(
            "- No false detection failures occurred."
        )
        print(
            "- Camera remained inside world boundaries."
        )

    else:
        print(
            "FAIL: Camera follow limit test did not "
            "behave as expected."
        )

        if detection_failures > 0:
            print(
                f"- {detection_failures} detection "
                f"failures occurred while target was visible."
            )

        if out_of_view_frames == 0:
            print(
                "- Target never became out of view."
            )

        if not camera_inside_world:
            print(
                "- Camera exceeded world boundaries."
            )


if __name__ == "__main__":
    main()