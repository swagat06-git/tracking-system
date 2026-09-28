import json
import time

from simulation.simulator import Simulator
from tracking.system import TrackingSystem


def main():
    with open("config/config.json", "r") as f:
        config = json.load(f)

    simulator = Simulator(config)

    tracking_system = TrackingSystem(
        config=config,
        acceleration_noise=500.0,
        gain=4.0,
        velocity_scale=0.5,
        max_missed_frames=10,
    )

    fps = config["camera"]["fps"]
    dt = 1.0 / fps

    warmup_frames = 30

    # Long enough to exceed max_missed_frames = 10.
    lost_frames = 30

    recovery_frames = 60

    print("=" * 60)
    print("ATLAS RE-ACQUISITION TEST")
    print("=" * 60)
    print(f"FPS                 : {fps}")
    print(f"Warmup              : {warmup_frames / fps:.2f} sec")
    print(f"Forced detector loss: {lost_frames / fps:.2f} sec")
    print(f"Recovery window     : {recovery_frames / fps:.2f} sec")
    print(
        f"Max missed frames   : "
        f"{tracking_system.max_missed_frames}"
    )
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. Establish normal tracking
    # ------------------------------------------------------------

    print("\nEstablishing target lock...")

    for _ in range(warmup_frames):

        simulator.update()

        frame = simulator.get_frame()

        result = tracking_system.process(frame)

        command = result["command"]

        simulator.move_camera(
            command["pan_speed"],
            command["tilt_speed"],
        )

        time.sleep(dt)

    print("Initial tracking established.")

    # ------------------------------------------------------------
    # 2. Force detector failure
    #
    # IMPORTANT:
    # We temporarily replace detector.detect() with a function
    # that returns None. This simulates a genuine detector miss
    # without modifying production detector code.
    # ------------------------------------------------------------

    print("\nForcing detector loss...")

    original_detect = tracking_system.detector.detect

    tracking_system.detector.detect = lambda frame: None

    lost_confirmed = False
    lost_frame_number = None

    loss_start = time.perf_counter()

    for frame_number in range(lost_frames):

        simulator.update()

        frame = simulator.get_frame()

        result = tracking_system.process(frame)

        command = result["command"]

        simulator.move_camera(
            command["pan_speed"],
            command["tilt_speed"],
        )

        if not result["tracking"]:

            lost_confirmed = True
            lost_frame_number = frame_number + 1

            print(
                f"Tracker declared LOST after "
                f"{lost_frame_number} missed frames."
            )

            break

        time.sleep(dt)

    loss_duration = time.perf_counter() - loss_start

    # ------------------------------------------------------------
    # 3. Restore detector
    # ------------------------------------------------------------

    tracking_system.detector.detect = original_detect

    if not lost_confirmed:

        print()
        print(
            "ERROR: Tracker never entered tracking=False."
        )

        print(
            "The production loss-state logic was not reached."
        )

        print("\n" + "=" * 60)
        print("RE-ACQUISITION TEST ABORTED")
        print("=" * 60)

        return

    # ------------------------------------------------------------
    # 4. Restore target observation and measure reacquisition
    # ------------------------------------------------------------

    print("\nDetector restored.")
    print("Measuring reacquisition...")

    restore_start = time.perf_counter()

    reacquired = False
    reacquisition_time = None
    reacquired_frame = None

    for frame_number in range(recovery_frames):

        simulator.update()

        frame = simulator.get_frame()

        result = tracking_system.process(frame)

        if result["detected"] and result["tracking"]:

            reacquisition_time = (
                time.perf_counter() - restore_start
            )

            reacquired = True
            reacquired_frame = frame_number + 1

            break

        command = result["command"]

        simulator.move_camera(
            command["pan_speed"],
            command["tilt_speed"],
        )

        time.sleep(dt)

    # ------------------------------------------------------------
    # 5. Results
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("RE-ACQUISITION RESULTS")
    print("=" * 60)

    print(
        f"Forced detector loss : "
        f"{loss_duration:.3f} sec"
    )

    print(
        f"Tracker loss confirmed : "
        f"YES"
    )

    print(
        f"Missed frames before loss : "
        f"{lost_frame_number}"
    )

    if reacquired:

        print(
            f"Re-acquisition frame : "
            f"{reacquired_frame}"
        )

        print(
            f"Re-acquisition time  : "
            f"{reacquisition_time:.3f} sec"
        )

        print(
            "Re-acquisition       : SUCCESS"
        )

        if reacquisition_time <= 1.0:

            print(
                "Requirement (<=1 sec) : PASS"
            )

        else:

            print(
                "Requirement (<=1 sec) : FAIL"
            )

    else:

        print(
            "Re-acquisition time  : NOT REACQUIRED"
        )

        print(
            "Re-acquisition       : FAIL"
        )

        print(
            "Requirement (<=1 sec) : FAIL"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()