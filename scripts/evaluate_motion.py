import math
import json
import random

from simulation.simulator import Simulator
from tracking.pipeline import TrackingPipeline


MOTION_TYPES = [
    "linear",
    "circular",
    "figure8",
    "random",
]

TOTAL_FRAMES = 60

MAX_MEAN_ERROR = 15.0
MAX_MAX_ERROR = 30.0


def evaluate_motion(config, motion_type):
    config["motion"]["type"] = motion_type

    simulator = Simulator(config)
    pipeline = TrackingPipeline()

    errors = []

    detected_frames = 0
    out_of_view_frames = 0
    detection_failure_frames = 0

    camera_width = config["camera"]["width"]
    camera_height = config["camera"]["height"]

    for _ in range(TOTAL_FRAMES):

        simulator.update()

        frame = simulator.get_frame()
        ground_truth = simulator.get_ground_truth()

        target_x = ground_truth["x"]
        target_y = ground_truth["y"]

        inside_camera = (
            0 <= target_x < camera_width
            and
            0 <= target_y < camera_height
        )

        result = pipeline.process(frame)

        if not result["detected"]:

            if inside_camera:
                detection_failure_frames += 1
            else:
                out_of_view_frames += 1

            continue

        detected_frames += 1

        estimated = result["position"]

        error = math.sqrt(
            (estimated["x"] - target_x) ** 2
            +
            (estimated["y"] - target_y) ** 2
        )

        errors.append(error)

    if errors:
        mean_error = sum(errors) / len(errors)
        max_error = max(errors)
    else:
        mean_error = float("inf")
        max_error = float("inf")

    mean_pass = mean_error <= MAX_MEAN_ERROR
    max_pass = max_error <= MAX_MAX_ERROR
    detection_pass = detection_failure_frames == 0

    passed = (
        mean_pass
        and max_pass
        and detection_pass
    )

    return {
        "motion": motion_type,
        "detected": detected_frames,
        "out_of_view": out_of_view_frames,
        "detection_failures": detection_failure_frames,
        "mean_error": mean_error,
        "max_error": max_error,
        "passed": passed,
    }


def main():

    # -----------------------------------------
    # Load configuration
    # -----------------------------------------

    with open("config/config.json", "r") as file:
        config = json.load(file)

    # Make random-motion results reproducible
    random.seed(42)

    # -----------------------------------------
    # Run evaluation
    # -----------------------------------------

    print("Motion Model Evaluation")
    print("=" * 70)

    results = []

    for motion_type in MOTION_TYPES:

        print(
            f"\nTesting motion: "
            f"{motion_type}"
        )

        result = evaluate_motion(
            config.copy(),
            motion_type
        )

        results.append(result)

        print(
            f"Detected frames       : "
            f"{result['detected']}/{TOTAL_FRAMES}"
        )

        print(
            f"Out-of-view frames    : "
            f"{result['out_of_view']}/{TOTAL_FRAMES}"
        )

        print(
            f"Detection failures    : "
            f"{result['detection_failures']}"
        )

        print(
            f"Mean position error   : "
            f"{result['mean_error']:.2f} pixels"
        )

        print(
            f"Maximum position error: "
            f"{result['max_error']:.2f} pixels"
        )

        if result["passed"]:
            print("Result                : PASS")
        else:
            print("Result                : FAIL")

    # -----------------------------------------
    # Summary
    # -----------------------------------------

    print("\n" + "=" * 70)
    print("MOTION EVALUATION SUMMARY")
    print("=" * 70)

    all_passed = True

    for result in results:

        status = "PASS" if result["passed"] else "FAIL"

        print(
            f"{result['motion']:10s} | "
            f"Mean: {result['mean_error']:6.2f}px | "
            f"Max: {result['max_error']:6.2f}px | "
            f"Failures: {result['detection_failures']:3d} | "
            f"{status}"
        )

        if not result["passed"]:
            all_passed = False

    print("\n" + "=" * 70)

    if all_passed:
        print(
            "PASS: Tracking pipeline handles "
            "all tested motion models."
        )
    else:
        print(
            "FAIL: One or more motion models "
            "did not meet baseline requirements."
        )


if __name__ == "__main__":
    main()