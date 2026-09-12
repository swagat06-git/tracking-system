import os
import sys
import json
import cv2


# -----------------------------------------
# PROJECT ROOT
# -----------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)


# -----------------------------------------
# IMPORT SIMULATOR
# -----------------------------------------

from simulation.simulator import Simulator


# -----------------------------------------
# LOAD CONFIGURATION
# -----------------------------------------

config_path = os.path.join(
    PROJECT_ROOT,
    "config",
    "config.json"
)

with open(config_path, "r") as file:
    config = json.load(file)


# -----------------------------------------
# CREATE SIMULATOR
# -----------------------------------------

simulator = Simulator(config)

print("Camera position:", simulator.camera.get_position())

# -----------------------------------------
# SIMULATION LOOP
# -----------------------------------------

while True:

    # Advance simulation by one frame
    simulator.update()

    # Get the official 640x480 camera frame
    frame = simulator.get_frame()

    # Get ground truth
    ground_truth = simulator.get_ground_truth()

    # Verify frame dimensions
    print("Frame shape:", frame.shape)

    # Print ground truth
    print("Ground truth:", ground_truth)

    # Show camera frame
    cv2.imshow(
        "FSOC Camera View",
        frame
    )

    # Exit with q
    key = cv2.waitKey(
        int(1000 / config["camera"]["fps"])
    ) & 0xFF

    if key == ord("q"):
        break


# -----------------------------------------
# CLEANUP
# -----------------------------------------

cv2.destroyAllWindows()