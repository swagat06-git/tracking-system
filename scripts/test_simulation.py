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


# -----------------------------------------
# SIMULATION LOOP
# -----------------------------------------

while True:

    # Advance simulation by one frame
    simulator.update()

    # Get complete world frame
    frame = simulator.get_world_frame()

    # Get ground truth
    ground_truth = simulator.get_ground_truth()

    # Resize only for display
    display = cv2.resize(frame, (800, 800))

    # Show simulation
    cv2.imshow(
        "FSOC Target Simulation",
        display
    )

    # Print ground truth
    print(ground_truth)

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