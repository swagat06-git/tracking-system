import sys
import os
import cv2

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)

from simulation.scene import Scene
from simulation.target import Target
from simulation.motion import Motion


# ----------------------------
# CONFIGURATION
# ----------------------------

CANVAS_WIDTH = 2000
CANVAS_HEIGHT = 2000

TARGET_SIZE = 10
TARGET_BRIGHTNESS = 255

FPS = 30
DT = 1 / FPS

MOTION_TYPE = "linear"


# ----------------------------
# CREATE OBJECTS
# ----------------------------

scene = Scene(CANVAS_WIDTH, CANVAS_HEIGHT)

target = Target(
    x=500,
    y=500,
    size=TARGET_SIZE,
    brightness=TARGET_BRIGHTNESS
)

motion = Motion(
    MOTION_TYPE,
    CANVAS_WIDTH,
    CANVAS_HEIGHT
)


# Initial velocity for linear motion
target.set_velocity(200, 100)


# ----------------------------
# SIMULATION LOOP
# ----------------------------

while True:

    # Update target position
    motion.update(target, DT)

    # Create a fresh canvas
    canvas = scene.create_canvas()

    # Draw target
    scene.draw_target(canvas, target)

    # Display the simulation
    cv2.imshow(
        "FSOC Target Simulation",
        canvas
    )

    # Get ground truth
    x, y = target.get_position()

    # Exit when q is pressed
    key = cv2.waitKey(int(1000 / FPS))

    if key == ord("q"):
        break


cv2.destroyAllWindows()