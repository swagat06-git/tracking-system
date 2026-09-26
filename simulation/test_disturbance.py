import os
import sys
import json
import numpy as np


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
# LOAD BASE CONFIGURATION
# -----------------------------------------

config_path = os.path.join(
    PROJECT_ROOT,
    "config",
    "config.json"
)

with open(config_path, "r") as file:
    base_config = json.load(file)


# =========================================
# TEST 1 — CLEAN FRAME
# =========================================

clean_config = json.loads(json.dumps(base_config))

clean_config["disturbance"]["noise_type"] = "none"
clean_config["disturbance"]["noise_level"] = 0
clean_config["disturbance"]["camera_jitter"] = 0

clean_simulator = Simulator(clean_config)

clean_frame = clean_simulator.get_frame()

print("Test 1 - Clean frame:")
print("Frame shape:", clean_frame.shape)

assert clean_frame.shape == (480, 640)


# =========================================
# TEST 2 — GAUSSIAN NOISE
# =========================================

noise_config = json.loads(json.dumps(base_config))

noise_config["disturbance"]["noise_type"] = "gaussian"
noise_config["disturbance"]["noise_level"] = 20
noise_config["disturbance"]["camera_jitter"] = 0

clean_simulator = Simulator(clean_config)
noisy_simulator = Simulator(noise_config)

clean_frame = clean_simulator.get_frame()
noisy_frame = noisy_simulator.get_frame()

difference = np.mean(
    np.abs(
        clean_frame.astype(np.float32)
        - noisy_frame.astype(np.float32)
    )
)

print("\nTest 2 - Gaussian noise:")
print("Mean pixel difference:", difference)

assert difference > 0


# =========================================
# TEST 3 — CAMERA JITTER
# =========================================

jitter_config = json.loads(json.dumps(base_config))

jitter_config["disturbance"]["noise_type"] = "none"
jitter_config["disturbance"]["noise_level"] = 0
jitter_config["disturbance"]["camera_jitter"] = 10

jitter_simulator = Simulator(jitter_config)

# Save commanded camera position
position_before = jitter_simulator.camera.get_position()

# Generate frames with temporary jitter
frame1 = jitter_simulator.get_frame()
frame2 = jitter_simulator.get_frame()

# Camera position must NOT permanently change
position_after = jitter_simulator.camera.get_position()

print("\nTest 3 - Camera jitter:")
print("Camera position before:", position_before)
print("Camera position after :", position_after)

assert position_before == position_after


# =========================================
# FINAL RESULT
# =========================================

print("\nAll disturbance tests passed!")