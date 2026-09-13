from control.controller import CameraController


# Create controller
controller = CameraController(
    frame_width=640,
    frame_height=480,
    max_pan_speed=5.0,
    max_tilt_speed=5.0,
    gain=0.02
)


# -----------------------------------------
# TEST 1: TARGET AT CENTER
# -----------------------------------------

result = controller.compute_command(320, 240)

print("Test 1 - Target at center:")
print(result)

assert result["pan_speed"] == 0
assert result["tilt_speed"] == 0


# -----------------------------------------
# TEST 2: TARGET TO THE RIGHT
# -----------------------------------------

result = controller.compute_command(500, 240)

print("\nTest 2 - Target to the right:")
print(result)

assert result["pan_speed"] > 0
assert result["tilt_speed"] == 0


# -----------------------------------------
# TEST 3: TARGET TO THE LEFT
# -----------------------------------------

result = controller.compute_command(100, 240)

print("\nTest 3 - Target to the left:")
print(result)

assert result["pan_speed"] < 0


# -----------------------------------------
# TEST 4: SPEED LIMIT
# -----------------------------------------

result = controller.compute_command(10000, 10000)

print("\nTest 4 - Speed limit:")
print(result)

assert result["pan_speed"] == 5.0
assert result["tilt_speed"] == 5.0


print("\nAll controller tests passed!")