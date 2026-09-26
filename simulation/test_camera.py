from simulation.camera import Camera


# Create camera
camera = Camera(
    width=640,
    height=480,
    world_width=2000,
    world_height=2000,
    max_pan_speed=5.0,
    max_tilt_speed=5.0
)


print("Initial position:", camera.get_position())


# -----------------------------------------
# TEST 1: NORMAL MOVEMENT
# -----------------------------------------

camera.move(
    pan_speed=5.0,
    tilt_speed=3.0,
    dt=10
)

print("After normal movement:", camera.get_position())


# -----------------------------------------
# TEST 2: SPEED LIMIT
# -----------------------------------------

camera.move(
    pan_speed=100.0,
    tilt_speed=100.0,
    dt=1
)

print("After speed-limit test:", camera.get_position())


# -----------------------------------------
# TEST 3: WORLD BOUNDARY
# -----------------------------------------

# Try to move far outside the world
camera.move(
    pan_speed=100000,
    tilt_speed=100000,
    dt=1000
)

x, y = camera.get_position()

print("After boundary test:", (x, y))


# Verify boundaries
assert 0 <= x <= 1360
assert 0 <= y <= 1520

print("\nAll camera movement tests passed!")