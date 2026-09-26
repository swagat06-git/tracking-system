import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from simulation.simulator import Simulator
from tracking.system import TrackingSystem
from api.response import create_tracking_response


app = FastAPI(
    title="Target Tracking API",
    description="Backend API for the target tracking system",
    version="1.0.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Load configuration
# --------------------------------------------------

with open("config/config.json", "r") as file:
    config = json.load(file)


# --------------------------------------------------
# Initialize tracking system
# --------------------------------------------------

simulator = Simulator(config)

tracking_system = TrackingSystem(
    config=config,
    acceleration_noise=500.0,
    gain=4.0,
    velocity_scale=0.5
)


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


# --------------------------------------------------
# Tracking endpoint
# --------------------------------------------------

@app.get("/tracking")
def get_tracking_data():

    simulator.update()

    frame = simulator.get_frame()

    result = tracking_system.process(frame)

    response = create_tracking_response(result)

    simulator.move_camera(
        response["camera"]["pan_speed"],
        response["camera"]["tilt_speed"]
    )

    return response