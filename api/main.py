import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from simulation.simulator import Simulator
from tracking.system import TrackingSystem
from api.response import create_tracking_response
from api.benchmark import get_last_benchmark, run_benchmark


app = FastAPI(
    title="Target Tracking API",
    description="Backend API for the target tracking system",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:8081",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "https://atlas-henna-nu.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


with open("config/config.json", "r") as file:
    config = json.load(file)


simulator = Simulator(config)

tracking_system = TrackingSystem(
    config=config,
    acceleration_noise=50.0,
    gain=4.0,
    velocity_scale=0.5
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


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


@app.get("/benchmark")
def get_benchmark():
    return get_last_benchmark()


@app.post("/benchmark")
def create_benchmark():
    return run_benchmark()
