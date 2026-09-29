import json

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from simulation.simulator import Simulator
from tracking.system import TrackingSystem
from api.response import create_tracking_response
from api.benchmark import get_last_benchmark, run_benchmark
from api.video_benchmark import run_uploaded_video
from api.scenario_benchmark import run_scenario


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
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


with open("config/config.json", "r") as file:
    config = json.load(file)


simulator = Simulator(config)



class ScenarioRequest(BaseModel):
    motion: str = "linear"
    atmosphere: str = "clear"
    noise_type: str = "none"
    noise_level: float = Field(default=0.0, ge=0.0)

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



@app.post("/benchmark/scenario")
def create_scenario_benchmark(request: ScenarioRequest):
    try:
        return run_scenario(
            config,
            motion=request.motion,
            atmosphere=request.atmosphere,
            noise_type=request.noise_type,
            noise_level=request.noise_level,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Scenario benchmark failed safely.",
        ) from exc

@app.post("/benchmark/video")
async def create_video_benchmark(
    video: UploadFile = File(...),
):
    try:
        return await run_uploaded_video(video, config)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Uploaded video benchmark failed safely. "
            "The existing simulator benchmark is unchanged.",
        ) from exc
