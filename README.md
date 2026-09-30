````markdown
# ATLAS Tracking System

## AI-Based Virtual Camera Tracking System for FSOC

This repository contains the backend, simulation environment, computer vision pipeline, machine learning detector, Kalman tracker, camera controller, and benchmarking system for ATLAS.

ATLAS is an AI-based virtual camera tracking system developed for coarse alignment of mobile Free Space Optical Communication (FSOC) terminals.

The system autonomously detects a moving beacon, tracks its position, estimates its motion, and controls a virtual camera to keep the target within the camera field of view.

## Project

ATLAS stands for:

**Autonomous Target Localization and Acquisition System**

The project is developed for the Smart India Hackathon problem statement:

**Development of an AI-Based Virtual Camera Tracking System for Coarse Alignment of Mobile Free Space Optical Communication (FSOC) Terminals**

The system focuses on the coarse alignment stage of a Pointing, Acquisition and Tracking (PAT) system.

## Repository

GitHub Repository:

https://github.com/swagat06-git/tracking-system

Live Frontend:

https://atlas-henna-nu.vercel.app/

Backend API:

https://atlas-2ejd.onrender.com/

API Documentation:

https://atlas-2ejd.onrender.com/docs

## System Architecture

The tracking pipeline is structured as:

```text
Virtual Scene
      |
      v
Target Motion
      |
      v
Virtual Camera
      |
      v
Camera Frame
      |
      v
ML Target Detector
      |
      v
Kalman Tracker
      |
      v
Camera Controller
      |
      v
Virtual Camera Update
      |
      +--------------------+
                           |
                           v
                    Performance Metrics
````

The core software pipeline is:

```text
Simulator
    |
    v
ML Detector
    |
    v
Kalman Tracker
    |
    v
Camera Controller
    |
    v
Camera Update
```

## Main Components

### 1. Virtual Simulation Environment

The simulator generates a configurable virtual environment containing a moving beacon target and a movable virtual camera.

The simulator supports different target motion models:

* Linear
* Circular
* Figure-8
* Random

The environment can also introduce configurable disturbances for robustness testing.

### 2. Virtual Camera

The virtual camera generates image frames from the simulated scene.

The camera supports configurable:

* Resolution
* Field of view
* Position
* Pan movement
* Tilt movement
* Camera jitter

Camera jitter can be used to simulate platform and camera movement.

### 3. Target Detection

The tracking system contains an AI-based target detector implemented using PyTorch.

The detector processes monochrome camera frames and estimates the target position.

The system combines lightweight image processing and a CNN-based detector to provide target localization.

The detector also includes jump-gating logic to reject unrealistic frame-to-frame target movements.

### 4. Kalman Tracker

The detected target coordinates are passed to a Kalman tracking stage.

The Kalman filter provides:

* Smoothed target position
* Motion estimation
* Improved stability under noisy detections
* Prediction during temporary detection uncertainty

The tracker provides the camera controller with a more stable target position than raw detection alone.

### 5. Camera Controller

The camera controller converts the target's position error into camera pan and tilt commands.

The controller attempts to keep the target near the center of the camera frame.

Configurable parameters include:

* Maximum pan speed
* Maximum tilt speed
* Controller gain
* Velocity scaling

### 6. Disturbance Simulation

The simulator supports several image and environmental disturbances.

#### Noise

Supported noise models include:

* None
* Gaussian
* Salt-and-pepper
* Poisson

#### Atmospheric Conditions

Supported atmosphere modes include:

* Clear
* Haze
* Fog
* Rain
* Low light

#### Camera Disturbance

Camera jitter can be configured to simulate movement of the camera or platform.

## Benchmarking

The repository includes a benchmarking system for evaluating the tracking pipeline.

The benchmark measures:

* Processing FPS
* Average processing time
* Maximum processing time
* Acquisition time
* Detection rate
* Lock retention
* Target loss
* Average centroid error
* Maximum centroid error
* RMSE

The benchmarking system supports both verified benchmark results and configurable simulation scenarios.

## Verified Benchmark

The current verified synthetic-video benchmark contains:

| Metric                  |        Result |
| ----------------------- | ------------: |
| Resolution              |     640 × 480 |
| Frame Rate              |        30 FPS |
| Frames                  |           900 |
| Duration                |    30 seconds |
| Processing FPS          |   1318.77 FPS |
| Average Processing Time |       0.76 ms |
| Maximum Processing Time |      96.51 ms |
| Acquisition Time        | 0.033 seconds |
| Detection Rate          |        97.11% |
| Lock Retention          |          100% |
| Target Loss             |            0% |
| Average Centroid Error  |       4.33 px |
| Maximum Centroid Error  |      47.38 px |
| RMSE                    |       5.51 px |

The offline benchmark processing rate should not be interpreted as the processing rate of the deployed web service. The offline benchmark does not include the same network, upload, deployment, and request-handling overhead.

## Video Benchmarking

The backend provides an endpoint for uploading MP4 videos for tracking evaluation.

The video benchmark validates the input video and processes it frame-by-frame through the tracking pipeline.

The current service supports videos with:

* MP4 format
* Maximum file size of 100 MB
* Maximum duration of 30 seconds
* Resolution of 640 × 480
* Frame rate within the supported benchmark range

Endpoint:

```http
POST /benchmark/video
```

## Scenario Benchmarking

The repository also provides configurable scenario benchmarking.

A scenario can be configured using:

```text
Motion
Atmosphere
Noise Type
Noise Level
```

Example supported motion values:

```text
linear
circular
figure8
random
```

Example atmosphere values:

```text
clear
haze
fog
rain
low_light
```

Example noise values:

```text
none
gaussian
salt_pepper
poisson
```

Endpoint:

```http
POST /benchmark/scenario
```

The scenario system allows different tracking conditions to be evaluated without manually modifying the simulation code.

## API

The backend is implemented using FastAPI.

### Health Check

```http
GET /health
```

Returns the current API health status.

### Live Tracking

```http
GET /tracking
```

Updates the simulator, processes the current frame, generates tracking data, and updates the virtual camera.

### Benchmark Results

```http
GET /benchmark
```

Returns the latest verified benchmark results.

### Run Benchmark

```http
POST /benchmark
```

Returns the verified benchmark result.

### Video Benchmark

```http
POST /benchmark/video
```

Accepts an uploaded MP4 video and processes it through the tracking pipeline.

### Scenario Benchmark

```http
POST /benchmark/scenario
```

Runs a configurable simulation benchmark.

### API Documentation

FastAPI automatically provides interactive API documentation at:

[https://atlas-2ejd.onrender.com/docs](https://atlas-2ejd.onrender.com/docs)

## Configuration

System parameters are configured through:

```text
config/config.json
```

Configuration includes:

* Canvas dimensions
* Camera resolution
* Camera field of view
* Target size
* Target brightness
* Motion type
* Noise type
* Noise level
* Atmospheric condition
* Camera jitter
* Maximum pan speed
* Maximum tilt speed
* Controller gain

Example configuration structure:

```json
{
  "canvas": {
    "width": 2000,
    "height": 2000
  },
  "camera": {
    "width": 640,
    "height": 480
  },
  "target": {
    "size": 10
  },
  "motion": {
    "type": "linear"
  }
}
```

The exact configuration available in the repository should be treated as the authoritative configuration for a given deployment.

## Project Structure

```text
tracking-system/
|
├── api/
│   ├── main.py
│   ├── benchmark.py
│   ├── scenario_benchmark.py
│   ├── video_benchmark.py
│   └── response.py
|
├── simulation/
│   ├── __init__.py
│   ├── camera.py
│   ├── motion.py
│   ├── scene.py
│   ├── target.py
│   ├── simulator.py
│   └── atmosphere.py
|
├── tracking/
│   ├── ml_detector.py
│   ├── kalman.py
│   ├── camera_controller.py
│   └── system.py
|
├── config/
│   └── config.json
|
├── scripts/
|
├── videos/
|
├── requirements.txt
├── Dockerfile
└── README.md
```

## Technology Stack

### Programming Language

* Python

### Backend

* FastAPI
* Uvicorn

### Computer Vision

* OpenCV
* NumPy
* SciPy

### Machine Learning

* PyTorch
* Torchvision

### Data Processing

* Pandas
* NumPy

### Deployment

* Render

## Installation

Clone the repository:

```bash
git clone https://github.com/swagat06-git/tracking-system.git
```

Enter the project directory:

```bash
cd tracking-system
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Locally

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Start the FastAPI server:

```bash
python -m uvicorn api.main:app --reload
```

The local API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

## Machine Learning Pipeline

The ML pipeline consists of:

```text
Input Frame
     |
     v
Grayscale Conversion
     |
     v
Target Detection
     |
     v
CNN / Image Processing
     |
     v
Coordinate Prediction
     |
     v
Jump Validation
     |
     v
Kalman Filtering
     |
     v
Camera Control
```

The detector is designed to remain lightweight enough for real-time tracking.

On Apple Silicon systems, the implementation can use the Metal Performance Shaders backend when available.

CPU execution is also optimized using controlled PyTorch threading and inference mode.

## Performance Optimization

Several optimizations are used in the tracking pipeline:

* Lightweight CNN architecture
* PyTorch inference mode
* MPS acceleration on compatible Apple Silicon systems
* CUDA support where available
* CPU thread optimization
* Efficient NumPy operations
* OpenCV-based image processing
* Detection jump gating
* Frame-by-frame processing without unnecessary intermediate data
* Lightweight benchmark execution

## Deployment

The backend is deployed using Render.

Backend:

[https://atlas-2ejd.onrender.com/](https://atlas-2ejd.onrender.com/)

The frontend application communicates with the deployed API for live tracking and benchmark functionality.

## Development

Create a feature branch before making major changes:

```bash
git checkout -b feature/your-feature
```

After making changes:

```bash
git status
git add .
git commit -m "Describe your changes"
git push origin feature/your-feature
```

Changes can then be reviewed and merged into the appropriate integration or production branch.

## Testing

The repository contains scripts for testing different parts of the system.

Examples include:

```text
scripts/
```

Tests can be used to validate:

* Target detection
* Kalman tracking
* Simulation
* Disturbance handling
* ML tracking
* Benchmark processing

## Limitations

The current implementation is primarily a virtual and synthetic tracking system.

The atmospheric models are simplified approximations intended for software-level robustness testing rather than physically rigorous FSOC propagation simulation.

The current uploaded-video benchmark does not automatically calculate centroid accuracy unless corresponding ground-truth data is available.

The current system focuses on coarse alignment and does not implement a complete fine-alignment optical tracking loop.

Standalone executable packaging is also a future release deliverable.

## Future Work

Potential future improvements include:

* Real camera integration
* Physical PTZ camera integration
* Hardware-in-the-loop testing
* More realistic atmospheric propagation models
* More realistic platform motion models
* Improved target detection architectures
* Automatic ground-truth generation
* Multi-target tracking
* Fine alignment integration
* Standalone executable packaging
* Extended benchmark datasets
* GPU-enabled cloud deployment

## Project Objective

The objective of this repository is to provide a modular and configurable software implementation of an AI-based virtual camera tracking system for coarse FSOC terminal alignment.

The system combines:

```text
Simulation
+
Computer Vision
+
Machine Learning
+
Kalman Tracking
+
Camera Control
+
Benchmarking
```

to provide an end-to-end target acquisition and tracking pipeline.

## Links

ATLAS Live Application:

[https://atlas-henna-nu.vercel.app/](https://atlas-henna-nu.vercel.app/)

Backend API:

[https://atlas-2ejd.onrender.com/](https://atlas-2ejd.onrender.com/)

API Documentation:

[https://atlas-2ejd.onrender.com/docs](https://atlas-2ejd.onrender.com/docs)

GitHub Repository:

[https://github.com/swagat06-git/tracking-system](https://github.com/swagat06-git/tracking-system)
