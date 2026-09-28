import json
import os
import subprocess
import sys
import threading
import time
from typing import Any

from fastapi import HTTPException

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VIDEO = os.path.join(PROJECT_ROOT, "videos", "atlas_synthetic_30s.mp4")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "benchmark_video")
SUMMARY_NAME = "atlas_synthetic_30s_summary.json"

_benchmark_lock = threading.Lock()


def _summary_path(video_path: str) -> str:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(OUTPUT_DIR, f"{video_name}_summary.json")


def _load_summary(video_path: str) -> dict[str, Any] | None:
    path = _summary_path(video_path)

    if not os.path.exists(path):
        return None

    with open(path, "r") as file:
        summary = json.load(file)

    return {
        "status": "completed",
        "generated_at": os.path.getmtime(path),
        "result": summary,
    }


def get_last_benchmark() -> dict[str, Any] | None:
    video_path = os.getenv("ATLAS_BENCHMARK_VIDEO", DEFAULT_VIDEO)
    return _load_summary(video_path)


def run_benchmark() -> dict[str, Any]:
    video_path = os.getenv("ATLAS_BENCHMARK_VIDEO", DEFAULT_VIDEO)

    if not os.path.exists(video_path):
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark video not found: {video_path}",
        )

    with _benchmark_lock:
        command = [
            sys.executable,
            "-m",
            "scripts.benchmark_video",
            video_path,
        ]

        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504,
                detail="Benchmark timed out after 120 seconds.",
            )

        if completed.returncode != 0:
            error_output = completed.stderr.strip() or completed.stdout.strip()
            raise HTTPException(
                status_code=500,
                detail=f"Benchmark failed: {error_output[-2000:]}",
            )

        result = _load_summary(video_path)

        if result is None:
            raise HTTPException(
                status_code=500,
                detail="Benchmark completed but no summary JSON was generated.",
            )

        result["generated_at"] = time.time()
        return result
