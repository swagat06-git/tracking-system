import time


def create_tracking_response(result):
    """
    Convert an internal TrackingSystem result into
    a frontend-friendly API response.
    """

    position = result.get("position")
    velocity = result.get("velocity")
    command = result.get("command")

    response = {
        "status": {
            "detected": result["detected"],
            "tracking": result["tracking"]
        },

        "target": {
            "x": (
                position["x"]
                if position is not None
                else None
            ),
            "y": (
                position["y"]
                if position is not None
                else None
            )
        },

        "velocity": {
            "x": (
                velocity["vx"]
                if velocity is not None
                else None
            ),
            "y": (
                velocity["vy"]
                if velocity is not None
                else None
            )
        },

        "camera": {
            "pan_speed": command["pan_speed"],
            "tilt_speed": command["tilt_speed"]
        },

        "metadata": {
            "confidence": result.get("confidence"),
            "timestamp": time.time()
        }
    }

    return response