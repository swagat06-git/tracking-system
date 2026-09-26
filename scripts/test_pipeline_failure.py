import numpy as np

from tracking.pipeline import TrackingPipeline


def main():
    # Create an empty frame with no target
    frame = np.zeros((480, 640), dtype=np.uint8)

    pipeline = TrackingPipeline()

    result = pipeline.process(frame)

    print("Pipeline Failure Test")
    print("=" * 40)
    print("Result:")
    print(result)

    # Validate the failure response
    assert result["detected"] is False
    assert result["position"] is None
    assert result["velocity"] is None
    assert result["confidence"] == 0.0

    print("\nPASS: Pipeline handles missing target correctly.")


if __name__ == "__main__":
    main()