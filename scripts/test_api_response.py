import json

from simulation.simulator import Simulator
from tracking.system import TrackingSystem
from api.response import create_tracking_response


def main():

    with open("config/config.json", "r") as file:
        config = json.load(file)

    simulator = Simulator(config)

    tracking_system = TrackingSystem(
        config=config,
        acceleration_noise=500.0,
        gain=4.0,
        velocity_scale=0.5
    )

    simulator.update()

    frame = simulator.get_frame()

    result = tracking_system.process(frame)

    response = create_tracking_response(result)

    print("\nFRONTEND API RESPONSE")
    print("=" * 50)

    print(response)


if __name__ == "__main__":
    main()