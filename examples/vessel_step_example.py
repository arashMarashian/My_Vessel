import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine_loader import load_engines_from_yaml
from energy.vessel_energy_system import Battery, VesselEnergySystem


def main() -> None:
    engines = load_engines_from_yaml("data/engine_data.yaml")
    # Duplicate to create four engines for the example
    engines = engines * 2
    battery = Battery(capacity_kwh=10000.0, soc_kwh=5000.0)
    vessel = VesselEnergySystem(engines, battery)

    env = {"wind_speed": 5.0, "wind_angle_diff": 30.0, "wave_height": 1.0}
    controller_action = {
        "engine_loads": [50, 75, 0, 0],
        "battery_power": -200000.0,  # discharging in watts
        "target_speed": 8.0,
    }

    result = vessel.step(controller_action, env, timestep_hours=1.0)
    print(result)


if __name__ == "__main__":
    main()
