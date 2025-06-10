import os
import sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from energy.power_model import propulsion_power


def main() -> None:
    """Plot propulsion power for various wind speeds."""
    base_env = {"wind_speed": 5.0, "wind_angle_diff": 45.0, "wave_height": 1.0}
    speeds = np.linspace(0.0, 12.0, num=25)
    wind_cases = [0.0, 5.0, 10.0]

    plt.figure(figsize=(6, 4))
    for w in wind_cases:
        env = base_env.copy()
        env["wind_speed"] = w
        powers = [propulsion_power(env, v) for v in speeds]
        plt.plot(speeds, powers, label=f"Wind {w} m/s")

    plt.xlabel("Vessel Speed (m/s)")
    plt.ylabel("Propulsion Power (W)")
    plt.title("Propulsion Power vs Speed")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    example_power = propulsion_power(base_env, 8.0)
    print(f"Power at 8 m/s (wind {base_env['wind_speed']} m/s): {example_power:.2f} W")


if __name__ == "__main__":
    main()
