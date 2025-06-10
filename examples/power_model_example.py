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
    """Plot propulsion power vs speed for variations in wind speed, angle, and wave height."""
    base_env = {"wind_speed": 5.0, "wind_angle_diff": 45.0, "wave_height": 1.0}
    speeds = np.linspace(0.0, 12.0, num=25)

    # Variable cases
    wind_speeds = [0.0, 5.0, 10.0]
    wind_angles = [0.0, 45.0, 90.0]
    wave_heights = [0.0, 1.0, 2.0]

    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # 1. Vary wind speed
    for ws in wind_speeds:
        env = base_env.copy()
        env["wind_speed"] = ws
        powers = [propulsion_power(env, v) for v in speeds]
        axs[0].plot(speeds, powers, label=f"Wind Speed {ws} m/s")
    axs[0].set_title("Effect of Wind Speed")
    axs[0].set_ylabel("Power (W)")
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Vary wind angle
    for angle in wind_angles:
        env = base_env.copy()
        env["wind_angle_diff"] = angle
        powers = [propulsion_power(env, v) for v in speeds]
        axs[1].plot(speeds, powers, label=f"Wind Angle {angle}°")
    axs[1].set_title("Effect of Wind Angle Difference")
    axs[1].set_ylabel("Power (W)")
    axs[1].legend()
    axs[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Vary wave height
    for wh in wave_heights:
        env = base_env.copy()
        env["wave_height"] = wh
        powers = [propulsion_power(env, v) for v in speeds]
        axs[2].plot(speeds, powers, label=f"Wave Height {wh} m")
    axs[2].set_title("Effect of Wave Height")
    axs[2].set_xlabel("Vessel Speed (m/s)")
    axs[2].set_ylabel("Power (W)")
    axs[2].legend()
    axs[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

    example_power = propulsion_power(base_env, 8.0)
    print(f"Power at 8 m/s (base env): {example_power:.2f} W")


if __name__ == "__main__":
    main()