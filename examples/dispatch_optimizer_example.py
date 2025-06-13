import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine_loader import load_engines_from_yaml
from energy.opt_control import DispatchOptimizer
from energy import propulsion_power, hotel_power, aux_power


def main() -> None:
    """Run the dispatch optimizer for a short horizon and plot results."""
    path = os.path.join("data", "engine_data_viking_star.yaml")
    engines = load_engines_from_yaml(path)[:2]

    with open(path) as f:
        data = yaml.safe_load(f)
    curves = []
    for eng in data["engines"][:2]:
        curve = {float(k): v for k, v in eng["sfoc"]["HFO"].items()}
        curves.append(curve)
        
    horizon = 9
    random.seed(42)  # Optional: Ensures consistent results across runs

    env = [
        {
            "wind_speed": round(random.uniform(2.0, 12.0), 1),            # realistic 2–12 m/s
            "wind_angle_diff": round(random.uniform(0.0, 180.0), 1),      # 0° (tailwind) to 180° (headwind)
            "wave_height": round(random.uniform(0.5, 3.5), 2),            # mild to rough sea
        }
        for _ in range(horizon)
    ]

    opt = DispatchOptimizer(
        engines,
        horizon=horizon,
        env=env,
        dt_hours=1.0,
        battery_capacity_kwh=500.0,
        sfoc_curves=curves,
        target_distance_m=300_000.0,
    )

    solver = "ipopt"
    opt.solve(solver)
    results = opt.results()

    time = np.arange(horizon)
    speeds = results["speed"]
    distance = np.sum(speeds) * 3600
    soc = results["soc"][:-1]
    loads = np.array(results["loads"]) * 100

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(time, speeds, marker="o", label="Speed")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Speed (m/s)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(time, soc, marker="s", color="tab:orange", label="SOC")
    ax2.set_ylabel("Battery SOC", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.tight_layout()

    fig2, axs = plt.subplots(len(engines), 1, figsize=(8, 3 * len(engines)), sharex=True)
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    for i, ax in enumerate(axs):
        ax.step(time, loads[:, i], where="mid", label=engines[i].name)
        ax.set_ylabel("Load %")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
    axs[-1].set_xlabel("Timestep")
    fig2.tight_layout()

    print(f"Total distance traveled: {distance:.1f} m")


    # compute power components for plotting
    prop_power = [propulsion_power(speeds[t]) for t in range(horizon)]
    hotel = [hotel_power(env[t]) for t in range(horizon)]
    auxiliary = [aux_power(env[t], prop_power[t]) for t in range(horizon)]

    cumulative_dist = np.cumsum(speeds) * 3600

    # power components figure
    fig3, (axp, axh, axa) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axp.plot(time, prop_power, label="Propulsion", color="tab:blue")
    axp.set_ylabel("P_prop (W)")
    axh.plot(time, hotel, label="Hotel", color="tab:orange")
    axh.set_ylabel("P_hotel (W)")
    axa.plot(time, auxiliary, label="Auxiliary", color="tab:green")
    axa.set_ylabel("P_aux (W)")
    for ax in (axp, axh, axa):
        ax.grid(True, linestyle="--", alpha=0.5)
    axa.set_xlabel("Timestep")
    fig3.tight_layout()

    # environment figure
    wind = [e.get("wind_speed", 0.0) for e in env]
    angle = [e.get("wind_angle_diff", 0.0) for e in env]
    wave = [e.get("wave_height", 0.0) for e in env]
    fig4, (axw, axang, axwave) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axw.plot(time, wind, color="tab:blue")
    axw.set_ylabel("Wind (m/s)")
    axang.plot(time, angle, color="tab:orange")
    axang.set_ylabel("Angle (deg)")
    axwave.plot(time, wave, color="tab:green")
    axwave.set_ylabel("Wave (m)")
    for ax in (axw, axang, axwave):
        ax.grid(True, linestyle="--", alpha=0.5)
    axwave.set_xlabel("Timestep")
    fig4.tight_layout()

    # distance over time figure
    fig5, axd = plt.subplots(figsize=(8, 4))
    axd.plot(time, cumulative_dist, marker="o")
    axd.set_xlabel("Timestep")
    axd.set_ylabel("Distance (m)")
    axd.grid(True, linestyle="--", alpha=0.5)
    fig5.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
