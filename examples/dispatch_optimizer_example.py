import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine_loader import load_engines_from_yaml
from energy.opt_control import DispatchOptimizer


def main() -> None:
    """Run the dispatch optimizer for a short horizon and plot results."""
    path = os.path.join("data", "engine_data.yaml")
    engines = load_engines_from_yaml(path)[:2]

    with open(path) as f:
        data = yaml.safe_load(f)
    curves = []
    for eng in data["engines"][:2]:
        curve = {float(k): v for k, v in eng["sfoc"]["HFO"].items()}
        curves.append(curve)

    horizon = 5
    env = [
        {"wind_speed": 5.0, "wind_angle_diff": 0.0, "wave_height": 1.0}
        for _ in range(horizon)
    ]

    opt = DispatchOptimizer(
        engines,
        horizon=horizon,
        env=env,
        dt_hours=1.0,
        battery_capacity_kwh=500.0,
        sfoc_curves=curves,
    )

    solver = "ipopt"
    opt.solve(solver)
    results = opt.results()

    time = np.arange(horizon)
    speeds = results["speed"]
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

    plt.show()


if __name__ == "__main__":
    main()
