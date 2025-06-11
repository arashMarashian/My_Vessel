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

# === CONFIGURATION ===
ENGINE_PATH = os.path.join("data", "engine_data.yaml")
HORIZON = 5
DT_HOURS = 1.0
BATTERY_CAPACITY_KWH = 500.0
FUEL_TYPE = "HFO"
TARGET_DISTANCE_M = 60_000.0
N_ENGINES = 4


def load_engines_and_curves(path: str, fuel_type: str, n_engines: int):
    engines = load_engines_from_yaml(path)[:n_engines]

    with open(path) as f:
        data = yaml.safe_load(f)

    curves = []
    for eng in data["engines"][:n_engines]:
        curve = {float(k): v for k, v in eng["sfoc"][fuel_type].items()}
        curves.append(curve)

    return engines, curves


def create_environment_sequence(length: int):
    return [{"wind_speed": 5.0, "wind_angle_diff": 0.0, "wave_height": 1.0}
            for _ in range(length)]


def plot_results(time, speeds, soc, loads, engines):
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

    plt.title("Speed and Battery SOC over Time")
    fig.tight_layout()

    fig2, axs = plt.subplots(len(engines), 1, figsize=(8, 3 * len(engines)), sharex=True)
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    for i, ax in enumerate(axs):
        ax.step(time, loads[:, i], where="mid", label=engines[i].name)
        ax.set_ylabel("Load %")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
    axs[-1].set_xlabel("Timestep")
    plt.suptitle("Engine Load Profiles", y=1.02)
    fig2.tight_layout()

    plt.show()


def main() -> None:
    engines, curves = load_engines_and_curves(ENGINE_PATH, FUEL_TYPE, N_ENGINES)
    env = create_environment_sequence(HORIZON)

    opt = DispatchOptimizer(
        engines=engines,
        horizon=HORIZON,
        env=env,
        dt_hours=DT_HOURS,
        battery_capacity_kwh=BATTERY_CAPACITY_KWH,
        sfoc_curves=curves,
        target_distance_m=TARGET_DISTANCE_M,
    )

    solver = "ipopt"
    opt.solve(solver)
    results = opt.results()

    time = np.arange(HORIZON)
    speeds = results["speed"]
    soc = results["soc"][:-1]
    loads = np.array(results["loads"]) * 100
    distance = np.sum(speeds) * 3600  # speed in m/s × seconds

    plot_results(time, speeds, soc, loads, engines)

    print("\n=== Optimization Summary ===")
    print(f"Total distance traveled: {distance:.1f} m")
    print(f"Final SOC: {results['soc'][-1]:.2f} kWh")
    print(f"Total fuel used (g): {sum(map(sum, results['fuel_used'])):.1f}")
    print()


if __name__ == "__main__":
    main()
