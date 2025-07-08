import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from path_planner import AStarPlanner
from path_planner.utils import plot_map, plot_path
from environment.map_generator import generate_environment_grid
from environment.map_utils import extract_environment_along_path
from engine_loader import load_engines_from_yaml
from energy.opt_control import DispatchOptimizer
from energy import propulsion_power, hotel_power, aux_power


def heading_degrees(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle % 360.0


def build_env_list(env_grid, path_xy):
    env_list = []
    for i, (x, y) in enumerate(path_xy):
        if i < len(path_xy) - 1:
            heading = heading_degrees(path_xy[i], path_xy[i + 1])
        else:
            heading = heading_degrees(path_xy[i - 1], path_xy[i])
        wind_speed = float(env_grid["wind_speed"][y, x])
        wind_dir = float(env_grid["wind_angle"][y, x])
        diff = abs((wind_dir - heading + 180) % 360 - 180)
        wave_h = float(env_grid["wave_height"][y, x])
        env_list.append(
            {"wind_speed": wind_speed, "wind_angle_diff": diff, "wave_height": wave_h}
        )
    return env_list


def main() -> None:
    # 1. Build grid with obstacles
    grid = np.zeros((20, 20), dtype=int)
    grid[10, 3:17] = 1
    grid[3:17, 10] = 1
    start = (2, 2)
    goal = (17, 17)
    overall_heading = heading_degrees(start, goal)

    planner = AStarPlanner()
    path = planner.plan(start, goal, grid)

    path_xy = [(c, r) for r, c in path]

    # 2. Generate environment data and sample along path
    env_grid = generate_environment_grid(size=grid.shape)
    env_on_path = extract_environment_along_path(env_grid, path_xy)
    env_list = build_env_list(env_grid, path_xy)

    wind_speed_map = env_grid["wind_speed"]
    wind_angle_map = np.abs((env_grid["wind_angle"] - overall_heading + 180) % 360 - 180)
    wave_height_map = env_grid["wave_height"]

    # 3. Load engines and set up optimizer
    path_to_data = os.path.join("data", "engine_data_main.yaml")
    engines = load_engines_from_yaml(path_to_data)[:2]
    with open(path_to_data) as f:
        data = yaml.safe_load(f)
    curves = []
    for eng in data["engines"][:2]:
        curves.append({float(k): v for k, v in eng["sfoc"]["HFO"].items()})

    horizon = len(env_list)
    opt = DispatchOptimizer(
        engines,
        horizon=horizon,
        env=env_list,
        dt_hours=1.0,
        battery_capacity_kwh=500.0,
        sfoc_curves=curves,
        target_distance_m=200_000.0,
    )

    solver = "ipopt"
    opt.solve(solver)
    results = opt.results()

    time = np.arange(horizon)
    speeds = results["speed"]
    soc = results["soc"][:-1]

    plt.figure(figsize=(5, 5))
    plot_map(grid)
    plot_path(path)
    plt.title("Planned Path")

    fig2, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(time, speeds, marker="o", label="Speed")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Speed (m/s)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(time, soc, marker="s", color="tab:orange", label="SOC")
    ax2.set_ylabel("Battery SOC")
    fig2.tight_layout()

    # Compute power components
    prop_power = [propulsion_power(env_list[t], speeds[t]) for t in range(horizon)]
    hotel = [hotel_power(env_list[t]) for t in range(horizon)]
    auxiliary = [aux_power(env_list[t], prop_power[t]) for t in range(horizon)]

    fig3, (axp, axh, axa) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axp.plot(time, prop_power, label="Propulsion")
    axp.set_ylabel("P_prop (W)")
    axh.plot(time, hotel, label="Hotel", color="tab:orange")
    axh.set_ylabel("P_hotel (W)")
    axa.plot(time, auxiliary, label="Aux", color="tab:green")
    axa.set_ylabel("P_aux (W)")
    axa.set_xlabel("Timestep")
    for ax in (axp, axh, axa):
        ax.grid(True, linestyle="--", alpha=0.5)
    fig3.tight_layout()

    # Plot environmental conditions along the horizon
    wind_speed = [c["wind_speed"] for c in env_list]
    wind_angle_diff = [c["wind_angle_diff"] for c in env_list]
    wave_height = [c["wave_height"] for c in env_list]

    fig4, ax_env = plt.subplots(figsize=(8, 4))
    ax_env.plot(time, wind_speed, label="Wind Speed (m/s)")
    ax_env.plot(time, wind_angle_diff, label="Wind Angle Diff (deg)")
    ax_env.plot(time, wave_height, label="Wave Height (m)")
    ax_env.set_xlabel("Timestep")
    ax_env.set_ylabel("Value")
    ax_env.set_title("Environmental Conditions")
    ax_env.grid(True, linestyle="--", alpha=0.5)
    ax_env.legend()
    fig4.tight_layout()

    # Visualize environmental fields across the grid
    fig5, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(wind_speed_map, cmap="viridis", origin="lower", aspect="auto")
    axes[0].set_title("Wind Speed (m/s)")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    fig5.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(wind_angle_map, cmap="twilight", origin="lower", aspect="auto")
    axes[1].set_title("Wind Angle Diff (deg)")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    fig5.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(wave_height_map, cmap="coolwarm", origin="lower", aspect="auto")
    axes[2].set_title("Wave Height (m)")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")
    fig5.colorbar(im2, ax=axes[2])

    fig5.tight_layout()

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
