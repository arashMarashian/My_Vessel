import os
import sys

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from path_planner import AStarPlanner
from path_planner.utils import plot_map, plot_path, densify_path, smooth_path
from environment.map_utils import load_environment_data, extract_environment_along_path


def main():
    """Run a basic A* path planning example."""
    # Create 10x10 grid with a horizontal obstacle wall
    grid = np.zeros((10, 10), dtype=int)
    grid[5, 1:8] = 1

    start = (2, 2)
    goal = (8, 8)

    planner = AStarPlanner()
    path = planner.plan(start, goal, grid)
    dense = densify_path(path, points_per_segment=10)
    smoothed = smooth_path(dense, smoothness=0.25)

    # Load environmental data and sample it along the discrete path
    env = load_environment_data()
    path_xy = [(c, r) for r, c in path]
    env_on_path = extract_environment_along_path(env, path_xy)

    print("Wind speed along path:", env_on_path["wind_speed"])
    print("Wave height along path:", env_on_path["wave_height"])

    print(f"Path length: {len(path)}")
    print("Path:", path)
    print("Smoothed path:", smoothed)

    plt.figure(figsize=(5, 5))
    plot_map(grid)
    plot_path(smoothed)
    plt.show()

    # Visualize environmental data along the planned path
    time = np.arange(len(path_xy))
    fig, axes = plt.subplots(len(env_on_path), 1,
                             figsize=(6, 2.5 * len(env_on_path)),
                             sharex=True)

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for ax, (key, values) in zip(axes, env_on_path.items()):
        ax.plot(time, values, marker="o")
        ax.set_ylabel(key)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Node ID")
    fig.suptitle("Environment along path")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
