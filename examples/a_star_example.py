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


def main():
    """Run a basic A* path planning example."""
    # Create 10x10 grid with a horizontal obstacle wall
    grid = np.zeros((10, 10), dtype=int)
    grid[5, 1:8] = 1

    start = (2, 2)
    goal = (8, 8)

    planner = AStarPlanner()
    path = planner.plan(start, goal, grid)
    dense = densify_path(path, points_per_segment=5)
    smoothed = smooth_path(dense, smoothness=0.5)

    print(f"Path length: {len(path)}")
    print("Path:", path)
    print("Smoothed path:", smoothed)

    plt.figure(figsize=(5, 5))
    plot_map(grid)
    plot_path(smoothed)
    plt.show()


if __name__ == "__main__":
    main()
