import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


def plot_grid_path(grid: np.ndarray, path: List[Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int]):
    """Plot a 2D grid map with a path, start, and goal."""
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y, x] == 1:
                plt.plot(x, y, 'ks')
    if path:
        px, py = zip(*path)
        plt.plot(px, py, 'r.-', label="Path")
    plt.plot(start[1], start[0], 'go', label="Start")
    plt.plot(goal[1], goal[0], 'bx', label="Goal")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(True)
    plt.title("A* Path on Grid")
    plt.show()
