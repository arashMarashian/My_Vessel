"""Utility functions for path planning."""

from typing import Iterable, Tuple, List

import matplotlib.pyplot as plt
import numpy as np


def plot_map(grid: np.ndarray) -> None:
    """Plot a binary occupancy grid."""
    plt.imshow(grid, cmap="Greys", origin="lower")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Occupancy Grid")


def plot_path(path: Iterable[Tuple[int, int]]) -> None:
    """Plot a sequence of grid coordinates on the current figure."""
    if not path:
        return
    pts = np.array(list(path))
    plt.plot(pts[:, 1], pts[:, 0], "r-", linewidth=2)
    plt.plot(pts[0, 1], pts[0, 0], "go", label="start")
    plt.plot(pts[-1, 1], pts[-1, 0], "bx", label="goal")
    plt.legend()

