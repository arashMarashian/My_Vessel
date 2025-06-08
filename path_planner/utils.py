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


def smooth_path(
    path: List[Tuple[int, int]], smoothness: float, iterations: int = 50
) -> List[Tuple[float, float]]:
    """Return a smoothed version of a grid path.

    Parameters
    ----------
    path : List[Tuple[int, int]]
        Discrete path returned by a planner.
    smoothness : float
        Value between 0 and 1 specifying how much to smooth the path.
        ``0`` returns the original path while ``1`` applies the maximum
        smoothing.
    iterations : int, optional
        Number of smoothing iterations to perform.

    Returns
    -------
    List[Tuple[float, float]]
        The smoothed path coordinates.
    """

    if not path or smoothness <= 0.0:
        return list(path)

    smoothness = float(np.clip(smoothness, 0.0, 1.0))

    pts = np.asarray(path, dtype=float)
    new_pts = pts.copy()

    weight_data = 1.0 - smoothness
    weight_smooth = smoothness

    for _ in range(iterations):
        for i in range(1, len(pts) - 1):
            new_pts[i] += weight_data * (pts[i] - new_pts[i])
            new_pts[i] += weight_smooth * (
                new_pts[i - 1] + new_pts[i + 1] - 2.0 * new_pts[i]
            )

    return [tuple(p) for p in new_pts]

