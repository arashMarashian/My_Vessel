"""Collision utilities for grid-based path planners."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

Coord = Tuple[float, float]


def _sample_points(p0: Coord, p1: Coord, step: float) -> np.ndarray:
    r0, c0 = p0
    r1, c1 = p1
    dr, dc = r1 - r0, c1 - c0
    length = math.hypot(dr, dc)
    if length == 0:
        return np.array([[r0, c0]], dtype=float)
    step = max(float(step), 1e-3)
    n = max(int(math.ceil(length / step)), 1)
    ts = np.linspace(0.0, 1.0, n + 1)
    rows = r0 + ts * dr
    cols = c0 + ts * dc
    return np.vstack((rows, cols)).T


def segment_is_free(
    p0: Sequence[float],
    p1: Sequence[float],
    grid: np.ndarray,
    step: float = 0.25,
) -> bool:
    """Return True if the straight segment between ``p0`` and ``p1`` is collision-free."""

    if grid.size == 0:
        return True

    H, W = grid.shape
    samples = _sample_points(tuple(map(float, p0)), tuple(map(float, p1)), step)
    for r, c in samples:
        rr = int(round(r))
        cc = int(round(c))
        if rr < 0 or rr >= H or cc < 0 or cc >= W:
            return False
        if grid[rr, cc] != 0:
            return False
    return True


__all__ = ["segment_is_free"]
