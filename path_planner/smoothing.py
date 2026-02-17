"""Obstacle-aware path smoothing utilities."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .collision import segment_is_free

Coord = Tuple[float, float]


def _normalize_path(path: Sequence[Tuple[float, float]]) -> List[Coord]:
    return [tuple(map(float, p)) for p in path]


def shortcut_smooth_path(
    path: Sequence[Tuple[float, float]],
    grid: np.ndarray,
    max_iters: int = 200,
    step: float = 0.25,
) -> List[Coord]:
    """Simplify a path using line-of-sight shortcuts that respect obstacles."""
    if not path:
        return []
    if len(path) <= 2:
        return _normalize_path(path)

    pts = _normalize_path(path)
    iters = 0
    while iters < max_iters:
        improved = False
        i = 0
        while i < len(pts) - 2:
            j = len(pts) - 1
            removed = False
            while j > i + 1:
                if segment_is_free(pts[i], pts[j], grid, step=step):
                    del pts[i + 1 : j]
                    improved = True
                    removed = True
                    break
                j -= 1
            if not removed:
                i += 1
        if not improved:
            break
        iters += 1
    return pts


__all__ = ["shortcut_smooth_path"]
