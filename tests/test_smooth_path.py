import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from path_planner import AStarPlanner
from path_planner.utils import densify_path, smooth_path


def _create_grid():
    grid = np.zeros((7, 7), dtype=int)
    grid[3, :5] = 1
    return grid


def test_smooth_path_identity():
    grid = _create_grid()
    planner = AStarPlanner()
    start, goal = (0, 0), (6, 6)
    path = planner.plan(start, goal, grid)
    dense = densify_path(path, points_per_segment=4)

    smoothed = smooth_path(dense, smoothness=0.0)
    assert smoothed == dense


def test_smooth_path_changes():
    grid = _create_grid()
    planner = AStarPlanner()
    start, goal = (0, 0), (6, 6)
    path = planner.plan(start, goal, grid)
    dense = densify_path(path, points_per_segment=4)

    smoothed = smooth_path(dense, smoothness=1.0)
    assert smoothed[0] == dense[0]
    assert smoothed[-1] == dense[-1]
    assert smoothed != dense
