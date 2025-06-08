import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from path_planner import AStarPlanner
from path_planner.utils import smooth_path


def _create_grid():
    grid = np.zeros((7, 7), dtype=int)
    grid[3, :5] = 1
    return grid


def test_smooth_path_identity():
    grid = _create_grid()
    planner = AStarPlanner()
    start, goal = (0, 0), (6, 6)
    path = planner.plan(start, goal, grid)

    smoothed = smooth_path(path, smoothness=0.0)
    assert smoothed == path


def test_smooth_path_changes():
    grid = _create_grid()
    planner = AStarPlanner()
    start, goal = (0, 0), (6, 6)
    path = planner.plan(start, goal, grid)

    smoothed = smooth_path(path, smoothness=1.0)
    assert smoothed[0] == path[0]
    assert smoothed[-1] == path[-1]
    assert smoothed != path
