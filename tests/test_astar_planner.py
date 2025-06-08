import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from my_vessel.path_planner.a_star_planner import AStarPlanner
from my_vessel.utils.plotting import plot_grid_path


def test_astar_plotting():
    grid = np.array([
        [0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])

    start = (0, 0)
    goal = (4, 4)

    planner = AStarPlanner()
    path = planner.plan(start, goal, grid)

    assert path[0] == start and path[-1] == goal
    plot_grid_path(grid, path, start, goal)
