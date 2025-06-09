import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.map_generator import generate_environment_grid


def test_generate_environment_grid():
    env = generate_environment_grid()

    assert isinstance(env, dict)
    assert set(env.keys()) == {"grid", "start", "goal"}

    grid = env["grid"]
    assert isinstance(grid, np.ndarray)
    assert grid.shape == (10, 10)

    start, goal = env["start"], env["goal"]
    assert isinstance(start, tuple) and len(start) == 2
    assert isinstance(goal, tuple) and len(goal) == 2
