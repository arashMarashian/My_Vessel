import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from path_planner.utils import densify_path


def test_densify_increases_points():
    path = [(0, 0), (2, 0), (2, 2)]
    dense = densify_path(path, points_per_segment=4)
    assert dense[0] == (0.0, 0.0)
    assert dense[-1] == (2.0, 2.0)
    assert len(dense) > len(path)
