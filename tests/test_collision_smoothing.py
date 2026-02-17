import numpy as np

from my_vessel.path_planner.collision import segment_is_free
from my_vessel.path_planner.smoothing import shortcut_smooth_path


def test_segment_detects_obstacle_and_bounds():
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[2, 2] = 1
    assert not segment_is_free((0, 0), (4, 4), grid)
    assert segment_is_free((0, 0), (0, 4), grid)
    assert not segment_is_free((4, 4), (5, 5), grid)


def test_shortcut_smoothing_stays_collision_free():
    grid = np.zeros((6, 6), dtype=np.uint8)
    grid[3, 1:5] = 1  # horizontal wall across the grid

    path = [
        (0, 0),
        (2, 0),
        (2, 5),
        (5, 5),
    ]

    smoothed = shortcut_smooth_path(path, grid, max_iters=50)
    assert smoothed[0] == (0.0, 0.0)
    assert smoothed[-1] == (5.0, 5.0)

    for p0, p1 in zip(smoothed, smoothed[1:]):
        assert segment_is_free(p0, p1, grid), f"segment {p0}->{p1} hits obstacle"

    assert len(smoothed) >= 3, "Wall should prevent collapsing to a single segment"
