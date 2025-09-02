import numpy as np

from my_vessel.bathy.grid import bathy_to_occupancy


def test_obstacles_from_shallow():
    arr = np.array([[1.0, -0.5, -10.0, np.nan]], dtype="float32")
    grid = bathy_to_occupancy(arr, min_depth_m=2.0, dilate_cells=0)
    assert list(grid[0]) == [1, 1, 0, 1]
