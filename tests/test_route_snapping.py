import numpy as np
from my_vessel.pipeline.route_from_bathy import plan_route


def test_snaps_to_free_cell():
    arr = -10 * np.ones((50, 50), dtype="float32")
    arr[0, :] = 10.0  # land at north
    bounds = (0.0, 0.0, 1.0, 1.0)  # S,W,N,E
    start = (0.99, 0.01)  # near land row
    goal = (0.01, 0.99)
    grid, path_rc, path_ll = plan_route(arr, bounds, start, goal, min_depth_m=1.0, dilate_cells=0, verbose=True)
    assert len(path_rc) > 0
