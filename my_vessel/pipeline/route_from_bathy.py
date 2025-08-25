from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np

from ..bathy.grid import bathy_to_occupancy, latlon_to_rc, rc_to_latlon
from path_planner.a_star_planner import AStarPlanner
from path_planner.utils import densify_path, smooth_path


def plan_route(
    arr_bathy: np.ndarray,
    bounds: Tuple[float, float, float, float],
    start_latlon: Tuple[float, float],
    goal_latlon: Tuple[float, float],
    min_depth_m: float,
    dilate_cells: int = 0,
    densify_pts: int = 4,
    smoothness: float = 0.3,
    iterations: int = 200,
):
    grid = bathy_to_occupancy(arr_bathy, min_depth_m=min_depth_m, dilate_cells=dilate_cells)
    start_rc = latlon_to_rc(*start_latlon, bounds, grid.shape)
    goal_rc = latlon_to_rc(*goal_latlon, bounds, grid.shape)
    planner = AStarPlanner()
    path_rc = planner.plan(start_rc, goal_rc, grid)
    path_dense = densify_path(path_rc, points_per_segment=densify_pts)
    path_smooth = smooth_path(path_dense, smoothness=smoothness, iterations=iterations)
    path_ll = [rc_to_latlon(r, c, bounds, grid.shape) for (r, c) in path_smooth]
    return grid, path_rc, path_ll
