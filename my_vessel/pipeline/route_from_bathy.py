from __future__ import annotations

import numpy as np
from collections import deque
from typing import Tuple

from path_planner.a_star_planner import AStarPlanner
from path_planner.utils import densify_path, smooth_path

from my_vessel.bathy.grid import bathy_to_occupancy, latlon_to_rc, rc_to_latlon


def _find_nearest_free(grid: np.ndarray, r: int, c: int, max_radius: int = 50):
    H, W = grid.shape
    def inb(R, C):
        return 0 <= R < H and 0 <= C < W
    if inb(r, c) and grid[r, c] == 0:
        return r, c
    q = deque([(r, c, 0)])
    seen = {(r, c)}
    while q:
        R, C, d = q.popleft()
        if d > max_radius:
            break
        for dR, dC in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            R2, C2 = R + dR, C + dC
            if (R2, C2) in seen or not inb(R2, C2):
                continue
            seen.add((R2, C2))
            if grid[R2, C2] == 0:
                return R2, C2
            q.append((R2, C2, d + 1))
    return None


def _rc_info(arr_bathy: np.ndarray, grid: np.ndarray, rc: tuple[int, int]) -> str:
    r, c = rc
    H, W = grid.shape
    if not (0 <= r < H and 0 <= c < W):
        return "OOB"
    depth = arr_bathy[r, c]
    depth_s = "nan" if np.isnan(depth) else f"{float(depth):.2f}m"
    return f"{'free' if grid[r, c] == 0 else 'obst'}, depth={depth_s}"


def plan_route(
    arr_bathy: np.ndarray,
    bounds: tuple[float, float, float, float],
    start_latlon: tuple[float, float],
    goal_latlon: tuple[float, float],
    min_depth_m: float,
    dilate_cells: int = 0,
    densify_pts: int = 4,
    smoothness: float = 0.3,
    iterations: int = 200,
    snap_endpoints: bool = True,
    snap_radius: int = 50,
    verbose: bool = False,
):
    # 0=free, 1=obstacle
    grid = bathy_to_occupancy(arr_bathy, min_depth_m=min_depth_m, dilate_cells=dilate_cells)
    start_rc = latlon_to_rc(*start_latlon, bounds, grid.shape)
    goal_rc = latlon_to_rc(*goal_latlon, bounds, grid.shape)

    if verbose:
        print(f"[DEBUG] start_rc={start_rc} -> {_rc_info(arr_bathy, grid, start_rc)}")
        print(f"[DEBUG] goal_rc={goal_rc} -> {_rc_info(arr_bathy, grid, goal_rc)}")

    # Snap endpoints if on obstacles or out-of-bounds
    if snap_endpoints:
        H, W = grid.shape

        def inb(rc):
            r, c = rc
            return 0 <= r < H and 0 <= c < W

        if (not inb(start_rc)) or grid[start_rc] == 1:
            s = _find_nearest_free(grid, *start_rc, max_radius=snap_radius) if inb(start_rc) else None
            if s:
                if verbose:
                    print(f"[DEBUG] snapped start_rc -> {s}")
                start_rc = s
        if (not inb(goal_rc)) or grid[goal_rc] == 1:
            g = _find_nearest_free(grid, *goal_rc, max_radius=snap_radius) if inb(goal_rc) else None
            if g:
                if verbose:
                    print(f"[DEBUG] snapped goal_rc -> {g}")
                goal_rc = g

    planner = AStarPlanner()

    # Attempt 1: assume 0=free, 1=obstacle
    path_rc = planner.plan(start_rc, goal_rc, grid)

    # Fallback: some planners expect the opposite convention
    if not path_rc:
        if verbose:
            print("[DEBUG] no path with 0=free/1=obst; retrying with inverted grid...")
        inv = (1 - grid).astype(np.uint8)
        path_rc = planner.plan(start_rc, goal_rc, inv)

    if not path_rc:
        if verbose:
            print("[DEBUG] still no path after inversion.")
        return grid, [], []

    path_dense = densify_path(path_rc, points_per_segment=densify_pts)
    path_smooth = smooth_path(path_dense, smoothness=smoothness, iterations=iterations)
    path_ll = [rc_to_latlon(r, c, bounds, grid.shape) for (r, c) in path_smooth]
    return grid, path_rc, path_ll

