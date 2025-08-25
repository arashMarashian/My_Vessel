from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import binary_dilation

from ..config import MAX_PIXELS


def oriented_array_and_bounds(src) -> tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Return raster band as north-up array and bounds (S, W, N, E)."""
    a = src.read(1).astype("float32")
    nodata = src.nodata
    if nodata is not None:
        a = np.where(a == nodata, np.nan, a)
    if src.transform.e > 0:  # row increases upward -> flip vertically
        a = np.flipud(a)
    if src.transform.a < 0:  # col increases leftward -> flip horizontally
        a = np.fliplr(a)
    south, west, north, east = src.bounds.bottom, src.bounds.left, src.bounds.top, src.bounds.right
    if a.size > MAX_PIXELS:
        raise ValueError(f"ROI too large ({a.shape}), reduce bbox or downsample.")
    return a, (south, west, north, east)


def downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    h = (arr.shape[0] // factor) * factor
    w = (arr.shape[1] // factor) * factor
    block = arr[:h, :w].reshape(h // factor, factor, w // factor, factor)
    return np.nanmean(block, axis=(1, 3)).astype("float32")


def bathy_to_occupancy(arr_bathy: np.ndarray, min_depth_m: float, dilate_cells: int = 0) -> np.ndarray:
    land = arr_bathy >= 0
    shallow = (arr_bathy > -float(min_depth_m)) & (arr_bathy <= 0)
    obstacles = np.where(np.isnan(arr_bathy), True, (land | shallow))
    if dilate_cells > 0:
        obstacles = binary_dilation(obstacles, iterations=dilate_cells)
    return obstacles.astype(np.uint8)


def rc_to_latlon(r: int, c: int, bounds: Tuple[float, float, float, float], shape: Tuple[int, int]) -> Tuple[float, float]:
    S, W, N, E = bounds
    H, Wpx = shape
    lat = N - (r + 0.5) * (N - S) / H
    lon = W + (c + 0.5) * (E - W) / Wpx
    return float(lat), float(lon)


def latlon_to_rc(lat: float, lon: float, bounds: Tuple[float, float, float, float], shape: Tuple[int, int]) -> Tuple[int, int]:
    S, W, N, E = bounds
    H, Wpx = shape
    r = int((N - lat) * H / (N - S))
    c = int((lon - W) * Wpx / (E - W))
    return max(0, min(H - 1, r)), max(0, min(Wpx - 1, c))
