from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
import rasterio

from ..config import OPENTOPO_API_KEY, DEFAULT_DEM_TYPE
from utils.paths import ensure_results_subdir

OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"


@dataclass
class BBox:
    south: float
    west: float
    north: float
    east: float


def _cache_path(bbox: BBox, dem_type: str) -> Path:
    cache_dir = ensure_results_subdir("cache_opentopo")
    safe = f"{dem_type}_{bbox.south}_{bbox.west}_{bbox.north}_{bbox.east}.tif".replace(" ", "")
    return cache_dir / safe


def fetch_geotiff_bytes(bbox: BBox, dem_type: str = DEFAULT_DEM_TYPE, timeout: int = 60, use_cache: bool = True) -> bytes:
    if not OPENTOPO_API_KEY:
        raise RuntimeError("OPENTOPO_API_KEY is not set; use local GeoTIFF or set the env var.")
    if use_cache:
        cp = _cache_path(bbox, dem_type)
        if cp.exists():
            return cp.read_bytes()
    params = {
        "demtype": dem_type,
        "south": bbox.south,
        "north": bbox.north,
        "west": bbox.west,
        "east": bbox.east,
        "outputFormat": "GTiff",
        "API_Key": OPENTOPO_API_KEY,
    }
    r = requests.get(OPENTOPO_URL, params=params, timeout=timeout)
    r.raise_for_status()
    content = r.content
    if use_cache:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(content)
    return content


def read_raster_from_bytes(tif_bytes: bytes):
    return rasterio.open(io.BytesIO(tif_bytes))


def read_raster_from_file(path: str):
    return rasterio.open(path)
