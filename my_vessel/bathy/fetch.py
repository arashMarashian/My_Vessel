from __future__ import annotations

import io
from dataclasses import dataclass

import requests
import rasterio

from ..config import OPENTOPO_API_KEY, DEFAULT_DEM_TYPE

OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"


@dataclass
class BBox:
    south: float
    west: float
    north: float
    east: float


def fetch_geotiff_bytes(bbox: BBox, dem_type: str = DEFAULT_DEM_TYPE, timeout: int = 60) -> bytes:
    if not OPENTOPO_API_KEY:
        raise RuntimeError("OPENTOPO_API_KEY is not set; use local GeoTIFF or set the env var.")
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
    return r.content


def read_raster_from_bytes(tif_bytes: bytes):
    return rasterio.open(io.BytesIO(tif_bytes))


def read_raster_from_file(path: str):
    return rasterio.open(path)
