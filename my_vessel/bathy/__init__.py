"""Bathymetry utilities for route planning."""

from .fetch import BBox, fetch_geotiff_bytes, read_raster_from_bytes, read_raster_from_file
from .grid import (
    oriented_array_and_bounds,
    downsample,
    bathy_to_occupancy,
    rc_to_latlon,
    latlon_to_rc,
)
from .overlay import make_overlay_data_url

__all__ = [
    "BBox",
    "fetch_geotiff_bytes",
    "read_raster_from_bytes",
    "read_raster_from_file",
    "oriented_array_and_bounds",
    "downsample",
    "bathy_to_occupancy",
    "rc_to_latlon",
    "latlon_to_rc",
    "make_overlay_data_url",
]
