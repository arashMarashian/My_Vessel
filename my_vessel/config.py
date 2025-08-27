import os

OPENTOPO_API_KEY = os.getenv("OPENTOPO_API_KEY")  # required for remote fetch
DEFAULT_DEM_TYPE = os.getenv("DEM_TYPE", "SRTM15Plus")
MAX_PIXELS = int(os.getenv("MAX_PIXELS", "8000000"))
