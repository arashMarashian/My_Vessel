import argparse
import csv
import json

import matplotlib.pyplot as plt

from my_vessel.bathy.fetch import (
    BBox,
    fetch_geotiff_bytes,
    read_raster_from_bytes,
    read_raster_from_file,
)
from my_vessel.bathy.grid import oriented_array_and_bounds, downsample
from my_vessel.pipeline.route_from_bathy import plan_route
from my_vessel.pipeline.speed_profile import feasible_speed_profile
from my_vessel.energy.vessel_energy_system import VesselEnergySystem
from my_vessel.engine_loader import load_engines_from_yaml


def main() -> None:
    p = argparse.ArgumentParser(description="Bathymetry-aware routing & speed profile")
    p.add_argument("--bbox", type=float, nargs=4, metavar=("S", "W", "N", "E"))
    p.add_argument("--local-tif", type=str, help="Path to local GeoTIFF (alternative to remote fetch)")
    p.add_argument("--downsample", type=int, default=1)
    p.add_argument("--draft", type=float, required=True)
    p.add_argument("--ukc", type=float, default=1.0)
    p.add_argument("--dilate-cells", type=int, default=0)
    p.add_argument("--start", type=float, nargs=2, metavar=("LAT", "LON"), required=True)
    p.add_argument("--goal", type=float, nargs=2, metavar=("LAT", "LON"), required=True)
    p.add_argument("--engine-yaml", type=str, default="data/engine_data.yaml")
    p.add_argument("--target-speed-kn", type=float, default=12.0)
    p.add_argument("--dt-s", type=int, default=60)
    p.add_argument("--out-prefix", type=str, default="route_out")
    args = p.parse_args()

    if args.local_tif:
        src = read_raster_from_file(args.local_tif)
    else:
        tif = fetch_geotiff_bytes(BBox(*args.bbox))
        src = read_raster_from_bytes(tif)

    arr, bounds = oriented_array_and_bounds(src)
    if args.downsample > 1:
        arr = downsample(arr, args.downsample)

    min_depth = args.draft + args.ukc
    grid, path_rc, path_ll = plan_route(
        arr,
        bounds,
        tuple(args.start),
        tuple(args.goal),
        min_depth_m=min_depth,
        dilate_cells=args.dilate_cells,
    )

    engines = load_engines_from_yaml(args.engine_yaml)
    ves = VesselEnergySystem(engines, battery=None)
    prof = feasible_speed_profile(
        ves, path_ll, target_speed_knots=args.target_speed_kn, dt_s=args.dt_s
    )

    gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon, lat] for lat, lon in path_ll],
                },
                "properties": {"min_depth_m": min_depth},
            }
        ],
    }
    with open(f"{args.out_prefix}.geojson", "w") as f:
        json.dump(gj, f)

    with open(f"{args.out_prefix}.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["i", "lat", "lon", "v_kn", "seg_nm", "time_s", "fuel_kg"]
        )
        w.writeheader()
        w.writerows(prof["segments"])

    plt.figure()
    lats = [lat for lat, lon in path_ll]
    lons = [lon for lat, lon in path_ll]
    plt.plot(lons, lats)
    plt.title("Planned Route (lat/lon)")
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.savefig(f"{args.out_prefix}.png", dpi=180)


if __name__ == "__main__":
    main()
