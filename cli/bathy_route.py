import argparse, json, csv
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
from my_vessel.energy import VesselEnergySystem
from engine_loader import load_engines_from_yaml


def main() -> None:
    p = argparse.ArgumentParser(description="Bathymetry-aware routing & speed profile")
    p.add_argument("--bbox", type=float, nargs=4, metavar=("S", "W", "N", "E"))
    p.add_argument("--local-tif", type=str, help="Path to local GeoTIFF (alternative to remote fetch)")
    p.add_argument("--dem-type", type=str, default=None, help="OpenTopography DEM type (e.g., SRTM15Plus)")
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
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.local_tif:
        src = read_raster_from_file(args.local_tif)
    else:
        tif = fetch_geotiff_bytes(BBox(*args.bbox), dem_type=args.dem_type)
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
        verbose=args.verbose,
    )

    if args.verbose:
        from my_vessel.bathy.grid import latlon_to_rc
        import numpy as np
        sr, sc = latlon_to_rc(*args.start, bounds, grid.shape)
        gr, gc = latlon_to_rc(*args.goal, bounds, grid.shape)

        def cellinfo(r, c):
            H, W = grid.shape
            if not (0 <= r < H and 0 <= c < W):
                return "OOB"
            depth = arr[r, c]
            depth_s = "nan" if np.isnan(depth) else f"{float(depth):.2f}m"
            return f"{'obst' if grid[r, c] == 1 else 'free'}, depth={depth_s}"

        print(f"[DEBUG] start rc={(sr, sc)} -> {cellinfo(sr, sc)}")
        print(f"[DEBUG] goal  rc={(gr, gc)} -> {cellinfo(gr, gc)}")

    engines = load_engines_from_yaml(args.engine_yaml)
    ves = VesselEnergySystem(engines, battery=None)
    prof = feasible_speed_profile(
        ves, path_ll, target_speed_knots=args.target_speed_kn, dt_s=args.dt_s
    )

    if args.verbose:
        import numpy as np
        total = grid.size
        obst = int(grid.sum())
        free = total - obst
        print(f"[DEBUG] CRS={src.crs}, bounds(S,W,N,E)={bounds}")
        print(
            f"[DEBUG] bathy: min={np.nanmin(arr):.2f}, max={np.nanmax(arr):.2f}, NaN%={np.isnan(arr).mean()*100:.1f}%"
        )
        print(
            f"[DEBUG] grid shape={grid.shape}, free={free} ({free/total*100:.1f}%), obst={obst} ({obst/total*100:.1f}%)"
        )
        print(f"[DEBUG] path_rc length={len(path_rc)}, path_ll length={len(path_ll)}")

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
        if prof:
            w.writerows(prof["segments"])

    plt.figure()
    if path_ll:
        lats = [lat for lat, lon in path_ll]
        lons = [lon for lat, lon in path_ll]
        plt.plot(lons, lats)
    plt.title("Planned Route (lat/lon)")
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.savefig(f"{args.out_prefix}.png", dpi=180)

    if args.verbose:
        print(
            f"[DEBUG] saved {args.out_prefix}.geojson {args.out_prefix}.csv {args.out_prefix}.png"
        )


if __name__ == "__main__":
    main()
