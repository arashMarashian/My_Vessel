import argparse, json, csv
import matplotlib.pyplot as plt
import folium

from my_vessel.bathy.overlay import make_overlay_data_url
from my_vessel.bathy.grid import oriented_array_and_bounds, downsample, latlon_to_rc, rc_to_latlon
from my_vessel.pipeline.route_from_bathy import plan_route
from my_vessel.pipeline.speed_profile import feasible_speed_profile
from my_vessel.bathy.fetch import BBox, fetch_geotiff_bytes, read_raster_from_bytes, read_raster_from_file
from energy.vessel_energy_system import VesselEnergySystem, Battery
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
    # map options
    p.add_argument("--map-html", type=str, default=None, help="If set, write an interactive map HTML here")
    p.add_argument("--map-tiles", type=str, default="Esri.WorldImagery", help="Basemap tiles: 'Esri.WorldImagery' (satellite) or 'OpenStreetMap'")
    p.add_argument("--bathy-cmap", type=str, default="viridis")
    p.add_argument("--bathy-opacity", type=float, default=0.6)
    p.add_argument("--bathy-vmin", type=float, default=None)
    p.add_argument("--bathy-vmax", type=float, default=None)
    # energy / env options
    p.add_argument("--engine-yaml", type=str, default="data/engine_data.yaml")
    p.add_argument("--target-speed-kn", type=float, default=12.0)
    p.add_argument("--dt-s", type=int, default=60)
    p.add_argument("--wind-speed", type=float, default=0.0)
    p.add_argument("--wind-angle-diff", type=float, default=0.0)
    p.add_argument("--wave-height", type=float, default=0.0)
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
        arr, bounds, tuple(args.start), tuple(args.goal),
        min_depth_m=min_depth, dilate_cells=args.dilate_cells,
        verbose=args.verbose
    )

    if args.verbose:
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
    ves = VesselEnergySystem(engines, battery=Battery(capacity_kwh=1e5))

    class _VESAdapter:
        def __init__(self, ves):
            self.ves = ves

        def step(self, environment, target_speed, timestep_seconds):
            loads = [eng.max_load for eng in self.ves.engines]
            res = self.ves.step(
                controller_action={
                    "engine_loads": loads,
                    "battery_power": 0.0,
                    "target_speed": target_speed,
                },
                environment=environment,
                timestep_hours=timestep_seconds / 3600.0,
            )
            return {
                "achieved_speed_knots": res.get("actual_speed", 0.0),
                "fuel_consumed_kg": sum(res.get("fuel_used_g", [])) / 1000.0,
            }

    env_const = {
        "wind_speed": args.wind_speed,
        "wind_angle_diff": args.wind_angle_diff,
        "wave_height": args.wave_height,
    }
    profile = feasible_speed_profile(
        _VESAdapter(ves), path_ll, target_speed_knots=args.target_speed_kn,
        dt_s=args.dt_s, env_const=env_const
    )

    # Save GeoJSON
    gj = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in path_ll]},
            "properties": {
                "min_depth_m": min_depth,
                "total_nm": profile["total_nm"],
                "total_time_s": profile["total_time_s"],
                "total_fuel_kg": profile["total_fuel_kg"],
            },
        }],
    }
    with open(f"{args.out_prefix}.geojson", "w") as f:
        json.dump(gj, f)

    # Save CSV per segment
    with open(f"{args.out_prefix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["i", "lat", "lon", "v_kn", "seg_nm", "time_s", "fuel_kg"])
        w.writeheader()
        for row in profile["segments"]:
            w.writerow(row)

    # Static plot
    plt.figure()
    if path_ll:
        lats = [lat for lat, lon in path_ll]
        lons = [lon for lat, lon in path_ll]
        plt.plot(lons, lats)
    plt.title("Planned Route (lat/lon)")
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.savefig(f"{args.out_prefix}.png", dpi=180)

    # Interactive map
    if args.map_html:
        center = list(path_ll[0]) if path_ll else [(bounds[2] + bounds[0]) / 2, (bounds[1] + bounds[3]) / 2]
        m = folium.Map(location=center, zoom_start=8, tiles=None)
        if args.map_tiles.lower() == "openstreetmap":
            folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(m)
        else:
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri.WorldImagery", name="Esri Satellite", control=True
            ).add_to(m)
            folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(m)

        data_url, (S, W, N, E), visible = make_overlay_data_url(
            arr, bounds, vmin=args.bathy_vmin, vmax=args.bathy_vmax,
            cmap_name=args.bathy_cmap, opacity=args.bathy_opacity
        )
        folium.raster_layers.ImageOverlay(
            image=data_url, bounds=[[S, W], [N, E]], opacity=args.bathy_opacity, name="Bathymetry"
        ).add_to(m)

        if path_ll:
            folium.PolyLine([(lat, lon) for lat, lon in path_ll], color="#00A", weight=4, opacity=0.9, tooltip="Route").add_to(m)
            folium.Marker(location=list(path_ll[0]), popup="Start").add_to(m)
            folium.Marker(location=list(path_ll[-1]), popup="Goal").add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        m.save(args.map_html)
        if args.verbose:
            print(f"[DEBUG] wrote interactive map -> {args.map_html}")

    if args.verbose:
        import numpy as np
        total = grid.size
        obst = int(grid.sum())
        free = total - obst
        print(f"[DEBUG] CRS={src.crs}, bounds(S,W,N,E)={bounds}")
        print(f"[DEBUG] bathy: min={np.nanmin(arr):.2f}, max={np.nanmax(arr):.2f}, NaN%={np.isnan(arr).mean()*100:.1f}%")
        print(f"[DEBUG] grid shape={grid.shape}, free={free} ({free/total*100:.1f}%), obst={obst} ({obst/total*100:.1f}%)")
        print(f"[DEBUG] path_rc length={len(path_rc)}, path_ll length={len(path_ll)}")
        print(f"[DEBUG] totals: nm={profile['total_nm']:.2f}, time_h={profile['total_time_s']/3600.0:.2f}, fuel_kg={profile['total_fuel_kg']:.2f}")
        print(f"[DEBUG] saved {args.out_prefix}.geojson {args.out_prefix}.csv {args.out_prefix}.png")


if __name__ == "__main__":
    main()

