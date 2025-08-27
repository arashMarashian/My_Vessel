import argparse, json, csv, os
from pathlib import Path
import matplotlib.pyplot as plt
import folium

from my_vessel.bathy.overlay import make_overlay_data_url
from my_vessel.bathy.grid import oriented_array_and_bounds, latlon_to_rc
from my_vessel.pipeline.route_from_bathy import plan_route
from my_vessel.pipeline.speed_profile import feasible_speed_profile
from my_vessel.environment.env_sources import sample_env_along_route
from my_vessel.bathy.fetch import BBox, fetch_geotiff_bytes, read_raster_from_bytes, read_raster_from_file
from energy.vessel_energy_system import (
    VesselEnergySystem,
    Battery,
    hotel_power,
    aux_power,
)
from energy.power_model import propulsion_power
from engine_loader import load_engines_from_yaml


def _save_plot(x, y, title, xlabel, ylabel, out_png):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(out_png, dpi=180)
    plt.close()


def main() -> None:
    default_engine_yaml = Path(__file__).resolve().parents[1] / "data" / "engine_data.yaml"

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
    p.add_argument(
        "--engine-yaml",
        type=str,
        default=str(default_engine_yaml),
        help="Path to engine YAML file (default: repo_root/data/engine_data.yaml)",
    )
    p.add_argument("--target-speed-kn", type=float, default=12.0)
    p.add_argument("--dt-s", type=int, default=60)
    p.add_argument("--wind-speed", type=float, default=0.0)
    p.add_argument("--wind-angle-diff", type=float, default=0.0)
    p.add_argument("--wave-height", type=float, default=0.0)
    p.add_argument("--env-source", type=str, default="constant", choices=["constant", "openmeteo"],
                   help="Environment provider")
    p.add_argument("--depart-iso", type=str, default=None,
                   help="Departure time (UTC ISO8601), required for env-source=openmeteo")
    p.add_argument("--out-dir", type=str, default="outputs", help="Directory to write outputs")
    p.add_argument("--out-prefix", type=str, default="route_out")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.local_tif:
        src = read_raster_from_file(args.local_tif)
    else:
        tif = fetch_geotiff_bytes(BBox(*args.bbox), dem_type=args.dem_type)
        src = read_raster_from_bytes(tif)

    arr, bounds = oriented_array_and_bounds(src, downsample=args.downsample)

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
            engine_info = []
            engine_kw = []
            for eng, load, fuel_g in zip(self.ves.engines, loads, res.get("fuel_used_g", [])):
                kw = load / 100.0 * eng.max_power
                engine_kw.append(kw)
                sfoc = eng.get_fuel_consumption(load)
                engine_info.append({"power_kw": kw, "sfoc_g_per_kwh": sfoc})
            P_prop_w = propulsion_power(environment, target_speed)
            total_prop_kw = P_prop_w / 1000.0
            hotel_kw = hotel_power(environment) / 1000.0
            aux_kw = aux_power(environment, P_prop_w) / 1000.0
            return {
                "achieved_speed_knots": res.get("actual_speed", 0.0),
                "fuel_consumed_kg": sum(res.get("fuel_used_g", [])) / 1000.0,
                "engines": engine_info,
                "total_propulsion_power_kw": total_prop_kw,
                "hotel_kw": hotel_kw,
                "aux_kw": aux_kw,
            }

    if args.env_source == "constant":
        env_series = [
            {
                "wind_speed": args.wind_speed,
                "wind_angle_diff": args.wind_angle_diff,
                "wave_height": args.wave_height,
            }
            for _ in path_ll
        ]
    else:
        if not args.depart_iso:
            raise SystemExit("--depart-iso is required when --env-source=openmeteo")
        env_series = sample_env_along_route(path_ll, args.depart_iso, args.target_speed_kn)

    profile = feasible_speed_profile(
        _VESAdapter(ves), path_ll, target_speed_knots=args.target_speed_kn,
        dt_s=args.dt_s, env_const=env_series
    )

    # Save enriched CSV
    csv_path = os.path.join(args.out_dir, f"{args.out_prefix}.csv")
    with open(csv_path, "w", newline="") as f:
        max_e = max((len(r.get("per_engine_kw", [])) for r in profile["segments"]), default=0)
        base_fields = [
            "i", "lat", "lon", "v_kn", "seg_nm", "t_s", "t_total_s", "fuel_kg",
            "fuel_total_kg", "total_prop_kw", "hotel_kw", "aux_kw",
            "env_wind_speed", "env_wind_angle_diff", "env_wave_height"
        ]
        fieldnames = base_fields + [f"e{j}_kw" for j in range(max_e)] + [f"e{j}_sfoc_g_per_kwh" for j in range(max_e)]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in profile["segments"]:
            row = {k: r.get(k, "") for k in base_fields}
            for j in range(max_e):
                row[f"e{j}_kw"] = (
                    r.get("per_engine_kw", [None] * max_e)[j]
                    if j < len(r.get("per_engine_kw", []))
                    else ""
                )
                row[f"e{j}_sfoc_g_per_kwh"] = (
                    r.get("per_engine_sfoc_g_per_kwh", [None] * max_e)[j]
                    if j < len(r.get("per_engine_sfoc_g_per_kwh", []))
                    else ""
                )
            w.writerow(row)

    # Save GeoJSON with totals
    gj_path = os.path.join(args.out_dir, f"{args.out_prefix}.geojson")
    gj = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in path_ll]},
            "properties": {"min_depth_m": min_depth, **profile["totals"]},
        }],
    }
    with open(gj_path, "w") as f:
        json.dump(gj, f)

    # Save static route plot
    png_route = os.path.join(args.out_dir, f"{args.out_prefix}.png")
    plt.figure()
    if path_ll:
        lats = [lat for lat, lon in path_ll]
        lons = [lon for lat, lon in path_ll]
        plt.plot(lons, lats)
    plt.title("Planned Route (lat/lon)")
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.savefig(png_route, dpi=180)
    plt.close()

    # Interactive map
    if args.map_html:
        map_path = os.path.join(args.out_dir, args.map_html)
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
            folium.PolyLine([(lat, lon) for lat, lon in path_ll], color="#0066FF", weight=4, opacity=0.9, tooltip="Route").add_to(m)
            folium.Marker(location=list(path_ll[0]), popup="Start").add_to(m)
            folium.Marker(location=list(path_ll[-1]), popup="Goal").add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        m.save(map_path)
        if args.verbose:
            print(f"[DEBUG] map -> {map_path}, visible bathy px = {visible}")

    # Additional time-series plots
    ts = [r["t_total_s"] / 3600.0 for r in profile["segments"]]
    speed_kn = [r["v_kn"] for r in profile["segments"]]
    fuel_total = [r["fuel_total_kg"] for r in profile["segments"]]
    total_kw = [r["total_prop_kw"] for r in profile["segments"]]
    hotel_kw = [r["hotel_kw"] for r in profile["segments"]]
    aux_kw = [r["aux_kw"] for r in profile["segments"]]
    wind = [r["env_wind_speed"] for r in profile["segments"]]
    wind_diff = [r["env_wind_angle_diff"] for r in profile["segments"]]
    wave_h = [r["env_wave_height"] for r in profile["segments"]]

    _save_plot(ts, fuel_total, "Fuel cumulative vs time", "time [h]", "fuel [kg]",
               os.path.join(args.out_dir, f"{args.out_prefix}_fuel_vs_time.png"))
    _save_plot(ts, speed_kn, "Speed vs time", "time [h]", "speed [kn]",
               os.path.join(args.out_dir, f"{args.out_prefix}_speed_vs_time.png"))
    _save_plot(ts, total_kw, "Total propulsion power vs time", "time [h]", "power [kW]",
               os.path.join(args.out_dir, f"{args.out_prefix}_total_power_vs_time.png"))
    _save_plot(ts, hotel_kw, "Hotel power vs time", "time [h]", "power [kW]",
               os.path.join(args.out_dir, f"{args.out_prefix}_hotel_power_vs_time.png"))
    _save_plot(ts, aux_kw, "Aux power vs time", "time [h]", "power [kW]",
               os.path.join(args.out_dir, f"{args.out_prefix}_aux_power_vs_time.png"))

    max_e = 0
    for r in profile["segments"]:
        max_e = max(max_e, len(r.get("per_engine_kw", [])))
    for j in range(max_e):
        ej = [
            (r.get("per_engine_kw", [None] * max_e)[j] if j < len(r.get("per_engine_kw", [])) else 0.0)
            for r in profile["segments"]
        ]
        _save_plot(ts, ej, f"Engine {j} power vs time", "time [h]", "power [kW]",
                   os.path.join(args.out_dir, f"{args.out_prefix}_engine{j}_power_vs_time.png"))

    for j in range(max_e):
        sfocj = [
            (
                r.get("per_engine_sfoc_g_per_kwh", [None] * max_e)[j]
                if j < len(r.get("per_engine_sfoc_g_per_kwh", []))
                else 0.0
            )
            for r in profile["segments"]
        ]
        _save_plot(ts, sfocj, f"Engine {j} SFOC vs time", "time [h]", "SFOC [g/kWh]",
                   os.path.join(args.out_dir, f"{args.out_prefix}_engine{j}_sfoc_vs_time.png"))

    _save_plot(ts, wind, "Wind speed vs time", "time [h]", "wind [m/s]",
               os.path.join(args.out_dir, f"{args.out_prefix}_wind_vs_time.png"))
    _save_plot(ts, wind_diff, "Wind angle diff vs time", "time [h]", "angle [deg]",
               os.path.join(args.out_dir, f"{args.out_prefix}_wind_angle_vs_time.png"))
    _save_plot(ts, wave_h, "Wave height vs time", "time [h]", "Hs [m]",
               os.path.join(args.out_dir, f"{args.out_prefix}_wave_vs_time.png"))

    if args.verbose:
        import numpy as np
        total = grid.size
        obst = int(grid.sum())
        free = total - obst
        print(f"[DEBUG] CRS={src.crs}, bounds(S,W,N,E)={bounds}")
        print(f"[DEBUG] bathy: min={np.nanmin(arr):.2f}, max={np.nanmax(arr):.2f}, NaN%={np.isnan(arr).mean()*100:.1f}%")
        print(f"[DEBUG] grid shape={grid.shape}, free={free} ({free/total*100:.1f}%), obst={obst} ({obst/total*100:.1f}%)")
        print(f"[DEBUG] path_rc length={len(path_rc)}, path_ll length={len(path_ll)}")
        print(
            f"[DEBUG] totals: nm={profile['totals']['nm']:.2f}, time_h={profile['totals']['time_s']/3600.0:.2f}, fuel_kg={profile['totals']['fuel_kg']:.2f}"
        )
        print(f"[DEBUG] wrote: {gj_path} {csv_path} {png_route}")


if __name__ == "__main__":
    main()

