from __future__ import annotations

import os
import io
import json
import hashlib
import sys
from pathlib import Path

# Ensure repository root on sys.path so we can import local packages when running from app/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import pandas as pd
import plotly.express as px

from utils.paths import ensure_results_subdir
from my_vessel.bathy.fetch import BBox, fetch_geotiff_bytes, read_raster_from_bytes
from my_vessel.bathy.grid import oriented_array_and_bounds
from my_vessel.bathy.overlay import make_overlay_data_url
from my_vessel.pipeline.route_from_bathy import plan_route
from my_vessel.environment.env_sources import sample_env_along_route
from my_vessel.pipeline.speed_profile import feasible_speed_profile
from energy.vessel_energy_system import (
    VesselEnergySystem,
    Battery,
    hotel_power,
    aux_power,
)
from energy.power_model import propulsion_power
from engine_loader import load_engines_from_yaml


st.set_page_config(page_title="Bathymetry Route Planner", layout="wide")

# Keep last successful results across reruns
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# --- Simple Login Gate ---
ADMIN_USER = "admin"
ADMIN_PASS_HASH = hashlib.sha256("ArashReza".encode("utf-8")).hexdigest()

if "auth_ok" not in st.session_state:
    st.session_state["auth_ok"] = False

def _login_ui():
    st.title("Login")
    st.caption("Please sign in to access the app")
    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
    if submit:
        ok = (u == ADMIN_USER) and (hashlib.sha256(p.encode("utf-8")).hexdigest() == ADMIN_PASS_HASH)
        if ok:
            st.session_state["auth_ok"] = True
            st.success("Logged in successfully")
            st.rerun()
        else:
            st.error("Invalid username or password")

if not st.session_state.get("auth_ok"):
    _login_ui()
    st.stop()


def _adapter_for_ves(ves: VesselEnergySystem, reserve_n_engines: int = 1):
    class _Adapter:
        def __init__(self, ves):
            self.ves = ves

        def step(self, environment, target_speed, timestep_seconds):
            P_prop_w = propulsion_power(environment, target_speed)
            hotel_kw = hotel_power(environment) / 1000.0
            aux_kw = aux_power(environment, P_prop_w) / 1000.0
            total_prop_kw = P_prop_w / 1000.0
            total_power_kw = total_prop_kw + hotel_kw + aux_kw

            # Staged dispatch: run the minimum number of engines (reserving some) and share load uniformly
            n = len(self.ves.engines)
            idx_sorted = sorted(range(n), key=lambda i: self.ves.engines[i].max_power, reverse=True)
            reserve = set(idx_sorted[-max(0, min(reserve_n_engines, n)) :])
            pool = [i for i in idx_sorted if i not in reserve]

            def try_k(indices):
                sum_max = sum(self.ves.engines[i].max_power for i in indices)
                share = 0.0 if sum_max <= 0 else total_power_kw / sum_max
                load_pct = share * 100.0
                # Check if within bounds for all selected engines
                mins = [self.ves.engines[i].min_load for i in indices]
                maxs = [self.ves.engines[i].max_load for i in indices]
                ok_upper = load_pct <= min(maxs)
                ok_lower = load_pct >= max(mins)
                return ok_lower and ok_upper, max(mins), min(maxs), load_pct

            active = []
            chosen_load_pct = None
            # Prefer not to use reserved engines
            for k in range(1, max(1, len(pool)) + 1):
                cand = pool[:k]
                ok, lo, hi, load_pct = try_k(cand)
                if ok:
                    active = cand
                    chosen_load_pct = load_pct
                    break
            # If still not ok, include reserved engines progressively
            if not active:
                all_sorted = idx_sorted
                for k in range(1, n + 1):
                    cand = all_sorted[:k]
                    ok, lo, hi, load_pct = try_k(cand)
                    if ok:
                        active = cand
                        chosen_load_pct = load_pct
                        break
            # Fallback: if nothing meets bounds even with all engines, use all engines at max load
            if not active:
                active = idx_sorted
                chosen_load_pct = max(e.min_load for e in self.ves.engines)

            target_loads = [0.0] * n
            if chosen_load_pct is not None:
                for i in active:
                    eng = self.ves.engines[i]
                    target_loads[i] = max(eng.min_load, min(eng.max_load, chosen_load_pct))
            else:
                # Very low demand: use a single largest engine at min load and charge battery
                i0 = idx_sorted[0]
                target_loads[i0] = self.ves.engines[i0].min_load

            engine_kw_targets = [ld / 100.0 * eng.max_power for ld, eng in zip(target_loads, self.ves.engines)]
            total_engine_kw = sum(engine_kw_targets)
            battery_req_w = (total_engine_kw - total_power_kw) * 1000.0

            res = self.ves.step(
                controller_action={
                    "engine_loads": target_loads,
                    "battery_power": battery_req_w,
                    "target_speed": target_speed,
                },
                environment=environment,
                timestep_hours=timestep_seconds / 3600.0,
            )
            engine_info = []
            for eng, load in zip(self.ves.engines, target_loads):
                kw = load / 100.0 * eng.max_power
                # When engine is off (load==0 or power==0), report SFOC as 0
                if load <= 0.0 or kw <= 0.0:
                    sfoc = 0.0
                else:
                    sfoc = eng.get_fuel_consumption(load)
                engine_info.append({"power_kw": kw, "sfoc_g_per_kwh": sfoc})
            battery_power_kw = float(res.get("battery_power_w", 0.0)) / 1000.0
            battery_soc_kwh = float(res.get("battery_soc_kwh", 0.0))
            return {
                "achieved_speed_knots": res.get("actual_speed", 0.0),
                "fuel_consumed_kg": sum(res.get("fuel_used_g", [])) / 1000.0,
                "engines": engine_info,
                "total_propulsion_power_kw": total_prop_kw,
                "hotel_kw": hotel_kw,
                "aux_kw": aux_kw,
                "total_power_kw": total_power_kw,
                "battery_power_kw": battery_power_kw,
                "battery_soc_kwh": battery_soc_kwh,
            }

    return _Adapter(ves)


def _df_from_profile(profile: Dict) -> pd.DataFrame:
    rows = profile.get("segments", [])
    df = pd.DataFrame(rows)
    if not df.empty:
        df["t_total_h"] = df["t_total_s"].astype(float) / 3600.0
        # Expand per-engine lists into columns for plotting
        max_e_kw = max((len(r.get("per_engine_kw", [])) for r in rows), default=0)
        for j in range(max_e_kw):
            df[f"e{j}_kw"] = [
                (r.get("per_engine_kw", [None] * max_e_kw)[j] if j < len(r.get("per_engine_kw", [])) else None)
                for r in rows
            ]
        max_e_sfoc = max((len(r.get("per_engine_sfoc_g_per_kwh", [])) for r in rows), default=0)
        for j in range(max_e_sfoc):
            df[f"e{j}_sfoc_g_per_kwh"] = [
                (
                    r.get("per_engine_sfoc_g_per_kwh", [None] * max_e_sfoc)[j]
                    if j < len(r.get("per_engine_sfoc_g_per_kwh", []))
                    else None
                )
                for r in rows
            ]
    return df


def _parse_drawings(drawings: List[Dict]) -> Tuple[Optional[Tuple[float, float, float, float]], List[Tuple[float, float]]]:
    bbox = None
    pts: List[Tuple[float, float]] = []
    for d in drawings or []:
        geom = d.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if gtype == "Polygon" and coords:
            # rectangle polygon in lon/lat
            lats = [p[1] for p in coords[0]]
            lons = [p[0] for p in coords[0]]
            S, N = min(lats), max(lats)
            W, E = min(lons), max(lons)
            bbox = (S, W, N, E)
        elif gtype == "Point" and coords:
            lon, lat = coords
            pts.append((float(lat), float(lon)))
    return bbox, pts


def run_pipeline(
    bbox: Tuple[float, float, float, float],
    start: Tuple[float, float],
    goal: Tuple[float, float],
    *,
    draft: float,
    ukc: float,
    dilate_cells: int,
    downsample: int,
    dem_type: str,
    env_source: str,
    env_sample_stride: int,
    depart_iso: Optional[str],
    target_speed_kn: float,
    dt_s: int,
    engine_yaml: str = "data/engine_data.yaml",
    out_prefix: str = "route_out",
    report: Optional[callable] = None,
    dem_bytes: Optional[bytes] = None,
    api_key: Optional[str] = None,
    use_battery: bool = False,
    bat_cap_kwh: float = 0.0,
    bat_soc_pct: float = 50.0,
    bat_eta_c_pct: float = 95.0,
    bat_eta_d_pct: float = 95.0,
) -> Tuple[Dict, List[Tuple[float, float]], Tuple[float, float, float, float], any]:
    os.environ.setdefault("MPLBACKEND", "Agg")
    out_dir = ensure_results_subdir("bathy_route")

    # Fetch DEM
    if report:
        report("Downloading DEM…")
    if dem_bytes is None:
        tif = fetch_geotiff_bytes(BBox(*bbox), dem_type=dem_type, timeout=30, api_key=api_key)
    else:
        tif = dem_bytes
    if report:
        report("Opening raster…")
    src = read_raster_from_bytes(tif)
    arr, bounds = oriented_array_and_bounds(src, downsample=downsample)

    # Plan route
    if report:
        report("Planning route…")
    min_depth = float(draft) + float(ukc)
    grid, path_rc, path_ll = plan_route(
        arr, bounds, start, goal,
        min_depth_m=min_depth, dilate_cells=int(dilate_cells),
        densify_pts=4, smoothness=0.3, iterations=200, snap_radius=50,
        verbose=False,
    )

    # Energy system
    engines = load_engines_from_yaml(engine_yaml)
    if use_battery and bat_cap_kwh > 0:
        bat = Battery(
            capacity_kwh=float(bat_cap_kwh),
            soc_kwh=float(bat_cap_kwh) * float(bat_soc_pct) / 100.0,
            charge_eff=float(bat_eta_c_pct) / 100.0,
            discharge_eff=float(bat_eta_d_pct) / 100.0,
        )
    else:
        bat = Battery(capacity_kwh=0.0)
    ves = VesselEnergySystem(engines, battery=bat)
    adapter = _adapter_for_ves(ves)

    # Environment series
    if report:
        report("Sampling environment…")
    if env_source == "openmeteo":
        if not depart_iso:
            raise ValueError("depart_iso is required for env_source=openmeteo")
        env_series = sample_env_along_route(path_ll, depart_iso, target_speed_kn, sample_stride=env_sample_stride)
    else:
        env_series = [{"wind_speed": 0.0, "wind_angle_diff": 0.0, "wave_height": 0.0} for _ in path_ll]

    if report:
        report("Building speed/power profile…")
    profile = feasible_speed_profile(
        adapter, path_ll, target_speed_knots=target_speed_kn, dt_s=dt_s, env_const=env_series
    )

    # Save CSV & GeoJSON like CLI for parity
    if report:
        report("Saving outputs…")
    csv_path = os.path.join(out_dir, f"{out_prefix}.csv")
    df = _df_from_profile(profile)
    if not df.empty:
        # expand engine columns
        max_e = max((len(r.get("per_engine_kw", [])) for r in profile["segments"]), default=0)
        for j in range(max_e):
            df[f"e{j}_kw"] = [
                (r.get("per_engine_kw", [None] * max_e)[j] if j < len(r.get("per_engine_kw", [])) else None)
                for r in profile["segments"]
            ]
            df[f"e{j}_sfoc_g_per_kwh"] = [
                (r.get("per_engine_sfoc_g_per_kwh", [None] * max_e)[j] if j < len(r.get("per_engine_sfoc_g_per_kwh", [])) else None)
                for r in profile["segments"]
            ]
        df.to_csv(csv_path, index=False)

    gj = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in path_ll]},
            "properties": {"min_depth_m": min_depth, **profile.get("totals", {})},
        }],
    }
    gj_path = os.path.join(out_dir, f"{out_prefix}.geojson")
    with open(gj_path, "w") as f:
        json.dump(gj, f)

    return profile, path_ll, bounds, arr


def _render_route_map(arr, bounds, path_ll):
    data_url, (S, W, N, E), visible = make_overlay_data_url(arr, bounds)
    center = list(path_ll[0]) if path_ll else [(bounds[2] + bounds[0]) / 2, (bounds[1] + bounds[3]) / 2]
    m = folium.Map(location=center, zoom_start=6, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri.WorldImagery", name="Esri Satellite", control=True
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(m)
    folium.raster_layers.ImageOverlay(
        image=data_url, bounds=[[S, W], [N, E]], opacity=0.6, name="Bathymetry"
    ).add_to(m)
    if path_ll:
        folium.PolyLine([(lat, lon) for lat, lon in path_ll], color="#0066FF", weight=4, opacity=0.9, tooltip="Route").add_to(m)
        folium.Marker(location=list(path_ll[0]), popup="Start").add_to(m)
        folium.Marker(location=list(path_ll[-1]), popup="Goal").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, height=500, width=None)


def _plot_standard(df: pd.DataFrame):
    cols = {
        "Cumulative Fuel [kg]": ("t_total_h", "fuel_total_kg"),
        "Speed [kn]": ("t_total_h", "v_kn"),
        "Total Power [kW]": ("t_total_h", "total_power_kw" if "total_power_kw" in df else "total_prop_kw"),
    }
    plots = []
    for title, (xcol, ycol) in cols.items():
        if xcol in df and ycol in df:
            fig = px.line(df, x=xcol, y=ycol, title=title, labels={xcol: "time [h]"})
            plots.append(fig)
            st.plotly_chart(fig, use_container_width=True)
    return plots


def _available_extra_series(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Return (label->column mapping) and composite overlays (label->list of columns)."""
    label_to_col: Dict[str, str] = {}
    overlays: Dict[str, List[str]] = {}
    engine_power_cols = []
    engine_sfoc_cols = []
    # Detect engine columns and build friendly labels
    for c in df.columns:
        if c.startswith("e") and c.endswith("_kw"):
            try:
                j = int(c[1:c.index("_")])
                label_to_col[f"Engine {j+1} Power [kW]"] = c
                engine_power_cols.append(c)
            except Exception:
                pass
        if c.startswith("e") and c.endswith("_sfoc_g_per_kwh"):
            try:
                j = int(c[1:c.index("_")])
                label_to_col[f"Engine {j+1} SFOC [g/kWh]"] = c
                engine_sfoc_cols.append(c)
            except Exception:
                pass
    if engine_power_cols:
        overlays["Engine Power (all)"] = engine_power_cols
    if engine_sfoc_cols:
        overlays["Engine SFOC (all)"] = engine_sfoc_cols

    # Other time series
    other = [
        ("Hotel Power [kW]", "hotel_kw"),
        ("Aux Power [kW]", "aux_kw"),
        ("Battery SOC [kWh]", "battery_soc_kwh"),
        ("Battery Power [kW]", "battery_power_kw"),
        ("Wind Speed [m/s]", "env_wind_speed"),
        ("Wind Angle Diff [deg]", "env_wind_angle_diff"),
        ("Wave Height [m]", "env_wave_height"),
    ]
    for label, col in other:
        if col in df.columns:
            label_to_col[label] = col

    return label_to_col, overlays


st.title("Bathymetry Route Planner")
st.caption("Draw a bounding box and start/goal on the map, set parameters, and run.")

with st.sidebar:
    if st.button("Logout"):
        st.session_state["auth_ok"] = False
        st.rerun()
    st.header("Inputs")
    api_key = st.text_input("OpenTopography API Key", value=os.getenv("OPENTOPO_API_KEY", ""), type="password")
    dem_type = st.text_input("DEM Type", value="SRTM15Plus")
    downsample = st.number_input("Downsample factor", min_value=1, max_value=64, value=16, step=1)
    draft = st.number_input("Draft [m]", min_value=0.0, value=3.0, step=0.1)
    ukc = st.number_input("Under-keel clearance [m]", min_value=0.0, value=0.5, step=0.1)
    dilate_cells = st.number_input("Obstacle dilation [px]", min_value=0, value=0, step=1)
    target_speed_kn = st.number_input("Target speed [kn]", min_value=0.0, value=20.0, step=0.5)
    dt_s = st.number_input("Timestep [s]", min_value=10, max_value=3600, value=60, step=10)
    env_source = st.selectbox("Environment source", options=["openmeteo", "constant"], index=0)
    env_stride = st.number_input("Env sample stride", min_value=1, max_value=200, value=20, step=1)
    depart_iso = None
    if env_source == "openmeteo":
        depart_iso = st.text_input("Departure time (UTC ISO8601)", value="2025-08-27T10:00Z")
    engine_yaml = st.text_input("Engine YAML", value="data/engine_data.yaml")
    use_battery = st.checkbox("Use battery system", value=False)
    if use_battery:
        st.markdown("Battery parameters")
        bat_cap = st.number_input("Capacity [kWh]", min_value=0.0, value=1000.0, step=50.0)
        bat_soc_pct = st.number_input("Initial SOC [%]", min_value=0.0, max_value=100.0, value=50.0, step=5.0)
        bat_eta_c = st.number_input("Charge efficiency [%]", min_value=1.0, max_value=100.0, value=95.0, step=1.0)
        bat_eta_d = st.number_input("Discharge efficiency [%]", min_value=1.0, max_value=100.0, value=95.0, step=1.0)
    else:
        bat_cap = 0.0
        bat_soc_pct = 50.0
        bat_eta_c = 95.0
        bat_eta_d = 95.0
    out_prefix = st.text_input("Output prefix", value="hel_to_rey")
    st.caption("Optional: upload a local GeoTIFF to bypass remote fetch.")
    local_tif = st.file_uploader("Local GeoTIFF (optional)", type=["tif", "tiff"], accept_multiple_files=False)
    debug = st.checkbox("Show debug steps", value=True)
    run_btn = st.button("Run Routing")
    clear_btn = st.button("Clear results")


st.subheader("Draw BBox and Start/Goal")
default_center = [60.1699, 24.9384]
base_map = folium.Map(location=default_center, zoom_start=4)
Draw(
    export=False,
    position="topleft",
    draw_options={
        "polyline": False, "polygon": False, "circle": False, "circlemarker": False,
        "marker": True, "rectangle": True
    },
    edit_options={"edit": True, "remove": True},
).add_to(base_map)
draw_state = st_folium(base_map, height=520, width=None)

all_drawings = draw_state.get("all_drawings", []) if isinstance(draw_state, dict) else []
bbox_drawn, pts = _parse_drawings(all_drawings)

with st.expander("Manual inputs (optional)"):
    st.write("Use these if you prefer typing or to override map selections.")
    s_lat = st.number_input("Start lat", value=60.1699, format="%.6f")
    s_lon = st.number_input("Start lon", value=24.9384, format="%.6f")
    g_lat = st.number_input("Goal lat", value=64.1466, format="%.6f")
    g_lon = st.number_input("Goal lon", value=-21.9426, format="%.6f")
    bbox_manual = st.text_input("BBox S W N E (space-separated)", value="52 -1 66 26")

st.markdown("—")
col1, col2 = st.columns(2)
with col1:
    st.write("Drawn markers (choose which are start/goal):")
    if pts:
        df_pts = pd.DataFrame(pts, columns=["lat", "lon"]).reset_index().rename(columns={"index": "idx"})
        st.dataframe(df_pts, hide_index=True, use_container_width=True)
        start_idx = st.number_input("Start marker idx", value=0, min_value=0, max_value=len(pts) - 1, step=1)
        goal_idx = st.number_input("Goal marker idx", value=min(1, len(pts) - 1), min_value=0, max_value=len(pts) - 1, step=1)
        start = pts[start_idx]
        goal = pts[goal_idx]
    else:
        start = (s_lat, s_lon)
        goal = (g_lat, g_lon)
    st.write(f"Selected start: {start}")
    st.write(f"Selected goal: {goal}")

with col2:
    st.write("BBox selection:")
    if bbox_drawn:
        bbox = bbox_drawn
    else:
        try:
            parts = [float(x) for x in (bbox_manual.strip().split())]
            if len(parts) == 4:
                bbox = tuple(parts)  # type: ignore
            else:
                bbox = None
        except Exception:
            bbox = None
    st.write(f"Selected bbox: {bbox}")


if clear_btn:
    st.session_state["last_result"] = None

if run_btn:
    if not bbox:
        st.error("Please draw a rectangle on the map or enter a valid bbox.")
    else:
        if api_key:
            os.environ["OPENTOPO_API_KEY"] = api_key
        with st.status("Running routing pipeline...", expanded=True) as status:
            try:
                prog = st.progress(0)
                step = {"i": 0}

                def report(msg: str):
                    step["i"] += 1
                    if debug:
                        st.write(msg)
                    prog.progress(min(100, step["i"] * 16))

                report("Starting…")
                dem_bytes = None
                if local_tif is not None:
                    report("Reading local GeoTIFF…")
                    dem_bytes = local_tif.read()
                profile, path_ll, bounds, arr = run_pipeline(
                    bbox, start, goal,
                    draft=draft, ukc=ukc, dilate_cells=int(dilate_cells), downsample=int(downsample),
                    dem_type=dem_type, env_source=env_source, env_sample_stride=int(env_stride),
                    depart_iso=depart_iso, target_speed_kn=float(target_speed_kn), dt_s=int(dt_s),
                    engine_yaml=engine_yaml, out_prefix=out_prefix, report=report, dem_bytes=dem_bytes,
                    api_key=api_key or os.getenv("OPENTOPO_API_KEY"),
                    use_battery=use_battery, bat_cap_kwh=bat_cap, bat_soc_pct=bat_soc_pct,
                    bat_eta_c_pct=bat_eta_c, bat_eta_d_pct=bat_eta_d,
                )
                # Persist results across reruns, then re-render outside the button block
                st.session_state["last_result"] = {
                    "profile": profile,
                    "path_ll": path_ll,
                    "bounds": bounds,
                    "arr": arr,
                    "out_prefix": out_prefix,
                }
                status.update(label="Run complete", state="complete")
                st.rerun()
            except Exception as e:
                st.exception(e)
                status.update(label="Run failed", state="error")


# Always render last results if present (survives reruns)
if st.session_state.get("last_result"):
    res = st.session_state["last_result"]
    profile = res["profile"]
    path_ll = res["path_ll"]
    bounds = res["bounds"]
    arr = res["arr"]
    out_prefix = res.get("out_prefix", "route_out")

    st.subheader(f"Results: {out_prefix}")
    _render_route_map(arr, bounds, path_ll)

    df = _df_from_profile(profile)
    if df.empty:
        st.warning("No segments produced; check bbox, start/goal, and draft/UKC settings.")
    else:
        st.subheader("Standard Plots")
        _plot_standard(df)

        st.subheader("Additional Plots")
        label_to_col, overlays = _available_extra_series(df)
        options = list(label_to_col.keys()) + list(overlays.keys())
        if options:
            selected = st.multiselect("Select series or overlays to plot", options=options, key="extra_plots")
            if selected:
                for key in selected:
                    if key in label_to_col:
                        y = label_to_col[key]
                        fig = px.line(df, x="t_total_h", y=y, title=key, labels={"t_total_h": "time [h]"})
                        st.plotly_chart(fig, use_container_width=True)
                    elif key in overlays:
                        cols = overlays[key]
                        fig = px.line(df, x="t_total_h", y=cols, title=key, labels={"t_total_h": "time [h]"})
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No additional series available in results.")
