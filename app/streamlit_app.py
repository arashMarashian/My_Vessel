from __future__ import annotations

import os
import hashlib
import sys
from pathlib import Path
# Ensure repository root on sys.path so we can import local packages when running from app/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from typing import Dict, List, Tuple, Optional

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import pandas as pd
import plotly.express as px

from my_vessel.bathy.fetch import BBox, fetch_geotiff_bytes, read_raster_from_bytes
from my_vessel.bathy.grid import oriented_array_and_bounds
from my_vessel.bathy.overlay import make_overlay_data_url
from my_vessel.pipeline.run_scenario import run_scenario


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
    engine_yaml: str,
    out_prefix: str,
    report: Optional[callable],
    dem_bytes: Optional[bytes],
    api_key: Optional[str],
    energy_mode: str,
    energy_policy: str,
    energy_load_profile: str,
    energy_dt_s: float,
    generator_params: Dict[str, float],
    battery_params: Optional[Dict[str, float]],
    hotel_load_kw: float,
    propulsion_constant_kw: Optional[float],
    propulsion_fallback_kw: float,
) -> Dict[str, Any]:
    os.environ.setdefault("MPLBACKEND", "Agg")

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

    min_depth = float(draft) + float(ukc)
    env_cfg: Dict[str, Any]
    if env_source == "openmeteo":
        if not depart_iso:
            raise ValueError("Departure time required for Open-Meteo sampling")
        env_cfg = {
            "mode": "openmeteo",
            "depart_iso": depart_iso,
            "sample_stride": int(env_sample_stride),
            "target_speed_kn": target_speed_kn,
        }
    else:
        env_cfg = {
            "mode": "constant",
            "values": {"wind_speed": 0.0, "wind_angle_diff": 0.0, "wave_height": 0.0},
        }

    energy_cfg: Dict[str, Any] = {
        "mode": energy_mode,
        "policy": energy_policy,
        "load_profile": energy_load_profile,
        "dt_s": energy_dt_s,
        "target_speed_kn": target_speed_kn,
        "engine_yaml": engine_yaml,
        "hotel_load_kw": hotel_load_kw,
        "propulsion_kw_fallback": propulsion_fallback_kw,
        "generator": generator_params,
    }
    if propulsion_constant_kw is not None:
        energy_cfg["propulsion_kw"] = propulsion_constant_kw
    if battery_params:
        energy_cfg["battery"] = battery_params

    cfg = {
        "name": out_prefix,
        "start": list(start),
        "goal": list(goal),
        "bathy": {
            "array": arr,
            "bounds": list(bounds),
            "min_depth_m": min_depth,
            "dilate_cells": int(dilate_cells),
        },
        "planning": {
            "densify_pts": 4,
            "smoothness": 0.3,
            "iterations": 200,
            "snap_radius": 50,
        },
        "environment": env_cfg,
        "energy": energy_cfg,
    }
    if report:
        report("Running scenario…")
    result = run_scenario(cfg)
    result["_bathy_array"] = arr
    result["_bounds"] = bounds
    return result


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


def _energy_df(energy: Dict[str, Any]) -> pd.DataFrame:
    load = energy.get("load_series") or []
    dt_s = float(energy.get("dt_s") or 0.0)
    if not load or dt_s <= 0:
        return pd.DataFrame()
    times = [dt_s * i for i in range(len(load))]
    p_gen = energy.get("p_gen_series") or []
    p_batt = energy.get("p_batt_series") or []
    soc = energy.get("soc_series") or []
    rows = []
    for idx, (t, load_kw) in enumerate(zip(times, load)):
        rows.append(
            {
                "time_s": t,
                "load_kw": load_kw,
                "p_gen_kw": p_gen[idx] if idx < len(p_gen) else 0.0,
                "p_batt_kw": p_batt[idx] if idx < len(p_batt) else 0.0,
                "soc": soc[idx] if idx < len(soc) else None,
            }
        )
    return pd.DataFrame(rows)


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
    st.markdown("Energy Simulation")
    energy_mode = st.selectbox("Energy mode", options=["diesel_only", "hybrid"], index=1)
    energy_policy = st.selectbox("Dispatch policy", options=["naive", "load_smoothing"], index=1)
    load_profile = st.selectbox(
        "Load profile",
        options=["constant", "synthetic_peaks", "from_speed"],
        index=1 if energy_mode == "hybrid" else 0,
    )
    energy_dt_s = st.number_input("Energy timestep [s]", min_value=5, max_value=3600, value=60, step=5)
    hotel_load_kw = st.number_input("Hotel load [kW]", min_value=0.0, value=150.0, step=10.0)
    propulsion_override = st.checkbox("Use constant propulsion load", value=False)
    if propulsion_override:
        propulsion_constant_kw = st.number_input("Propulsion load [kW]", min_value=0.0, value=600.0, step=25.0)
    else:
        propulsion_constant_kw = None
    propulsion_fallback_kw = st.number_input("Fallback propulsion load [kW]", min_value=10.0, value=500.0, step=10.0)
    st.markdown("Generator parameters")
    gen_p_max = st.number_input("Generator max power [kW]", min_value=100.0, value=1500.0, step=50.0)
    gen_p_min = st.number_input("Generator min power [kW]", min_value=10.0, value=300.0, step=10.0)
    battery_params = None
    if energy_mode == "hybrid":
        st.markdown("Battery parameters")
        bat_capacity = st.number_input("Capacity [kWh]", min_value=10.0, value=1200.0, step=50.0)
        bat_soc_init = st.number_input("Initial SOC [0-1]", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        bat_soc_min = st.number_input("SOC min [0-1]", min_value=0.0, max_value=0.9, value=0.2, step=0.05)
        bat_soc_max = st.number_input("SOC max [0-1]", min_value=0.1, max_value=1.0, value=0.95, step=0.05)
        bat_p_charge = st.number_input("Charge limit [kW]", min_value=10.0, value=300.0, step=10.0)
        bat_p_discharge = st.number_input("Discharge limit [kW]", min_value=10.0, value=300.0, step=10.0)
        bat_eta_c = st.number_input("Charge eff. [%]", min_value=10.0, max_value=100.0, value=96.0, step=1.0)
        bat_eta_d = st.number_input("Discharge eff. [%]", min_value=10.0, max_value=100.0, value=95.0, step=1.0)
        battery_params = {
            "capacity_kwh": bat_capacity,
            "soc_init": bat_soc_init,
            "soc_min": bat_soc_min,
            "soc_max": bat_soc_max,
            "p_charge_max_kw": bat_p_charge,
            "p_discharge_max_kw": bat_p_discharge,
            "eta_charge": bat_eta_c / 100.0,
            "eta_discharge": bat_eta_d / 100.0,
        }
    generator_params = {"p_max_kw": gen_p_max, "p_min_kw": gen_p_min}
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
                run_result = run_pipeline(
                    bbox,
                    start,
                    goal,
                    draft=draft,
                    ukc=ukc,
                    dilate_cells=int(dilate_cells),
                    downsample=int(downsample),
                    dem_type=dem_type,
                    env_source=env_source,
                    env_sample_stride=int(env_stride),
                    depart_iso=depart_iso,
                    target_speed_kn=float(target_speed_kn),
                    engine_yaml=engine_yaml,
                    out_prefix=out_prefix,
                    report=report,
                    dem_bytes=dem_bytes,
                    api_key=api_key or os.getenv("OPENTOPO_API_KEY"),
                    energy_mode=energy_mode,
                    energy_policy=energy_policy,
                    energy_load_profile=load_profile,
                    energy_dt_s=float(energy_dt_s),
                    generator_params=generator_params,
                    battery_params=battery_params,
                    hotel_load_kw=hotel_load_kw,
                    propulsion_constant_kw=propulsion_constant_kw,
                    propulsion_fallback_kw=propulsion_fallback_kw,
                )
                # Persist results across reruns, then re-render outside the button block
                st.session_state["last_result"] = {"data": run_result, "out_prefix": out_prefix}
                status.update(label="Run complete", state="complete")
                st.rerun()
            except Exception as e:
                st.exception(e)
                status.update(label="Run failed", state="error")


# Always render last results if present (survives reruns)
if st.session_state.get("last_result"):
    res = st.session_state["last_result"]
    data = res.get("data", {})
    profile = data.get("profile", {})
    path_ll = data.get("smoothed_path") or data.get("path_latlon") or []
    bounds = data.get("_bounds") or tuple(data.get("grid_bounds", ()))
    arr = data.get("_bathy_array")
    if arr is None:
        arr = data.get("bathy")
    out_prefix = res.get("out_prefix", "route_out")

    st.subheader(f"Results: {out_prefix}")
    if arr is not None and bounds:
        _render_route_map(arr, bounds, path_ll)
    else:
        st.warning("Missing bathymetry data for map rendering.")

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

        energy_result = data.get("energy_result")
        if energy_result:
            st.subheader("Energy Simulation")
            fuel_total = energy_result.get("fuel_kg_total", 0.0)
            soc_min = energy_result.get("soc_min")
            cols = st.columns(3)
            cols[0].metric("Fuel (dispatch)", f"{fuel_total:.2f} kg")
            cols[1].metric("Mode", energy_result.get("mode", "n/a"))
            if soc_min is not None:
                cols[2].metric("Min SOC", f"{soc_min*100:.1f}%")
            energy_df = _energy_df(energy_result)
            if not energy_df.empty:
                fig = px.line(
                    energy_df,
                    x="time_s",
                    y=["load_kw", "p_gen_kw", "p_batt_kw"],
                    labels={"value": "Power [kW]", "time_s": "Time [s]"},
                    title="Power Balance",
                )
                st.plotly_chart(fig, use_container_width=True)
                if energy_df["soc"].notna().any():
                    fig_soc = px.line(
                        energy_df.dropna(subset=["soc"]),
                        x="time_s",
                        y="soc",
                        labels={"soc": "State of Charge", "time_s": "Time [s]"},
                        title="Battery SOC",
                    )
                    st.plotly_chart(fig_soc, use_container_width=True)
            else:
                st.info("Energy time series unavailable.")
