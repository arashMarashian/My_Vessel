from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import warnings

import numpy as np

from my_vessel.pipeline.route_from_bathy import plan_route
from my_vessel.pipeline.speed_profile import feasible_speed_profile
from my_vessel.bathy.grid import rc_to_latlon
from environment.map_utils import extract_environment_along_path, load_environment_data
from my_vessel.energy.vessel_energy_system import (
    Battery,
    VesselEnergySystem,
    hotel_power,
    aux_power,
)
from my_vessel.energy.power_model import propulsion_power
from engine_loader import Engine, load_engines_from_yaml


def _default_sfoc_curve(base: float, slope: float = 0.08):
    def curve(load_percent: float) -> float:
        lp = max(0.0, min(100.0, float(load_percent)))
        return base + slope * (100.0 - lp)

    return curve


def _default_engines() -> List[Engine]:
    return [
        Engine(
            name="fallback_main",
            max_power=1500.0,
            min_load=20.0,
            max_load=90.0,
            sfoc_curve={"HFO": _default_sfoc_curve(195.0)},
            startup_cost=0.0,
        ),
        Engine(
            name="fallback_aux",
            max_power=1000.0,
            min_load=25.0,
            max_load=95.0,
            sfoc_curve={"HFO": _default_sfoc_curve(198.0, slope=0.05)},
            startup_cost=0.0,
        ),
    ]


def _resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _load_bathy(bathy_cfg: Dict[str, Any], base_dir: Path) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    if "array" in bathy_cfg:
        arr = np.asarray(bathy_cfg["array"], dtype="float32")
    else:
        path_key = next((k for k in ("npy", "npz") if k in bathy_cfg), None)
        if not path_key:
            raise ValueError("bathy config must provide 'array' or 'npy'/'npz' path")
        path = _resolve_path(bathy_cfg[path_key], base_dir)
        if not path:
            raise ValueError(f"Unable to resolve bathy path from key '{path_key}'")
        if path.suffix == ".npz":
            data = np.load(path)
            first_key = next(iter(data.files))
            arr = data[first_key].astype("float32")
        else:
            arr = np.load(path).astype("float32")
    if arr.ndim != 2:
        raise ValueError("bathymetry array must be 2-D")
    bounds = bathy_cfg.get("bounds")
    if not bounds or len(bounds) != 4:
        raise ValueError("bathy.bounds must contain [south, west, north, east]")
    return arr, tuple(float(b) for b in bounds)


def _ensure_len(seq: Sequence[Dict[str, float]], length: int) -> List[Dict[str, float]]:
    if length <= len(seq):
        return [copy.deepcopy(item) for item in seq[:length]]
    if not seq:
        return [dict() for _ in range(length)]
    extra = [copy.deepcopy(seq[-1]) for _ in range(length - len(seq))]
    return [copy.deepcopy(item) for item in seq] + extra


def _env_series_and_samples(
    env_cfg: Dict[str, Any],
    path_rc: Sequence[Tuple[int, int]],
    path_len: int,
    base_dir: Path,
) -> tuple[List[Dict[str, float]], Dict[str, List[float]] | None]:
    mode = env_cfg.get("mode", "constant")
    samples: Dict[str, List[float]] | None = None
    if mode == "grid":
        path = _resolve_path(env_cfg.get("npz"), base_dir)
        fields = env_cfg.get("fields")
        if fields is None and path is None:
            raise ValueError("grid environment requires 'npz' path or inline 'fields'")
        if path is not None:
            env_data = load_environment_data(str(path))
        else:
            env_data = {k: np.asarray(v) for k, v in fields.items()}
        coords = [(c, r) for r, c in path_rc]
        samples = extract_environment_along_path(env_data, coords)
        env_series = []
        for i in range(max(path_len, 1)):
            idx = min(i, len(coords) - 1)
            env_series.append({k: float(vals[idx]) for k, vals in samples.items()})
    else:
        values = {k: float(v) for k, v in env_cfg.get("values", {}).items()}
        if not values:
            values = {"wind_speed": 0.0, "wind_angle_diff": 0.0, "wave_height": 0.0}
        env_series = [copy.deepcopy(values) for _ in range(max(path_len, 1))]
        samples = {k: [float(v) for _ in path_rc] for k, v in values.items()}
    env_series = _ensure_len(env_series, max(path_len, 1))
    return env_series, samples


class _VESAdapter:
    def __init__(self, ves: VesselEnergySystem) -> None:
        self.ves = ves

    def step(self, environment: Dict[str, float], target_speed: float, timestep_seconds: float) -> Dict[str, Any]:
        P_prop_w = propulsion_power(environment, target_speed)
        hotel_kw = hotel_power(environment) / 1000.0
        aux_kw = aux_power(environment, P_prop_w) / 1000.0
        total_prop_kw = P_prop_w / 1000.0
        total_power_kw = total_prop_kw + hotel_kw + aux_kw

        loads: List[float] = []
        for eng in self.ves.engines:
            if total_power_kw <= 0:
                loads.append(0.0)
                continue
            load_pct = total_power_kw / max(eng.max_power, 1e-6) * 100.0 / len(self.ves.engines)
            load_pct = max(0.0, load_pct)
            if load_pct > 0:
                load_pct = max(eng.min_load, min(eng.max_load, load_pct))
            loads.append(load_pct)

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
        for eng, load in zip(self.ves.engines, loads):
            kw = load / 100.0 * eng.max_power
            sfoc = 0.0 if load <= 0 else eng.get_fuel_consumption(load)
            engine_info.append({"power_kw": kw, "sfoc_g_per_kwh": sfoc})

        return {
            "achieved_speed_knots": res.get("actual_speed", target_speed),
            "fuel_consumed_kg": sum(res.get("fuel_used_g", [])) / 1000.0,
            "engines": engine_info,
            "total_propulsion_power_kw": total_prop_kw,
            "hotel_kw": hotel_kw,
            "aux_kw": aux_kw,
            "total_power_kw": total_power_kw,
            "battery_power_kw": float(res.get("battery_power_w", 0.0)) / 1000.0,
            "battery_soc_kwh": float(res.get("battery_soc_kwh", 0.0)),
        }


def run_scenario(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a lightweight routing + energy evaluation scenario."""
    if "start" not in config or "goal" not in config:
        raise ValueError("config must include 'start' and 'goal'")

    base_dir = Path(config.get("_base_dir") or ".").resolve()
    bathy_cfg = config.get("bathy", {})
    arr_bathy, bounds = _load_bathy(bathy_cfg, base_dir)

    start_latlon = tuple(config["start"])
    goal_latlon = tuple(config["goal"])

    planning_cfg = config.get("planning", {})
    min_depth = float(bathy_cfg.get("min_depth_m", planning_cfg.get("min_depth_m", 5.0)))
    dilate = int(bathy_cfg.get("dilate_cells", planning_cfg.get("dilate_cells", 0)))

    densify = int(planning_cfg.get("densify_pts", 4))
    smoothness = float(planning_cfg.get("smoothness", 0.3))
    iterations = int(planning_cfg.get("iterations", 200))
    snap_radius = int(planning_cfg.get("snap_radius", 50))
    snap_endpoints = bool(planning_cfg.get("snap_endpoints", True))
    verbose = bool(planning_cfg.get("verbose", False))

    t0 = time.perf_counter()
    grid, path_rc, path_ll = plan_route(
        arr_bathy,
        bounds,
        start_latlon,
        goal_latlon,
        min_depth_m=min_depth,
        dilate_cells=dilate,
        densify_pts=densify,
        smoothness=smoothness,
        iterations=iterations,
        snap_endpoints=snap_endpoints,
        snap_radius=snap_radius,
        verbose=verbose,
    )

    if not path_rc:
        raise RuntimeError("Planner could not find a path for the given scenario")

    raw_path_ll = [rc_to_latlon(r, c, bounds, grid.shape) for r, c in path_rc]

    env_cfg = config.get("environment", {})
    env_series, env_samples = _env_series_and_samples(env_cfg, path_rc, len(path_ll), base_dir)

    energy_cfg = config.get("energy", {})
    engine_yaml = energy_cfg.get("engine_yaml")
    engines: List[Engine] = []
    fallback_reason: str | None = None
    if engine_yaml:
        engine_path = _resolve_path(engine_yaml, base_dir)
        if engine_path and engine_path.exists():
            engines = load_engines_from_yaml(str(engine_path))
        else:
            fallback_reason = f"engine file not found at {engine_path}" if engine_path else "engine path could not be resolved"
    else:
        fallback_reason = "engine_yaml not provided in config"

    if not engines:
        warnings.warn(
            f"{fallback_reason or 'engine configuration produced no engines'}; using built-in demo engines.",
            RuntimeWarning,
            stacklevel=2,
        )
        engines = _default_engines()

    max_engines = int(energy_cfg.get("max_engines", 0))
    if max_engines > 0:
        engines = engines[:max_engines]
    if not engines:
        raise ValueError("No engines available for energy simulation")

    battery_cfg = energy_cfg.get("battery", {})
    soc_val = battery_cfg.get("soc_kwh")
    battery = Battery(
        capacity_kwh=float(battery_cfg.get("capacity_kwh", 2000.0)),
        soc_kwh=None if soc_val is None else float(soc_val),
        charge_eff=float(battery_cfg.get("charge_eff", 0.95)),
        discharge_eff=float(battery_cfg.get("discharge_eff", 0.95)),
    )
    ves = VesselEnergySystem(engines, battery)

    adapter = _VESAdapter(ves)
    target_speed_kn = float(energy_cfg.get("target_speed_kn", 12.0))
    dt_s = float(energy_cfg.get("dt_s", 60.0))

    profile = feasible_speed_profile(
        adapter,
        path_ll,
        target_speed_knots=target_speed_kn,
        dt_s=int(dt_s),
        env_const=env_series,
    )

    runtime_s = time.perf_counter() - t0
    totals = profile.get("totals", {})
    metrics = {
        "path_cells": len(path_rc),
        "smoothed_points": len(path_ll),
        "distance_nm": float(totals.get("nm", 0.0)),
        "travel_time_s": float(totals.get("time_s", 0.0)),
        "fuel_kg": float(totals.get("fuel_kg", 0.0)),
        "runtime_s": runtime_s,
        "target_speed_kn": target_speed_kn,
    }

    result = {
        "grid": grid,
        "grid_bounds": bounds,
        "bathy": arr_bathy,
        "start": start_latlon,
        "goal": goal_latlon,
        "path": path_rc,
        "path_latlon": raw_path_ll,
        "smoothed_path": path_ll,
        "environment_series": env_series,
        "environment_samples": env_samples,
        "profile": profile,
        "metrics": metrics,
    }
    return result


__all__ = ["run_scenario"]
