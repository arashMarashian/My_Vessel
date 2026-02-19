from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import warnings

import numpy as np

from my_vessel.pipeline.route_from_bathy import plan_route
from my_vessel.pipeline.speed_profile import feasible_speed_profile
from my_vessel.bathy.grid import rc_to_latlon
from environment.map_utils import extract_environment_along_path, load_environment_data
from my_vessel.environment.env_sources import sample_env_along_route
from my_vessel.energy.vessel_energy_system import (
    Battery,
    VesselEnergySystem,
    hotel_power,
    aux_power,
)
from my_vessel.energy.diesel_generator import DieselGenerator
from my_vessel.energy.hybrid_system import HybridPowerSystem
from my_vessel.energy.battery_model import BatteryModel
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
    path_ll: Sequence[Tuple[float, float]],
    path_len: int,
    base_dir: Path,
    default_speed_kn: float,
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
    elif mode == "openmeteo":
        depart_iso = env_cfg.get("depart_iso")
        if not depart_iso:
            raise ValueError("environment.mode=openmeteo requires 'depart_iso'")
        stride = int(env_cfg.get("sample_stride", 1))
        tgt_speed = float(env_cfg.get("target_speed_kn", default_speed_kn))
        env_series = sample_env_along_route(list(path_ll), depart_iso, tgt_speed, sample_stride=stride)
        if env_series:
            samples = {k: [float(row.get(k, 0.0)) for row in env_series] for k in env_series[0].keys()}
        else:
            samples = None
    else:
        values = {k: float(v) for k, v in env_cfg.get("values", {}).items()}
        if not values:
            values = {"wind_speed": 0.0, "wind_angle_diff": 0.0, "wave_height": 0.0}
        env_series = [copy.deepcopy(values) for _ in range(max(path_len, 1))]
        samples = {k: [float(v) for _ in path_rc] for k, v in values.items()}
    env_series = _ensure_len(env_series, max(path_len, 1))
    return env_series, samples


def _load_from_profile_segments(profile: Dict[str, Any], dt_s: float) -> List[float] | None:
    segments = profile.get("segments", [])
    if not segments or dt_s <= 0:
        return None
    timeline: List[Tuple[float, float, float]] = []
    t_cursor = 0.0
    for seg in segments:
        duration = float(seg.get("t_s", 0.0))
        if duration <= 0:
            continue
        load_kw = float(
            seg.get(
                "total_power_kw",
                seg.get("total_prop_kw", 0.0) + seg.get("hotel_kw", 0.0) + seg.get("aux_kw", 0.0),
            )
        )
        timeline.append((t_cursor, t_cursor + duration, load_kw))
        t_cursor += duration
    if not timeline or t_cursor <= 0:
        return None
    steps = max(1, int(math.ceil(t_cursor / dt_s)))
    loads: List[float] = []
    idx = 0
    for step in range(steps):
        t = step * dt_s + 0.5 * dt_s
        while idx < len(timeline) and t >= timeline[idx][1]:
            idx += 1
        if idx >= len(timeline):
            load_kw = timeline[-1][2]
        else:
            load_kw = timeline[idx][2]
        loads.append(load_kw)
    return loads


def _apply_synthetic_peaks(load_series: Sequence[float]) -> List[float]:
    n = len(load_series)
    if n == 0:
        return []
    modulated: List[float] = []
    for i, base in enumerate(load_series):
        phase = 2.0 * math.pi * i / max(1, n - 1)
        wave = 0.8 + 0.35 * math.sin(phase) + 0.2 * math.sin(3.0 * phase + 0.5)
        pos = i / max(1, n - 1)
        peak1 = 0.25 * math.exp(-((pos - 0.3) / 0.08) ** 2)
        peak2 = 0.35 * math.exp(-((pos - 0.75) / 0.05) ** 2)
        factor = max(0.4, wave + peak1 + peak2)
        modulated.append(base * factor)
    return modulated


def _build_load_series(
    profile: Dict[str, Any],
    env_series: Sequence[Dict[str, float]],
    energy_cfg: Dict[str, Any],
    target_speed_kn: float,
) -> tuple[List[float], float]:
    dt_s = float(energy_cfg.get("dt_s", 60.0))
    if dt_s <= 0:
        raise ValueError("energy.dt_s must be positive")
    totals = profile.get("totals", {})
    total_time = float(totals.get("time_s", 0.0))
    if total_time <= 0:
        total_time = dt_s * max(1, len(env_series) or 1)
    steps = max(1, int(math.ceil(total_time / dt_s)))

    const_prop_kw = energy_cfg.get("propulsion_kw")
    fallback_kw = float(energy_cfg.get("propulsion_kw_fallback", 800.0))
    hotel_kw = float(energy_cfg.get("hotel_load_kw", 0.0))
    load_series: List[float] = []
    env_len = len(env_series)
    speed_m_s = target_speed_kn * 0.514444

    for idx in range(steps):
        env = env_series[min(idx, env_len - 1)] if env_len else {}
        if const_prop_kw is None:
            prop_kw = max(0.0, propulsion_power(env, speed_m_s) / 1000.0)
            if not math.isfinite(prop_kw) or prop_kw <= 0:
                prop_kw = fallback_kw
        else:
            prop_kw = float(const_prop_kw)
        load_series.append(prop_kw + hotel_kw)

    profile_mode = (energy_cfg.get("load_profile") or "constant").lower()
    if profile_mode == "from_speed":
        seg_series = _load_from_profile_segments(profile, dt_s)
        if seg_series:
            return seg_series, dt_s
    if profile_mode == "synthetic_peaks":
        return _apply_synthetic_peaks(load_series), dt_s
    return load_series, dt_s


def _diesel_only_result(
    generator: DieselGenerator,
    load_series: Sequence[float],
    dt_s: float,
    policy: str,
) -> Dict[str, Any]:
    p_gen_series: List[float] = []
    p_batt_series: List[float] = []
    fuel_total = 0.0
    for load_kw in load_series:
        p_gen = generator.clamp_power(load_kw)
        p_gen_series.append(p_gen)
        p_batt_series.append(load_kw - p_gen)
        fuel_total += generator.fuel_kg_per_s(p_gen) * dt_s
    return {
        "mode": "diesel_only",
        "policy": policy,
        "dt_s": dt_s,
        "fuel_kg_total": fuel_total,
        "soc_series": [],
        "p_gen_series": p_gen_series,
        "p_batt_series": p_batt_series,
        "load_series": list(load_series),
        "soc_min": None,
    }


def _run_energy_simulation(
    load_series: Sequence[float],
    dt_s: float,
    energy_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    if not load_series:
        return {}
    policy = energy_cfg.get("policy", "naive")
    generator_cfg = energy_cfg.get("generator", {})
    generator = DieselGenerator(
        p_max_kw=float(generator_cfg.get("p_max_kw", 2000.0)),
        p_min_kw=float(generator_cfg.get("p_min_kw", 200.0)),
        sfoc_curve=generator_cfg.get("sfoc_curve"),
    )

    mode = (energy_cfg.get("mode") or "diesel_only").lower()
    if mode == "hybrid":
        battery_cfg = energy_cfg.get("battery")
        if not battery_cfg:
            raise ValueError("energy.battery section required for hybrid mode")
        battery = BatteryModel(**battery_cfg)
        system = HybridPowerSystem(generator=generator, battery=battery)
        sim = system.simulate(load_series, dt_s=dt_s, policy=policy)
        soc_series = sim.get("soc_series", [])
        return {
            "mode": mode,
            "policy": policy,
            "dt_s": dt_s,
            "fuel_kg_total": sim.get("fuel_kg_total", 0.0),
            "soc_series": soc_series,
            "soc_min": min(soc_series) if soc_series else None,
            "p_gen_series": sim.get("p_gen_series", []),
            "p_batt_series": sim.get("p_batt_series", []),
            "load_series": list(load_series),
        }
    return _diesel_only_result(generator, load_series, dt_s, policy)


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

    energy_cfg = config.get("energy", {})
    target_speed_kn = float(energy_cfg.get("target_speed_kn", 12.0))

    env_cfg = config.get("environment", {})
    env_series, env_samples = _env_series_and_samples(
        env_cfg,
        path_rc,
        raw_path_ll,
        len(path_ll),
        base_dir,
        target_speed_kn,
    )
    # energy_cfg already defined above
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
    capacity_kwh = float(battery_cfg.get("capacity_kwh", 2000.0))
    soc_kwh = battery_cfg.get("soc_kwh")
    if soc_kwh is None and "soc_init" in battery_cfg:
        soc_kwh = float(battery_cfg["soc_init"]) * capacity_kwh
    charge_eff = float(battery_cfg.get("charge_eff", battery_cfg.get("eta_charge", 0.95)))
    discharge_eff = float(battery_cfg.get("discharge_eff", battery_cfg.get("eta_discharge", 0.95)))
    battery_kwargs: Dict[str, Any] = {
        "capacity_kwh": capacity_kwh,
        "soc_min": float(battery_cfg.get("soc_min", 0.0)),
        "soc_max": float(battery_cfg.get("soc_max", 1.0)),
        "p_charge_max_kw": battery_cfg.get("p_charge_max_kw"),
        "p_discharge_max_kw": battery_cfg.get("p_discharge_max_kw"),
        "eta_charge": charge_eff,
        "eta_discharge": discharge_eff,
    }
    if soc_kwh is not None:
        battery_kwargs["soc_kwh"] = float(soc_kwh)
    elif "soc_init" in battery_cfg:
        battery_kwargs["soc_init"] = float(battery_cfg["soc_init"])
    battery = Battery(**battery_kwargs)
    ves = VesselEnergySystem(engines, battery)

    adapter = _VESAdapter(ves)
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

    energy_result: Dict[str, Any] | None = None
    if energy_cfg.get("mode"):
        try:
            load_series, energy_dt = _build_load_series(profile, env_series, energy_cfg, target_speed_kn)
            energy_result = _run_energy_simulation(load_series, energy_dt, energy_cfg)
        except Exception as exc:
            warnings.warn(f"energy simulation failed: {exc}", RuntimeWarning, stacklevel=2)
            energy_result = None
    if energy_result:
        metrics["energy_fuel_kg"] = float(energy_result.get("fuel_kg_total", 0.0))

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
    if energy_result:
        result["energy_result"] = energy_result
    return result


__all__ = ["run_scenario"]
