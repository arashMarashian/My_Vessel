from __future__ import annotations
from typing import List, Dict, Tuple
import math


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    to_rad = lambda d: d * math.pi / 180.0
    dlat = to_rad(lat2 - lat1)
    dlon = to_rad(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(to_rad(lat1)) * math.cos(to_rad(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def feasible_speed_profile(ves, path_ll: List[Tuple[float, float]],
                           target_speed_knots: float, dt_s: int = 60,
                           env_const: Dict = None):
    """
    Step a VesselEnergySystem along the path. Returns dict with segments and totals.
    Assumes ves.step(environment, target_speed, timestep_seconds) -> dict with:
      - "achieved_speed_knots" and "fuel_consumed_kg" (per step)
    """
    if env_const is None:
        env_const = {}
    segs = []
    total_time_s = 0.0
    total_fuel_kg = 0.0
    total_nm = 0.0
    if len(path_ll) < 2:
        return {"segments": [], "total_time_s": 0.0, "total_fuel_kg": 0.0, "total_nm": 0.0}
    for i in range(1, len(path_ll)):
        lat1, lon1 = path_ll[i-1]
        lat2, lon2 = path_ll[i]
        dist_nm = haversine_km(lat1, lon1, lat2, lon2) * 0.539957
        # Try to hold target speed for dt; we scale outputs to segment duration
        step = ves.step(environment=env_const, target_speed=target_speed_knots, timestep_seconds=dt_s)
        v_ach = max(1e-6, step.get("achieved_speed_knots", target_speed_knots))
        time_s = (dist_nm / v_ach) * 3600.0
        # Scale fuel to segment time if step returns per-dt consumption
        fuel_per_dt = float(step.get("fuel_consumed_kg", 0.0))
        fuel_kg = fuel_per_dt * (time_s / dt_s)

        total_time_s += time_s
        total_fuel_kg += fuel_kg
        total_nm += dist_nm
        segs.append({
            "i": i, "lat": lat2, "lon": lon2,
            "v_kn": v_ach, "seg_nm": dist_nm, "time_s": time_s, "fuel_kg": fuel_kg
        })
    return {"segments": segs, "total_time_s": total_time_s, "total_fuel_kg": total_fuel_kg, "total_nm": total_nm}

