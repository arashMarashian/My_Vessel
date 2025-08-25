from __future__ import annotations
from typing import List, Dict, Tuple, Any
import math


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    to_rad = lambda d: d * math.pi / 180.0
    dlat = to_rad(lat2 - lat1)
    dlon = to_rad(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(to_rad(lat1)) * math.cos(to_rad(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def feasible_speed_profile(
    ves,
    path_ll: List[Tuple[float, float]],
    target_speed_knots: float,
    dt_s: int = 60,
    env_const: Dict[str, float] | List[Dict[str, float]] | None = None,
    hotel_power_kw: float = 0.0,
    aux_power_kw: float = 0.0,
) -> Dict[str, Any]:
    """
    Step a VesselEnergySystem along the path, logging detailed time series.
    Assumes ves.step(environment, target_speed, timestep_seconds, ...) returns dict with at least:
      - 'achieved_speed_knots'
      - 'fuel_consumed_kg'   (for the dt)
      - Optionally: 'engines' -> list of per-engine dicts with 'power_kw' and 'sfoc_g_per_kwh'
      - Optionally: 'total_propulsion_power_kw', 'hotel_kw', 'aux_kw'
    """
    if env_const is None:
        env_const = {}
    rows = []
    totals = {"time_s": 0.0, "fuel_kg": 0.0, "nm": 0.0}
    if len(path_ll) < 2:
        return {"segments": rows, "totals": totals}

    t_s = 0.0
    for i in range(1, len(path_ll)):
        lat1, lon1 = path_ll[i - 1]
        lat2, lon2 = path_ll[i]
        dist_nm = haversine_km(lat1, lon1, lat2, lon2) * 0.539957

        if isinstance(env_const, list):
            env = dict(env_const[i - 1]) if i - 1 < len(env_const) else {}
        else:
            env = dict(env_const)
        # You can enrich env here with gridded fields if available in the future
        step = ves.step(environment=env, target_speed=target_speed_knots, timestep_seconds=dt_s)

        v_ach = max(1e-6, float(step.get("achieved_speed_knots", target_speed_knots)))
        # time to traverse this segment at achieved speed
        seg_time_s = (dist_nm / v_ach) * 3600.0

        fuel_dt = float(step.get("fuel_consumed_kg", 0.0))
        # scale fuel if segment time != dt
        fuel_seg = fuel_dt * (seg_time_s / dt_s)

        engines = step.get("engines", [])
        per_engine_kw = [float(e.get("power_kw", 0.0)) for e in engines]
        per_engine_sfoc = [float(e.get("sfoc_g_per_kwh", 0.0)) for e in engines]
        total_prop_kw = float(step.get("total_propulsion_power_kw", sum(per_engine_kw)))
        hotel_kw = float(step.get("hotel_kw", hotel_power_kw))
        aux_kw = float(step.get("aux_kw", aux_power_kw))

        t_s += seg_time_s
        totals["time_s"] += seg_time_s
        totals["fuel_kg"] += fuel_seg
        totals["nm"] += dist_nm

        rows.append({
            "i": i,
            "lat": lat2,
            "lon": lon2,
            "v_kn": v_ach,
            "seg_nm": dist_nm,
            "t_s": seg_time_s,
            "t_total_s": t_s,
            "fuel_kg": fuel_seg,
            "fuel_total_kg": totals["fuel_kg"],
            "total_prop_kw": total_prop_kw,
            "hotel_kw": hotel_kw,
            "aux_kw": aux_kw,
            "per_engine_kw": per_engine_kw,
            "per_engine_sfoc_g_per_kwh": per_engine_sfoc,
            "env_wind_speed": float(env.get("wind_speed", 0.0)),
            "env_wind_angle_diff": float(env.get("wind_angle_diff", 0.0)),
            "env_wave_height": float(env.get("wave_height", 0.0)),
        })
    return {"segments": rows, "totals": totals}

