from __future__ import annotations

from typing import Dict, List, Tuple

import math

from energy.vessel_energy_system import VesselEnergySystem
from energy import aux_power, hotel_power, propulsion_power


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    to_rad = lambda d: d * math.pi / 180.0
    dlat = to_rad(lat2 - lat1)
    dlon = to_rad(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(to_rad(lat1)) * math.cos(to_rad(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def feasible_speed_profile(
    ves: VesselEnergySystem,
    path_ll: List[Tuple[float, float]],
    target_speed_knots: float,
    dt_s: int = 60,
    env_const: Dict | None = None,
):
    if env_const is None:
        env_const = {}
    out = []
    total_time_s = 0.0
    total_fuel_kg = 0.0
    for i in range(1, len(path_ll)):
        lat1, lon1 = path_ll[i - 1]
        lat2, lon2 = path_ll[i]
        dist_nm = haversine_km(lat1, lon1, lat2, lon2) * 0.539957

        # Estimate engine loads needed to meet power demand at target speed
        P_prop = propulsion_power(env_const, target_speed_knots)
        P_hotel = hotel_power(env_const)
        P_aux = aux_power(env_const, P_prop)
        demand_w = P_prop + P_hotel + P_aux
        loads = []
        per_engine = demand_w / max(1, len(ves.engines))
        for eng in ves.engines:
            load_pct = per_engine / (eng.max_power * 1000.0) * 100.0
            load_pct = max(eng.min_load, min(eng.max_load, load_pct))
            loads.append(load_pct)

        step = ves.step(
            controller_action={
                "engine_loads": loads,
                "battery_power": 0.0,
                "target_speed": target_speed_knots,
            },
            environment=env_const,
            timestep_hours=dt_s / 3600.0,
        )
        v_ach = float(step.get("actual_speed", 0.0))
        time_s = (dist_nm / max(1e-6, v_ach)) * 3600.0
        fuel_g = float(sum(step.get("fuel_used_g", [])))
        fuel_kg = fuel_g / 1000.0 * (time_s / dt_s)
        total_time_s += time_s
        total_fuel_kg += fuel_kg
        out.append(
            {
                "i": i,
                "lat": lat2,
                "lon": lon2,
                "v_kn": v_ach,
                "seg_nm": dist_nm,
                "time_s": time_s,
                "fuel_kg": fuel_kg,
            }
        )
    return {"segments": out, "total_time_s": total_time_s, "total_fuel_kg": total_fuel_kg}
