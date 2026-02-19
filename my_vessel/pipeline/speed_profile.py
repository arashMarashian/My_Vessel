from __future__ import annotations
from typing import List, Dict, Tuple, Any
import math

from my_vessel.energy.hybrid_system import HybridPowerSystem
from my_vessel.energy.power_model import propulsion_power, solve_speed_from_power
from my_vessel.energy.vessel_energy_system import aux_power


KNOT_TO_MPS = 0.514444


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    to_rad = lambda d: d * math.pi / 180.0
    dlat = to_rad(lat2 - lat1)
    dlon = to_rad(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(to_rad(lat1)) * math.cos(to_rad(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _battery_discharge_limit_kw(battery: Any) -> float:
    limit = getattr(battery, "p_discharge_max_kw", None)
    if limit is None and hasattr(battery, "to_dict"):
        limit = battery.to_dict().get("p_discharge_max_kw")
    if limit is None:
        return float("inf")
    try:
        return float(limit)
    except (TypeError, ValueError):
        return float("inf")


def feasible_speed_profile(
    ves,
    path_ll: List[Tuple[float, float]],
    target_speed_knots: float,
    dt_s: int = 60,
    env_const: Dict[str, float] | List[Dict[str, float]] | None = None,
    hotel_power_kw: float = 0.0,
    aux_power_kw: float = 0.0,
    hybrid_system: HybridPowerSystem | None = None,
    dispatch_policy: str = "load_smoothing",
) -> Dict[str, Any]:
    """
    Step a VesselEnergySystem along the path, logging detailed time series.
    Assumes ves.step(environment, target_speed, timestep_seconds, ...) returns dict with at least:
      - 'achieved_speed_knots'
      - 'fuel_consumed_kg'   (for the dt)
      - Optionally: 'engines' -> list of per-engine dicts with 'power_kw' and 'sfoc_g_per_kwh'
      - Optionally: 'total_propulsion_power_kw', 'hotel_kw', 'aux_kw'
    When ``hybrid_system`` is provided, ``dispatch_policy`` is forwarded to
    ``HybridPowerSystem.simulate`` for each time step so rows also include generator
    and battery dispatch fields such as ``battery_soc`` (0-1 fraction) and ``p_gen_kw``.
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
        env_for_prop = {
            "wind_speed": float(env.get("wind_speed", 0.0)),
            "wind_angle_diff": float(env.get("wind_angle_diff", 0.0)),
            "wave_height": float(env.get("wave_height", 0.0)),
        }
        v_target_mps = float(target_speed_knots) * KNOT_TO_MPS
        # You can enrich env here with gridded fields if available in the future
        step = ves.step(environment=env, target_speed=target_speed_knots, timestep_seconds=dt_s)

        v_ach_kn = max(1e-6, float(step.get("achieved_speed_knots", target_speed_knots)))
        seg_time_s = (dist_nm / v_ach_kn) * 3600.0

        fuel_dt = float(step.get("fuel_consumed_kg", 0.0))
        # scale fuel if segment time != dt
        fuel_seg = fuel_dt * (seg_time_s / dt_s)

        engines = step.get("engines", [])
        per_engine_kw = [float(e.get("power_kw", 0.0)) for e in engines]
        per_engine_sfoc = [float(e.get("sfoc_g_per_kwh", 0.0)) for e in engines]
        prop_kw_model = propulsion_power(env_for_prop, v_target_mps) / 1000.0
        total_prop_kw = float(step.get("total_propulsion_power_kw", sum(per_engine_kw)))
        if total_prop_kw <= 0.0:
            total_prop_kw = prop_kw_model
        hotel_kw = float(step.get("hotel_kw", hotel_power_kw))
        aux_kw = float(step.get("aux_kw", aux_power_kw))
        hotel_aux_kw = hotel_kw + aux_kw
        total_power_kw = float(step.get("total_power_kw", total_prop_kw + hotel_aux_kw))
        battery_power_kw = float(step.get("battery_power_kw", 0.0))
        battery_power_cmd_kw = float(step.get("battery_power_cmd_kw", battery_power_kw))
        battery_soc_kwh = float(step.get("battery_soc_kwh", 0.0))
        battery_soc = float(step.get("battery_soc", 0.0)) if "battery_soc" in step else None
        p_gen_kw = float(step.get("p_gen_kw", 0.0))
        unserved_kw = float(step.get("unserved_kw", 0.0)) if "unserved_kw" in step else 0.0

        if hybrid_system is not None:
            generator = hybrid_system.generator
            battery = hybrid_system.battery
            batt_limit_kw = _battery_discharge_limit_kw(battery)
            supply_max_kw = float(generator.p_max_kw)
            if math.isfinite(batt_limit_kw):
                supply_max_kw += max(0.0, batt_limit_kw)
            else:
                supply_max_kw = float("inf")

            P_prop_target_w = propulsion_power(env_for_prop, v_target_mps)
            total_prop_kw = P_prop_target_w / 1000.0
            aux_kw = aux_power(env_for_prop, P_prop_target_w) / 1000.0
            hotel_aux_kw = hotel_kw + aux_kw
            P_total_target_kw = total_prop_kw + hotel_aux_kw
            v_ach_mps = v_target_mps

            if P_total_target_kw > supply_max_kw:
                prop_cap_kw = max(0.0, supply_max_kw - hotel_aux_kw)
                P_prop_avail_w = prop_cap_kw * 1000.0
                v_ach_mps = solve_speed_from_power(env_for_prop, P_prop_avail_w, v_target_mps)
                P_prop_w = propulsion_power(env_for_prop, v_ach_mps)
                total_prop_kw = P_prop_w / 1000.0
                aux_kw = aux_power(env_for_prop, P_prop_w) / 1000.0
                hotel_aux_kw = hotel_kw + aux_kw
                total_power_kw = total_prop_kw + hotel_aux_kw
            else:
                total_power_kw = P_total_target_kw

            dispatch_load_kw = total_power_kw
            dispatch = hybrid_system.simulate(
                [dispatch_load_kw],
                dt_s=dt_s,
                policy=dispatch_policy,
                reset_state=False,
            )
            p_batt_series = dispatch.get("p_batt_series", [])
            p_batt_cmd_series = dispatch.get("p_batt_cmd_series", [])
            soc_series = dispatch.get("soc_series", [])
            p_gen_series = dispatch.get("p_gen_series", [])
            unserved_series = dispatch.get("unserved_kw_series", [])
            mode_series = dispatch.get("mode_series", [])
            battery_power_kw = float(p_batt_series[-1]) if p_batt_series else 0.0
            battery_power_cmd_kw = float(p_batt_cmd_series[-1]) if p_batt_cmd_series else battery_power_cmd_kw
            battery_soc = float(soc_series[-1]) if soc_series else battery_soc
            p_gen_kw = float(p_gen_series[-1]) if p_gen_series else max(0.0, dispatch_load_kw - battery_power_kw)
            unserved_kw = float(unserved_series[-1]) if unserved_series else 0.0
            gen_mode = str(mode_series[-1]) if mode_series else None
            capacity_kwh = getattr(hybrid_system.battery, "capacity_kwh", None)
            if capacity_kwh is not None and battery_soc is not None:
                battery_soc_kwh = battery_soc * float(capacity_kwh)
            limited_speed_kn = max(1e-6, v_ach_mps / KNOT_TO_MPS)
            v_ach_kn = min(v_ach_kn, limited_speed_kn)
            seg_time_s = (dist_nm / v_ach_kn) * 3600.0
        else:
            gen_mode = None

        p_batt_actual_kw = battery_power_kw
        p_batt_cmd_kw = battery_power_cmd_kw

        t_s += seg_time_s
        totals["time_s"] += seg_time_s
        totals["fuel_kg"] += fuel_seg
        totals["nm"] += dist_nm

        rows.append({
            "i": i,
            "lat": lat2,
            "lon": lon2,
            "v_kn": v_ach_kn,
            "seg_nm": dist_nm,
            "t_s": seg_time_s,
            "t_total_s": t_s,
            "fuel_kg": fuel_seg,
            "fuel_total_kg": totals["fuel_kg"],
            "total_prop_kw": total_prop_kw,
            "hotel_kw": hotel_kw,
            "aux_kw": aux_kw,
            "total_power_kw": total_power_kw,
            "battery_power_kw": battery_power_kw,
            "p_batt_actual_kw": p_batt_actual_kw,
            "p_batt_cmd_kw": p_batt_cmd_kw,
            "battery_soc_kwh": battery_soc_kwh,
            "battery_soc": battery_soc,
            "p_gen_kw": p_gen_kw,
            "unserved_kw": unserved_kw,
            "gen_mode": gen_mode,
            "per_engine_kw": per_engine_kw,
            "per_engine_sfoc_g_per_kwh": per_engine_sfoc,
            "env_wind_speed": float(env.get("wind_speed", 0.0)),
            "env_wind_angle_diff": float(env.get("wind_angle_diff", 0.0)),
            "env_wave_height": float(env.get("wave_height", 0.0)),
        })
    return {"segments": rows, "totals": totals}
