from __future__ import annotations

from typing import Dict, List

from my_vessel.energy.battery_model import BatteryModel
from my_vessel.energy.diesel_generator import DieselGenerator
from my_vessel.energy.hybrid_system import HybridPowerSystem
from my_vessel.pipeline.speed_profile import feasible_speed_profile


class _StubVES:
    def __init__(self, propulsive_loads_kw: List[float]) -> None:
        self.loads = list(propulsive_loads_kw)
        self.idx = 0

    def step(self, environment: Dict[str, float], target_speed: float, timestep_seconds: float):
        load_kw = self.loads[min(self.idx, len(self.loads) - 1)]
        self.idx += 1
        engines = [{"power_kw": load_kw, "sfoc_g_per_kwh": 200.0}]
        hotel_kw = 20.0
        aux_kw = 5.0
        return {
            "achieved_speed_knots": target_speed,
            "fuel_consumed_kg": 0.0,
            "engines": engines,
            "total_propulsion_power_kw": load_kw,
            "hotel_kw": hotel_kw,
            "aux_kw": aux_kw,
            "total_power_kw": load_kw + hotel_kw + aux_kw,
        }


def test_speed_profile_hybrid_reports_battery_power():
    path = [(0.0, 0.0), (0.0, 0.01), (0.0, 0.02), (0.0, 0.04)]
    loads = [150.0, 600.0, 180.0]
    ves = _StubVES(loads)
    env_series = [
        {"wind_speed": 0.0, "wind_angle_diff": 0.0, "wave_height": 0.0}
        for _ in range(len(path))
    ]
    generator = DieselGenerator(p_max_kw=800.0, p_min_kw=300.0)
    battery = BatteryModel(
        capacity_kwh=400.0,
        soc_init=0.7,
        soc_min=0.2,
        soc_max=0.95,
        p_charge_max_kw=250.0,
        p_discharge_max_kw=250.0,
        eta_charge=0.96,
        eta_discharge=0.95,
    )
    hybrid = HybridPowerSystem(generator=generator, battery=battery)

    profile = feasible_speed_profile(
        ves,
        path,
        target_speed_knots=10.0,
        dt_s=60,
        env_const=env_series,
        hybrid_system=hybrid,
        dispatch_policy="load_smoothing",
    )

    segments = profile["segments"]
    assert segments, "speed profile should include segments"
    battery_series = [seg.get("battery_power_kw", 0.0) for seg in segments]
    assert any(abs(p) > 1e-3 for p in battery_series), "battery dispatch should respond to varying load"
    assert any("p_gen_kw" in seg for seg in segments)
    assert any("p_batt_actual_kw" in seg and "p_batt_cmd_kw" in seg for seg in segments)
    assert any(seg.get("gen_mode") for seg in segments)
