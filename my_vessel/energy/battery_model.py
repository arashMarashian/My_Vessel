"""Battery model with power limits and efficiencies."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class BatteryConfig:
    capacity_kwh: float
    soc_init: float
    soc_min: float = 0.0
    soc_max: float = 1.0
    p_charge_max_kw: float = 1000.0
    p_discharge_max_kw: float = 1000.0
    eta_charge: float = 0.95
    eta_discharge: float = 0.95


class BatteryModel:
    """Simple battery with SoC constraints and asymmetric efficiencies."""

    def __init__(self, **kwargs) -> None:
        cfg = BatteryConfig(**kwargs)
        if not (0 < cfg.capacity_kwh):
            raise ValueError("capacity_kwh must be positive")
        if not (0 < cfg.eta_charge <= 1 and 0 < cfg.eta_discharge <= 1):
            raise ValueError("efficiencies must be within (0, 1]")
        if cfg.soc_min > cfg.soc_max:
            raise ValueError("soc_min must be <= soc_max")
        if cfg.soc_init < cfg.soc_min or cfg.soc_init > cfg.soc_max:
            raise ValueError("soc_init must lie within [soc_min, soc_max]")
        self.cfg = cfg
        self._soc = cfg.soc_init

    @property
    def soc(self) -> float:
        return self._soc

    def reset(self) -> None:
        self._soc = self.cfg.soc_init

    def to_dict(self) -> Dict[str, float]:
        data = asdict(self.cfg)
        data.update({"soc": self._soc})
        return data

    def _available_discharge_kwh(self) -> float:
        return (self._soc - self.cfg.soc_min) * self.cfg.capacity_kwh

    def _available_charge_kwh(self) -> float:
        return (self.cfg.soc_max - self._soc) * self.cfg.capacity_kwh

    def step(self, p_batt_kw: float, dt_s: float) -> Dict[str, float]:
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")

        # Clamp requested power to limits
        req = float(p_batt_kw)
        p_max = self.cfg.p_discharge_max_kw
        p_min = -self.cfg.p_charge_max_kw
        p_cmd = max(p_min, min(p_max, req))

        dt_h = dt_s / 3600.0
        applied = p_cmd
        energy_delta = 0.0

        if applied >= 0:
            # Discharge towards load
            max_by_soc_kw = (self._available_discharge_kwh() * self.cfg.eta_discharge) / max(dt_h, 1e-9)
            if max_by_soc_kw < applied:
                applied = max(0.0, max_by_soc_kw)
            energy_out = (applied * dt_h) / self.cfg.eta_discharge
            self._soc -= energy_out / self.cfg.capacity_kwh
            energy_delta = -energy_out
        else:
            # Charging (negative power -> energy into battery)
            p_abs = -applied
            max_by_soc_kw = (self._available_charge_kwh() / self.cfg.eta_charge) / max(dt_h, 1e-9)
            if max_by_soc_kw < p_abs:
                p_abs = max_by_soc_kw
                applied = -p_abs
            energy_in = p_abs * dt_h * self.cfg.eta_charge
            self._soc += energy_in / self.cfg.capacity_kwh
            energy_delta = energy_in

        # Clamp SOC to allowed range
        self._soc = max(self.cfg.soc_min, min(self.cfg.soc_max, self._soc))

        return {
            "requested_power_kw": req,
            "applied_power_kw": applied,
            "soc": self._soc,
            "energy_delta_kwh": energy_delta,
        }


__all__ = ["BatteryModel", "BatteryConfig"]
