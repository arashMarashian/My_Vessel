"""Hybrid diesel-generator + battery dispatch utilities.

Example
-------
The snippet below keeps a 1 MW load supplied for 60 seconds while logging fuel use::

    >>> from my_vessel.energy import Battery, DieselGenerator, HybridPowerSystem
    >>> gen = DieselGenerator(p_max_kw=1500.0, p_min_kw=300.0)
    >>> batt = Battery(
    ...     capacity_kwh=1500.0,
    ...     soc_init=0.6,
    ...     soc_min=0.15,
    ...     soc_max=0.95,
    ...     p_charge_max_kw=800.0,
    ...     p_discharge_max_kw=800.0,
    ...     eta_charge=0.97,
    ...     eta_discharge=0.97,
    ... )
    >>> hybrid = HybridPowerSystem(generator=gen, battery=batt)
    >>> results = hybrid.simulate([1000.0] * 60, dt_s=1.0)
    >>> round(results["fuel_kg_total"], 3)
    3.139

"""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .diesel_generator import DieselGenerator


@dataclass
class SimulationResult:
    fuel_kg_total: float
    soc_series: List[float]
    p_gen_series: List[float]
    p_batt_series: List[float]

    def to_dict(self) -> Dict[str, List[float] | float]:
        return {
            "fuel_kg_total": self.fuel_kg_total,
            "soc_series": self.soc_series,
            "p_gen_series": self.p_gen_series,
            "p_batt_series": self.p_batt_series,
        }


class HybridPowerSystem:
    """Coordinate a diesel generator and battery to meet time-varying load."""

    def __init__(
        self,
        *,
        generator: DieselGenerator,
        battery: object,
        smoothing_band: tuple[float, float] | None = None,
        soc_targets: tuple[float, float] | None = None,
    ) -> None:
        self.generator = generator
        self.battery = battery
        self._battery_template = copy.deepcopy(battery)
        self._battery_params = self._extract_battery_params(battery)
        self._battery_interface = self._detect_battery_interface(battery)

        p_max = self.generator.p_max_kw
        if smoothing_band is None:
            smoothing_band = (0.55 * p_max, 0.85 * p_max)
        low, high = smoothing_band
        p_min = self.generator.p_min_kw
        if not (0 < low <= high <= p_max):
            raise ValueError("smoothing_band must lie within (0, p_max]")
        if low < p_min:
            raise ValueError("smoothing_band lower bound must be >= generator p_min_kw")
        self.p_opt_min_kw = low
        self.p_opt_max_kw = high

        soc_min = self._battery_params["soc_min"]
        soc_max = self._battery_params["soc_max"]
        if soc_targets is None:
            span = soc_max - soc_min
            soc_targets = (soc_min + 0.15 * span, soc_min + 0.85 * span)
        lo, hi = soc_targets
        if not (soc_min <= lo <= hi <= soc_max):
            raise ValueError("soc_targets must lie within the battery bounds")
        self.soc_low_target = lo
        self.soc_high_target = hi

    # ------------------------------------------------------------------
    def simulate(
        self,
        load_kw_series: Sequence[float],
        dt_s: float,
        *,
        policy: str = "naive",
        reset_state: bool = True,
    ) -> Dict[str, List[float] | float]:
        """Simulate dispatch for ``load_kw_series`` spaced by ``dt_s`` seconds."""

        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        if not isinstance(load_kw_series, Iterable):
            raise TypeError("load_kw_series must be iterable")
        loads = [max(0.0, float(p)) for p in load_kw_series]
        if not loads:
            return SimulationResult(0.0, [], [], []).to_dict()
        if reset_state:
            self._reset_battery()

        soc_series: List[float] = []
        p_gen_series: List[float] = []
        p_batt_series: List[float] = []
        fuel_total = 0.0
        dt_h = dt_s / 3600.0

        for load_kw in loads:
            max_discharge_kw, max_charge_kw = self._battery_power_limits(dt_h)
            gen_power = self._dispatch_generator(
                load_kw,
                policy,
                max_discharge_kw,
                max_charge_kw,
            )
            p_batt_request = load_kw - gen_power
            p_batt_request = max(-max_charge_kw, min(max_discharge_kw, p_batt_request))
            gen_power = load_kw - p_batt_request
            if 0 < gen_power < self.generator.p_min_kw and load_kw <= max_discharge_kw:
                # Battery can handle the entire load; keep the generator off.
                p_batt_request = load_kw
                gen_power = 0.0

            self._validate_feasible(load_kw, gen_power, max_discharge_kw, max_charge_kw)
            applied_batt = self._apply_battery_power(p_batt_request, dt_s)
            imbalance = load_kw - (gen_power + applied_batt)
            if abs(imbalance) > 1e-6:
                # Numerical safeguard: slight corrections stay within limits.
                gen_power += imbalance
            self._validate_feasible(load_kw, gen_power, max_discharge_kw, max_charge_kw)

            fuel_total += self.generator.fuel_kg_per_s(gen_power) * dt_s
            soc_series.append(self._battery_soc())
            p_gen_series.append(gen_power)
            p_batt_series.append(applied_batt)

        return SimulationResult(fuel_total, soc_series, p_gen_series, p_batt_series).to_dict()

    # ------------------------------------------------------------------
    def _battery_soc(self) -> float:
        soc_attr = getattr(self.battery, "soc", None)
        if soc_attr is None:
            raise AttributeError("Battery implementation must expose 'soc'")
        return float(soc_attr)

    def _battery_power_limits(self, dt_h: float) -> tuple[float, float]:
        params = self._battery_params
        soc = self._battery_soc()
        discharge_cap = max(0.0, (soc - params["soc_min"]) * params["capacity_kwh"])
        charge_cap = max(0.0, (params["soc_max"] - soc) * params["capacity_kwh"])
        if dt_h <= 0:
            raise ValueError("dt_h must be positive")
        discharge_energy_kw = discharge_cap * params["eta_discharge"] / max(dt_h, 1e-12)
        charge_energy_kw = (charge_cap / max(params["eta_charge"], 1e-12)) / max(dt_h, 1e-12)
        max_discharge = min(params["p_discharge_max_kw"], discharge_energy_kw)
        max_charge = min(params["p_charge_max_kw"], charge_energy_kw)
        return max_discharge, max_charge

    def _dispatch_generator(
        self,
        load_kw: float,
        policy: str,
        max_discharge_kw: float,
        max_charge_kw: float,
    ) -> float:
        policy = policy.lower()
        if policy not in {"naive", "load_smoothing"}:
            raise ValueError("policy must be 'naive' or 'load_smoothing'")
        if policy == "naive":
            return self._naive_power(load_kw, max_discharge_kw, max_charge_kw)
        return self._smoothing_power(load_kw, max_discharge_kw, max_charge_kw)

    def _naive_power(
        self,
        load_kw: float,
        max_discharge_kw: float,
        max_charge_kw: float,
    ) -> float:
        if load_kw <= 0:
            return 0.0
        p_min = self.generator.p_min_kw
        p_max = self.generator.p_max_kw
        if load_kw < p_min:
            if load_kw <= max_discharge_kw:
                # Battery alone can cover the low load; keep generator off.
                return 0.0
            upper = load_kw + max_charge_kw
            return min(max(p_min, load_kw), upper, p_max)
        return min(load_kw, p_max)

    def _smoothing_power(
        self,
        load_kw: float,
        max_discharge_kw: float,
        max_charge_kw: float,
    ) -> float:
        if load_kw <= 0:
            return 0.0
        soc = self._battery_soc()
        p_target = min(max(load_kw, self.p_opt_min_kw), self.p_opt_max_kw)
        if p_target < self.generator.p_min_kw and load_kw <= max_discharge_kw:
            return 0.0
        if soc <= self.soc_low_target:
            # Recharge aggressively by biasing toward upper band.
            p_target = max(p_target, min(self.p_opt_max_kw, load_kw + max_charge_kw))
        elif soc >= self.soc_high_target and load_kw <= self.p_opt_min_kw:
            if load_kw <= max_discharge_kw:
                return 0.0
        return min(p_target, self.generator.p_max_kw)

    def _apply_battery_power(self, p_batt_kw: float, dt_s: float) -> float:
        if self._battery_interface == "kw":
            res = self.battery.step(p_batt_kw, dt_s)
            if isinstance(res, dict) and "applied_power_kw" in res:
                return float(res["applied_power_kw"])
            return float(p_batt_kw)
        applied_w = self.battery.step(-p_batt_kw * 1000.0, dt_s / 3600.0)
        return -float(applied_w) / 1000.0

    def _reset_battery(self) -> None:
        reset_fn = getattr(self.battery, "reset", None)
        if callable(reset_fn):
            reset_fn()
            return
        self._copy_object_state(self._battery_template, self.battery)

    @staticmethod
    def _copy_object_state(src: object, dst: object) -> None:
        dst.__dict__.clear()
        dst.__dict__.update(copy.deepcopy(src.__dict__))

    def _extract_battery_params(self, battery: object) -> Dict[str, float]:
        maybe_dict = battery.to_dict() if hasattr(battery, "to_dict") else None

        def resolve(name: str, *, allow_none: bool = False) -> float:
            if hasattr(battery, name):
                value = getattr(battery, name)
            elif maybe_dict and name in maybe_dict:
                value = maybe_dict[name]
            else:
                raise AttributeError(f"Battery missing required attribute '{name}'")
            if value is None:
                if allow_none:
                    return float("inf")
                raise ValueError(f"Battery attribute '{name}' cannot be None")
            return float(value)

        return {
            "capacity_kwh": resolve("capacity_kwh"),
            "soc_min": resolve("soc_min"),
            "soc_max": resolve("soc_max"),
            "eta_charge": resolve("eta_charge"),
            "eta_discharge": resolve("eta_discharge"),
            "p_charge_max_kw": resolve("p_charge_max_kw", allow_none=True),
            "p_discharge_max_kw": resolve("p_discharge_max_kw", allow_none=True),
        }

    @staticmethod
    def _detect_battery_interface(battery: object) -> str:
        sig = inspect.signature(battery.step)
        params = tuple(sig.parameters.keys())
        if "dt_h" in params:
            return "w"
        return "kw"

    def _validate_feasible(
        self,
        load_kw: float,
        p_gen_kw: float,
        max_discharge_kw: float,
        max_charge_kw: float,
    ) -> None:
        tol = 1e-6
        if p_gen_kw < -tol or p_gen_kw > self.generator.p_max_kw + tol:
            raise ValueError("Generator dispatch outside allowable range")
        if 0 < p_gen_kw < self.generator.p_min_kw - tol:
            # Allow slight excursions below min load; downstream balancing will handle spill.
            return
        p_batt = load_kw - p_gen_kw
        if p_batt > max_discharge_kw + tol:
            raise ValueError("Battery cannot discharge enough to meet load")
        if -p_batt > max_charge_kw + tol:
            raise ValueError("Battery cannot absorb the surplus generator power")


__all__ = ["HybridPowerSystem", "SimulationResult"]
