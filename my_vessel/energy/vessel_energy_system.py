"""Vessel power balance simulation utilities."""

from __future__ import annotations

from typing import Dict, List

from .power_model import propulsion_power


class Battery:
    """Simple battery storage model with config-friendly parameters."""

    def __init__(
        self,
        capacity_kwh: float,
        soc_init: float | None = None,
        soc_min: float = 0.0,
        soc_max: float = 1.0,
        p_charge_max_kw: float | None = None,
        p_discharge_max_kw: float | None = None,
        eta_charge: float | None = None,
        eta_discharge: float | None = None,
        *,
        soc_kwh: float | None = None,
        soc_initial: float | None = None,
        initial_soc: float | None = None,
        soc0: float | None = None,
        charge_eff: float | None = None,
        discharge_eff: float | None = None,
    ) -> None:
        if capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive")

        self.capacity_kwh = float(capacity_kwh)

        self.eta_charge = self._resolve_alias(
            "eta_charge", eta_charge, {"charge_eff": charge_eff}, default=0.95
        )
        self.eta_discharge = self._resolve_alias(
            "eta_discharge", eta_discharge, {"discharge_eff": discharge_eff}, default=0.95
        )

        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        if not (0.0 <= self.soc_min <= self.soc_max <= 1.0):
            raise ValueError("soc_min and soc_max must satisfy 0 <= soc_min <= soc_max <= 1")

        self._soc = self._resolve_soc(
            soc_init,
            aliases={"soc_initial": soc_initial, "initial_soc": initial_soc, "soc0": soc0},
            soc_kwh=soc_kwh,
        )
        if not (self.soc_min <= self._soc <= self.soc_max):
            raise ValueError("Initial state of charge must lie within [soc_min, soc_max]")

        self.p_charge_max_kw = float(p_charge_max_kw) if p_charge_max_kw is not None else float("inf")
        self.p_discharge_max_kw = (
            float(p_discharge_max_kw) if p_discharge_max_kw is not None else float("inf")
        )

    @staticmethod
    def _resolve_alias(
        name: str,
        primary: float | None,
        aliases: Dict[str, float | None],
        default: float,
    ) -> float:
        provided = [(name, primary)] + list(aliases.items())
        chosen = [(n, v) for n, v in provided if v is not None]
        if len(chosen) > 1:
            raise ValueError(
                f"Conflicting values for {name}: provided via {', '.join(n for n, _ in chosen)}"
            )
        if chosen:
            return float(chosen[0][1])
        return default

    def _resolve_soc(
        self,
        soc_init: float | None,
        aliases: Dict[str, float | None],
        soc_kwh: float | None,
    ) -> float:
        provided = [( "soc_init", soc_init)] + list(aliases.items())
        chosen = [(n, v) for n, v in provided if v is not None]
        if chosen and soc_kwh is not None:
            raise ValueError("Provide either soc_init (or its aliases) or soc_kwh, not both")
        if len(chosen) > 1:
            raise ValueError(
                f"Conflicting values for soc_init provided via {', '.join(n for n, _ in chosen)}"
            )
        if chosen:
            return float(chosen[0][1])
        if soc_kwh is not None:
            return float(soc_kwh) / self.capacity_kwh
        # Default to mid SOC
        return 0.5

    @property
    def soc(self) -> float:
        """State of charge as a fraction of capacity."""
        return self._soc

    @property
    def soc_kwh(self) -> float:
        """State of charge expressed in kWh."""
        return self._soc * self.capacity_kwh

    def _clamp_power_kw(self, power_kw: float) -> float:
        if power_kw >= 0:
            return min(power_kw, self.p_charge_max_kw)
        return max(power_kw, -self.p_discharge_max_kw)

    def step(self, power_w: float, dt_h: float) -> float:
        """Apply power for ``dt_h`` hours and update state of charge.

        Positive power draws from the bus (charging). Returns the actual applied power in watts.
        """
        if dt_h <= 0:
            raise ValueError("dt_h must be positive")

        power_kw = self._clamp_power_kw(power_w / 1000.0)
        if power_kw >= 0:
            # charging
            energy_added = power_kw * dt_h * self.eta_charge
            capacity_remaining = (self.soc_max - self._soc) * self.capacity_kwh
            if energy_added > capacity_remaining and self.eta_charge > 0:
                energy_added = capacity_remaining
                power_kw = energy_added / (dt_h * self.eta_charge) if dt_h > 0 else 0.0
            self._soc += energy_added / self.capacity_kwh
        else:
            # discharging
            discharge_kw = -power_kw
            energy_needed = discharge_kw * dt_h / self.eta_discharge
            available = (self._soc - self.soc_min) * self.capacity_kwh
            if energy_needed > available and self.eta_discharge > 0:
                energy_needed = available
                discharge_kw = energy_needed * self.eta_discharge / dt_h if dt_h > 0 else 0.0
                power_kw = -discharge_kw
            self._soc -= energy_needed / self.capacity_kwh

        self._soc = max(self.soc_min, min(self.soc_max, self._soc))
        return power_kw * 1000.0


def hotel_power(env: Dict[str, float]) -> float:
    """Return hotel power demand in watts.

    Currently this is a simple constant based on environment.
    """
    base = 500_000.0  # 500 kW constant load
    temp = float(env.get("ambient_temp", 20.0))
    return base * (1.0 + 0.01 * max(0.0, 25.0 - temp))


def aux_power(env: Dict[str, float], propulsion_power_w: float) -> float:
    """Return auxiliary power demand in watts."""
    return 0.1 * propulsion_power_w


class VesselEnergySystem:
    """Simulate vessel power flows for one time step."""

    def __init__(self, engines: List, battery: Battery) -> None:
        self.engines = engines
        self.battery = battery
        self.speed = 0.0

    def step(
        self,
        controller_action: Dict[str, float],
        environment: Dict[str, float],
        timestep_hours: float,
    ) -> Dict[str, float | List[float]]:
        """Simulate one timestep of the vessel energy system."""
        loads = controller_action.get("engine_loads", [])
        if len(loads) != len(self.engines):
            raise ValueError("Number of engine loads must match number of engines")
        battery_req = float(controller_action.get("battery_power", 0.0))
        target_speed = float(controller_action.get("target_speed", 0.0))

        engine_powers = []
        fuel_used = []
        for eng, load in zip(self.engines, loads):
            if load == 0:
                engine_powers.append(0.0)
                fuel_used.append(0.0)
                continue

            if load < eng.min_load or load > eng.max_load:
                raise ValueError(
                    f"Engine load {load}% for {eng.name} outside operational range"
                )

            fuel = eng.step(load) * timestep_hours  # g
            power = load / 100.0 * eng.max_power * 1000.0  # W
            engine_powers.append(power)
            fuel_used.append(fuel)

        total_engine_power = sum(engine_powers)
        actual_batt_power = self.battery.step(battery_req, timestep_hours)

        supply = total_engine_power + (-actual_batt_power if actual_batt_power < 0 else 0.0)

        P_prop = propulsion_power(environment, target_speed)
        P_hotel = hotel_power(environment)
        P_aux = aux_power(environment, P_prop)
        demand = P_prop + P_hotel + P_aux + (actual_batt_power if actual_batt_power > 0 else 0.0)

        # If supply is insufficient at target speed, find the maximum achievable speed
        actual_speed = target_speed
        if supply < demand:
            lo, hi = 0.0, target_speed
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                P_prop_m = propulsion_power(environment, mid)
                P_aux_m = aux_power(environment, P_prop_m)
                demand_m = P_prop_m + P_hotel + P_aux_m + (actual_batt_power if actual_batt_power > 0 else 0.0)
                if demand_m <= supply:
                    lo = mid
                else:
                    hi = mid
            actual_speed = lo
            # Final check
            P_prop = propulsion_power(environment, actual_speed)
            P_aux = aux_power(environment, P_prop)
            demand = P_prop + P_hotel + P_aux + (actual_batt_power if actual_batt_power > 0 else 0.0)
            if supply + 1e-6 < demand:
                # Even at zero speed we cannot satisfy demand (should not happen with nonnegative hotel)
                actual_speed = 0.0
                P_prop = propulsion_power(environment, actual_speed)
                P_aux = aux_power(environment, P_prop)
                demand = P_prop + P_hotel + P_aux + (actual_batt_power if actual_batt_power > 0 else 0.0)

        self.speed = actual_speed

        return {
            "actual_speed": actual_speed,
            "fuel_used_g": fuel_used,
            "battery_soc_kwh": self.battery.soc_kwh,
            "power_demand_w": demand,
            "power_supply_w": supply,
            "battery_power_w": actual_batt_power,
        }


__all__ = [
    "Battery",
    "hotel_power",
    "aux_power",
    "VesselEnergySystem",
]
