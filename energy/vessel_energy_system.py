"""Vessel power balance simulation utilities."""

from __future__ import annotations

from typing import Dict, List

from energy.power_model import propulsion_power


class Battery:
    """Simple battery storage model."""

    def __init__(
        self,
        capacity_kwh: float,
        soc_kwh: float | None = None,
        charge_eff: float = 0.95,
        discharge_eff: float = 0.95,
    ) -> None:
        self.capacity_kwh = capacity_kwh
        self.soc_kwh = soc_kwh if soc_kwh is not None else capacity_kwh / 2.0
        self.charge_eff = charge_eff
        self.discharge_eff = discharge_eff

    @property
    def soc(self) -> float:
        return self.soc_kwh

    def step(self, power_w: float, dt_h: float) -> float:
        """Apply power for ``dt_h`` hours and update state of charge.

        Parameters
        ----------
        power_w : float
            Positive for charging (power drawn from the bus),
            negative for discharging (power delivered to the bus).
        dt_h : float
            Timestep duration in hours.

        Returns
        -------
        float
            Actual power applied (same sign as input).
        """
        if dt_h <= 0:
            raise ValueError("dt_h must be positive")

        if power_w >= 0:
            # Charging
            energy_added = power_w * dt_h / 1000.0 * self.charge_eff
            if self.soc_kwh + energy_added > self.capacity_kwh:
                energy_added = self.capacity_kwh - self.soc_kwh
                power_w = energy_added * 1000.0 / dt_h / self.charge_eff
            self.soc_kwh += energy_added
        else:
            # Discharging
            energy_needed = -power_w * dt_h / 1000.0 / self.discharge_eff
            if self.soc_kwh - energy_needed < 0:
                energy_needed = self.soc_kwh
                power_w = -energy_needed * 1000.0 * self.discharge_eff / dt_h
            self.soc_kwh -= energy_needed

        return power_w


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

        actual_speed = target_speed
        iter_count = 0
        while supply < demand and actual_speed > 0 and iter_count < 100:
            actual_speed = max(0.0, actual_speed - 0.5)
            P_prop = propulsion_power(environment, actual_speed)
            P_aux = aux_power(environment, P_prop)
            demand = P_prop + P_hotel + P_aux + (actual_batt_power if actual_batt_power > 0 else 0.0)
            iter_count += 1

        if supply < demand:
            raise RuntimeError("Insufficient power supply from engines and battery")

        self.speed = actual_speed

        return {
            "actual_speed": actual_speed,
            "fuel_used_g": fuel_used,
            "battery_soc_kwh": self.battery.soc_kwh,
            "power_demand_w": demand,
            "power_supply_w": supply,
        }


__all__ = [
    "Battery",
    "hotel_power",
    "aux_power",
    "VesselEnergySystem",
]
