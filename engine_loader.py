"""Utility to load engine specifications from YAML into ``Engine`` objects."""

from typing import Dict, Callable, List
import os

import yaml
import numpy as np


class Engine:
    """Simple engine model for fuel consumption estimation."""

    def __init__(
        self,
        name: str,
        max_power: float,
        min_load: float,
        max_load: float,
        sfoc_curve: Dict[str, Callable[[float], float]],
        startup_cost: float,
    ) -> None:
        self.name = name
        self.max_power = max_power
        self.min_load = min_load
        self.max_load = max_load
        self.sfoc_curve = sfoc_curve
        self.startup_cost = startup_cost

        self.active = False
        self.last_active = False

    def get_fuel_consumption(self, load_percent: float, fuel_type: str = "HFO") -> float:
        """Return specific fuel oil consumption in g/kWh for a given load."""
        if fuel_type not in self.sfoc_curve:
            raise ValueError("Fuel type not supported")
        return float(self.sfoc_curve[fuel_type](load_percent))

    def step(self, load_percent: float, fuel_type: str = "HFO") -> float:
        """Simulate engine at the given load and return fuel used in grams per hour."""
        if load_percent < self.min_load or load_percent > self.max_load:
            raise ValueError("Load out of operational range")

        self.active = load_percent > 0

        sfoc = self.get_fuel_consumption(load_percent, fuel_type)
        energy_output = load_percent / 100 * self.max_power
        fuel_used = sfoc * energy_output

        penalty = self.startup_cost if self.active and not self.last_active else 0
        self.last_active = self.active
        return fuel_used + penalty


def _interp_curve(points: Dict[str, float]) -> Callable[[float], float]:
    # Simple 1D linear interpolation using NumPy. Assumes monotonic loads.
    xs, ys = zip(*sorted(((float(k), float(v)) for k, v in points.items()), key=lambda p: p[0]))
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    def f(load_percent: float) -> float:
        lp = float(load_percent)
        # Clamp to [x.min(), x.max()] — curves cover 0..100 so extrapolation not required
        lp_clamped = min(max(lp, x[0]), x[-1])
        return float(np.interp(lp_clamped, x, y))

    return f


def load_engines_from_yaml(path: str = "data/engine_data.yaml") -> List[Engine]:
    """Load engine configurations from a YAML file and return initialized objects."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Engine data file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    engines = []
    for spec in data.get("engines", []):
        sfoc = {fuel: _interp_curve(curve) for fuel, curve in spec.get("sfoc", {}).items()}
        engine = Engine(
            name=spec["name"],
            max_power=spec["max_power_kw"],
            min_load=spec["min_load_percent"],
            max_load=spec["max_load_percent"],
            sfoc_curve=sfoc,
            startup_cost=spec.get("startup_cost_mj", 0),
        )
        engines.append(engine)
    return engines


__all__ = ["Engine", "load_engines_from_yaml"]
