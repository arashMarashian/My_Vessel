"""Power modeling utilities for vessel propulsion."""

from __future__ import annotations

import math
from typing import Dict, Optional


def propulsion_power(
    env: Dict[str, float],
    vessel_speed: float,
    coeffs: Optional[Dict[str, float]] = None,
    eta_prop: float = 0.7,
) -> float:
    """Calculate propulsion power demand under given conditions.

    Parameters
    ----------
    env : dict
        Environment values with keys ``wind_speed`` (m/s),
        ``wind_angle_diff`` (degrees), and ``wave_height`` (m).
    vessel_speed : float
        Vessel speed in meters per second.
    coeffs : dict, optional
        Optional dictionary overriding resistance coefficients. Keys are
        ``k1``, ``k2``, ``A_proj``, ``C_D``, and ``k_wave``.
    eta_prop : float, optional
        Propeller efficiency (0-1). Defaults to ``0.7``.

    Returns
    -------
    float
        Propulsion power in watts.
    """

    defaults = {
        "k1": 250.0,  # frictional resistance
        "k2": 80.0,  # wave-making drag
        "A_proj": 1000.0,  # frontal area (m^2)
        "C_D": 0.75,  # bluff body drag coefficient
        "k_wave": 450.0,  # wave influence coefficient
    }

    if coeffs:
        defaults.update(coeffs)
    c = defaults

    v = float(vessel_speed)
    k1 = c["k1"]
    k2 = c["k2"]
    A_proj = c["A_proj"]
    C_D = c["C_D"]
    k_wave = c["k_wave"]

    # Calm water resistance
    R_calm = k1 * v ** 2 + k2 * v ** 3

    # Wind resistance
    rho_air = 1.225  # kg/m^3
    wind_speed = float(env.get("wind_speed", 0.0))
    angle_diff = math.radians(float(env.get("wind_angle_diff", 0.0)))
    v_rel = v + wind_speed * math.cos(angle_diff)
    R_wind = 0.5 * rho_air * A_proj * C_D * v_rel ** 2

    # Wave resistance
    Hs = float(env.get("wave_height", 0.0))
    R_wave = k_wave * Hs * v ** 2

    R_total = R_calm + R_wind + R_wave

    if eta_prop <= 0:
        raise ValueError("eta_prop must be positive")

    return R_total * v / eta_prop
