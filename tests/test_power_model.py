import math
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from my_vessel.energy.power_model import propulsion_power, solve_speed_from_power


def test_propulsion_power_example():
    env = {"wind_speed": 10.0, "wind_angle_diff": 45.0, "wave_height": 2.0}
    speed = 8.0
    P = propulsion_power(env, speed)

    k1 = 250.0
    k2 = 80.0
    A_proj = 1000.0
    C_D = 0.75
    k_wave = 450.0
    eta_prop = 0.7

    R_calm = k1 * speed ** 2 + k2 * speed ** 3
    rho_air = 1.225
    v_rel = speed + env["wind_speed"] * math.cos(math.radians(env["wind_angle_diff"]))
    R_wind = 0.5 * rho_air * A_proj * C_D * v_rel ** 2
    R_wave = k_wave * env["wave_height"] * speed ** 2

    expected = (R_calm + R_wind + R_wave) * speed / eta_prop

    assert math.isclose(P, expected, rel_tol=1e-9)


def test_solve_speed_from_power_increases_with_available_power():
    env = {"wind_speed": 5.0, "wind_angle_diff": 0.0, "wave_height": 0.5}
    low_speed = solve_speed_from_power(env, 120_000.0, v_max_mps=15.0)
    high_speed = solve_speed_from_power(env, 600_000.0, v_max_mps=15.0)
    assert high_speed > low_speed >= 0.0
    assert high_speed <= 15.0 + 1e-9


def test_solve_speed_from_power_handles_rough_seas():
    calm_env = {"wind_speed": 5.0, "wind_angle_diff": 0.0, "wave_height": 0.2}
    rough_env = {"wind_speed": 5.0, "wind_angle_diff": 0.0, "wave_height": 3.0}
    power_cap = 400_000.0
    calm_speed = solve_speed_from_power(calm_env, power_cap, v_max_mps=15.0)
    rough_speed = solve_speed_from_power(rough_env, power_cap, v_max_mps=15.0)
    assert rough_speed < calm_speed
