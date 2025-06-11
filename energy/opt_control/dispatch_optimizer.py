from __future__ import annotations

import numpy as np
import pyomo.environ as pyo

from typing import Dict, List

from engine_loader import Engine

from energy.power_model import propulsion_power
from energy.vessel_energy_system import hotel_power, aux_power


class DispatchOptimizer:
    """Build and solve an engine dispatch problem using Pyomo."""

    def __init__(
        self,
        engines: List[Engine],
        horizon: int,
        env: List[Dict[str, float]],
        dt_hours: float,
        battery_capacity_kwh: float,
        sfoc_curves: List[Dict[float, float]] | None = None,
        soc0: float = 0.5,
        eta_c: float = 0.95,
        eta_d: float = 0.95,
        v_max: float = 10.0,
        target_distance_m: float | None = None,
        lambda_bat: float = 10.0,
        lambda_switch: float = 1.0,
    ) -> None:
        if len(env) != horizon:
            raise ValueError("Environment list length must equal horizon")

        self.engines = engines
        self.horizon = horizon
        self.env = env
        self.dt = dt_hours
        self.capacity = battery_capacity_kwh
        self.soc0 = soc0
        self.eta_c = eta_c
        self.eta_d = eta_d
        self.v_max = v_max
        self.target_distance = target_distance_m
        self.lambda_bat = lambda_bat
        self.lambda_switch = lambda_switch

        # fit cubic polynomials for SFOC curves
        if sfoc_curves is None:
            raise ValueError("sfoc_curves must be provided")

        if len(sfoc_curves) != len(engines):
            raise ValueError("sfoc_curves length must match engines")

        self.sfoc_coefs = []
        for curve in sfoc_curves:
            loads = [float(l) for l in curve.keys()]
            vals = [float(v) for v in curve.values()]
            coef = np.polyfit(loads, vals, 3)
            self.sfoc_coefs.append(coef)

        self.model = self._build_model()

    def _build_model(self) -> pyo.ConcreteModel:
        m = pyo.ConcreteModel()
        I = range(len(self.engines))
        T = range(self.horizon)
        m.I = pyo.Set(initialize=I)
        m.T = pyo.Set(initialize=T)

        m.x = pyo.Var(m.I, m.T, domain=pyo.Binary)
        m.u = pyo.Var(m.I, m.T, bounds=(0, 1))
        m.p_bat_charge = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.p_bat_discharge = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.soc = pyo.Var(range(self.horizon + 1), bounds=(0, 1))
        m.v = pyo.Var(m.T, bounds=(0, self.v_max))

        if self.target_distance is not None:
            def distance_rule(m):
                return sum(m.v[t] * self.dt * 3600 for t in m.T) >= self.target_distance
            m.distance_target = pyo.Constraint(rule=distance_rule)

        # initial SOC
        m.soc[0].fix(self.soc0)

        # engine load bounds
        def load_bounds_rule(m, i, t):
            eng = self.engines[i]
            return (
                eng.min_load / 100 * m.x[i, t],
                eng.max_load / 100 * m.x[i, t],
            )
        m.load_bounds = pyo.Constraint(m.I, m.T, rule=lambda m,i,t: m.u[i,t] >= load_bounds_rule(m,i,t)[0])
        m.load_bounds_upper = pyo.Constraint(m.I, m.T, rule=lambda m,i,t: m.u[i,t] <= load_bounds_rule(m,i,t)[1])

        # battery SOC update
        def soc_rule(m, t):
            if t == m.T.last():
                return pyo.Constraint.Skip
            charge = self.eta_c * m.p_bat_charge[t] * self.dt / (self.capacity * 1000)
            discharge = m.p_bat_discharge[t] * self.dt / (self.eta_d * self.capacity * 1000)
            return m.soc[t+1] == m.soc[t] + charge - discharge
        m.soc_update = pyo.Constraint(m.T, rule=soc_rule)

        # power balance
        def balance_rule(m, t):
            supply = sum(
                m.u[i, t] * self.engines[i].max_power * 1000 for i in m.I
            ) + m.p_bat_discharge[t]
            v = m.v[t]
            env = self.env[t]
            p_prop = (
                self._prop_power_expr(v, env)
            )
            p_hotel = hotel_power(env)
            p_aux = aux_power(env, p_prop)
            demand = p_prop + p_hotel + p_aux + m.p_bat_charge[t]
            return supply >= demand
        m.balance = pyo.Constraint(m.T, rule=balance_rule)

        # absolute SOC change
        m.soc_change_pos = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.soc_change_neg = pyo.Var(m.T, domain=pyo.NonNegativeReals)

        def soc_change_rule(m, t):
            return (
                m.soc_change_pos[t] - m.soc_change_neg[t] == m.soc[t+1] - m.soc[t]
            )
        m.soc_change = pyo.Constraint(m.T, rule=soc_change_rule)

        # engine switching penalty variables
        m.switch = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)

        def switch_rule1(m, i, t):
            prev = 0 if t == 0 else m.x[i, t-1]
            return m.switch[i, t] >= m.x[i, t] - prev

        def switch_rule2(m, i, t):
            prev = 0 if t == 0 else m.x[i, t-1]
            return m.switch[i, t] >= prev - m.x[i, t]

        m.switch_con1 = pyo.Constraint(m.I, m.T, rule=switch_rule1)
        m.switch_con2 = pyo.Constraint(m.I, m.T, rule=switch_rule2)

        # objective
        def sfoc_expr(i, load):
            coef = self.sfoc_coefs[i]
            l_pct = load * 100
            return (
                coef[0] * l_pct**3 + coef[1] * l_pct**2 + coef[2] * l_pct + coef[3]
            )

        def obj_rule(m):
            cost = 0
            for t in m.T:
                # fuel cost
                for i in m.I:
                    sfoc = sfoc_expr(i, m.u[i, t])
                    cost += (
                        m.u[i, t]
                        * self.engines[i].max_power
                        * sfoc
                    )
                cost += self.lambda_bat * (m.soc_change_pos[t] + m.soc_change_neg[t])
                for i in m.I:
                    cost += self.lambda_switch * m.switch[i, t]
            return cost

        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        return m

    def _prop_power_expr(self, v, env):
        k1 = 250.0
        k2 = 80.0
        A_proj = 1000.0
        C_D = 0.75
        k_wave = 450.0
        rho_air = 1.225
        wind_speed = env.get("wind_speed", 0.0)
        angle = np.deg2rad(env.get("wind_angle_diff", 0.0))
        v_rel = v + wind_speed * pyo.cos(angle)
        R_calm = k1 * v**2 + k2 * v**3
        R_wind = 0.5 * rho_air * A_proj * C_D * v_rel**2
        Hs = env.get("wave_height", 0.0)
        R_wave = k_wave * Hs * v**2
        return (R_calm + R_wind + R_wave) * v / 0.7

    def solve(self, solver: str = "ipopt") -> None:
        pyo.SolverFactory(solver).solve(self.model, tee=False)

    def results(self) -> Dict:
        m = self.model
        data = {
            "speed": [pyo.value(m.v[t]) for t in m.T],
            "soc": [pyo.value(m.soc[t]) for t in range(self.horizon + 1)],
            "loads": [
                [pyo.value(m.u[i, t]) for i in m.I] for t in m.T
            ],
        }
        return data

__all__ = ["DispatchOptimizer"]
