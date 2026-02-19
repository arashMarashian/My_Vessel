import math

from my_vessel.energy import Battery, DieselGenerator, HybridPowerSystem


def make_generator(**overrides):
    params = dict(p_max_kw=1500.0, p_min_kw=300.0)
    params.update(overrides)
    return DieselGenerator(**params)


def make_battery(**overrides):
    params = dict(
        capacity_kwh=2000.0,
        soc_init=0.6,
        soc_min=0.15,
        soc_max=0.95,
        p_charge_max_kw=700.0,
        p_discharge_max_kw=700.0,
        eta_charge=0.97,
        eta_discharge=0.97,
    )
    params.update(overrides)
    return Battery(**params)


def make_system(**kwargs):
    generator = make_generator()
    battery = make_battery()
    return HybridPowerSystem(generator=generator, battery=battery, **kwargs)


def test_power_balance_and_soc_bounds_naive_policy():
    system = make_system()
    loads = [0.0, 200.0, 700.0, 1200.0, 400.0, 950.0, 0.0, 500.0, 300.0]
    dt_s = 5.0
    result = system.simulate(loads, dt_s, policy="naive")

    for load_kw, p_gen, p_batt in zip(loads, result["p_gen_series"], result["p_batt_series"]):
        assert math.isclose(load_kw, p_gen + p_batt, abs_tol=1e-6)
        if p_gen > 0:
            assert p_gen >= system.generator.p_min_kw - 1e-6

    soc_min = system.battery.soc_min
    soc_max = system.battery.soc_max
    for soc in result["soc_series"]:
        assert soc_min - 1e-9 <= soc <= soc_max + 1e-9


def test_power_balance_and_soc_bounds_smoothing_policy():
    system = make_system(smoothing_band=(700.0, 1100.0))
    loads = [1500.0, 600.0, 1200.0, 300.0, 900.0] * 4
    result = system.simulate(loads, dt_s=8.0, policy="load_smoothing")

    for load_kw, p_gen, p_batt in zip(loads, result["p_gen_series"], result["p_batt_series"]):
        assert math.isclose(load_kw, p_gen + p_batt, abs_tol=1e-6)
        if p_gen > 0:
            assert p_gen >= system.generator.p_min_kw - 1e-6

    soc_min = system.battery.soc_min
    soc_max = system.battery.soc_max
    assert all(soc_min - 1e-9 <= soc <= soc_max + 1e-9 for soc in result["soc_series"])


def test_load_smoothing_uses_less_fuel_than_naive():
    system = make_system(smoothing_band=(700.0, 1000.0))
    loads = [600.0, 1500.0] * 200
    dt_s = 10.0

    naive = system.simulate(loads, dt_s=dt_s, policy="naive")
    smoothing = system.simulate(loads, dt_s=dt_s, policy="load_smoothing")

    assert smoothing["fuel_kg_total"] < naive["fuel_kg_total"]


def test_hybrid_system_with_battery_class_outputs_and_balance():
    generator = make_generator()
    battery = Battery(
        capacity_kwh=100.0,
        soc_init=0.5,
        soc_min=0.1,
        soc_max=0.9,
        p_charge_max_kw=50.0,
        p_discharge_max_kw=50.0,
        eta_charge=0.95,
        eta_discharge=0.95,
    )
    system = HybridPowerSystem(generator=generator, battery=battery)
    loads = [40.0, 35.0, 320.0, 450.0, 25.0, 10.0]

    for policy in ("naive", "load_smoothing"):
        result = system.simulate(loads, dt_s=5.0, policy=policy, reset_state=True)
        for key in ("fuel_kg_total", "soc_series", "p_gen_series", "p_batt_series"):
            assert key in result

        assert len(result["soc_series"]) == len(loads)
        soc_min = system.battery.soc_min
        soc_max = system.battery.soc_max
        for soc in result["soc_series"]:
            assert soc_min - 1e-9 <= soc <= soc_max + 1e-9

        for load_kw, p_gen, p_batt in zip(loads, result["p_gen_series"], result["p_batt_series"]):
            assert math.isclose(load_kw, p_gen + p_batt, abs_tol=1e-6)
