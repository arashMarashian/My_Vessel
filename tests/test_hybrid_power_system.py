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
        for key in (
            "fuel_kg_total",
            "soc_series",
            "p_gen_series",
            "p_batt_series",
            "p_batt_cmd_series",
            "mode_series",
        ):
            assert key in result

        assert len(result["soc_series"]) == len(loads)
        soc_min = system.battery.soc_min
        soc_max = system.battery.soc_max
        for soc in result["soc_series"]:
            assert soc_min - 1e-9 <= soc <= soc_max + 1e-9

        for load_kw, p_gen, p_batt in zip(loads, result["p_gen_series"], result["p_batt_series"]):
            assert math.isclose(load_kw, p_gen + p_batt, abs_tol=1e-6)


def test_hysteresis_policy_respects_minimum_dwell_time():
    generator = DieselGenerator(p_max_kw=400.0, p_min_kw=180.0)
    battery = Battery(
        capacity_kwh=250.0,
        soc_init=0.6,
        soc_min=0.2,
        soc_max=0.95,
        p_charge_max_kw=120.0,
        p_discharge_max_kw=120.0,
        eta_charge=0.97,
        eta_discharge=0.97,
    )
    min_on = 40.0
    min_off = 35.0
    system = HybridPowerSystem(
        generator=generator,
        battery=battery,
        min_on_s=min_on,
        min_off_s=min_off,
    )
    loads = [90.0] * 200
    dt_s = 5.0

    result = system.simulate(loads, dt_s=dt_s, policy="hysteresis")
    gen_series = result["gen_on_series"]
    mode_series = result["mode_series"]
    assert len(gen_series) == len(loads) == len(mode_series)
    prev_state = gen_series[0]
    prev_idx = 0
    for idx in range(1, len(gen_series)):
        if gen_series[idx] != prev_state:
            dwell = (idx - prev_idx) * dt_s
            if prev_state:
                assert dwell >= min_on - 1e-6
            else:
                assert dwell >= min_off - 1e-6
            prev_state = gen_series[idx]
            prev_idx = idx

    assert max(result["unserved_kw_series"]) <= 1e-6


def test_hysteresis_respects_ramp_rate():
    generator = DieselGenerator(p_max_kw=800.0, p_min_kw=200.0)
    battery = Battery(
        capacity_kwh=400.0,
        soc_init=0.7,
        soc_min=0.2,
        soc_max=0.95,
        p_charge_max_kw=200.0,
        p_discharge_max_kw=200.0,
        eta_charge=0.97,
        eta_discharge=0.97,
    )
    ramp = 50.0
    system = HybridPowerSystem(
        generator=generator,
        battery=battery,
        min_on_s=5.0,
        min_off_s=5.0,
        ramp_rate_kw_per_s=ramp,
    )
    loads = [50.0, 500.0, 500.0, 50.0]
    result = system.simulate(loads, dt_s=2.0, policy="hysteresis")
    series = result["p_gen_series"]
    for a, b in zip(series, series[1:]):
        assert abs(b - a) <= ramp * 2.0 + 1e-6


def test_hysteresis_spools_up_before_reaching_min_load():
    generator = DieselGenerator(p_max_kw=600.0, p_min_kw=200.0)
    battery = Battery(
        capacity_kwh=250.0,
        soc_init=0.2,
        soc_min=0.2,
        soc_max=0.9,
        p_charge_max_kw=150.0,
        p_discharge_max_kw=150.0,
        eta_charge=0.96,
        eta_discharge=0.95,
    )
    ramp = 40.0
    system = HybridPowerSystem(
        generator=generator,
        battery=battery,
        min_on_s=5.0,
        min_off_s=5.0,
        ramp_rate_kw_per_s=ramp,
    )
    loads = [350.0] * 12
    result = system.simulate(loads, dt_s=1.0, policy="hysteresis")

    p_gen = result["p_gen_series"]
    gen_on = result["gen_on_series"]
    assert any(gen_on), "generator should turn on under low SOC"
    first_on_idx = next(i for i, flag in enumerate(gen_on) if flag)
    assert p_gen[first_on_idx] < generator.p_min_kw
    assert p_gen[first_on_idx] > 0.0

    latch_idx = next(i for i, p in enumerate(p_gen) if p >= generator.p_min_kw - 1e-6)
    assert latch_idx > first_on_idx, "generator should take multiple steps to reach p_min"
    for j in range(latch_idx, min(len(p_gen), latch_idx + 3)):
        assert p_gen[j] >= generator.p_min_kw - 1e-6


def test_hysteresis_charges_when_soc_low():
    generator = make_generator()
    battery = Battery(
        capacity_kwh=200.0,
        soc_init=0.2,
        soc_min=0.2,
        soc_max=0.9,
        p_charge_max_kw=150.0,
        p_discharge_max_kw=150.0,
        eta_charge=0.96,
        eta_discharge=0.95,
    )
    system = HybridPowerSystem(
        generator=generator,
        battery=battery,
        min_on_s=5.0,
        min_off_s=5.0,
    )
    loads = [40.0] * 12
    result = system.simulate(loads, dt_s=5.0, policy="hysteresis")
    assert any(p < -1e-3 for p in result["p_batt_series"])
    assert min(result["soc_series"]) >= battery.soc_min - 1e-9


def test_battery_dispatch_respects_limits():
    generator = make_generator()
    batt_discharge_only = Battery(
        capacity_kwh=150.0,
        soc_init=0.6,
        soc_min=0.2,
        soc_max=0.95,
        p_charge_max_kw=0.0,
        p_discharge_max_kw=80.0,
    )
    system = HybridPowerSystem(generator=generator, battery=batt_discharge_only)
    loads = [60.0, 70.0]
    result = system.simulate(loads, dt_s=5.0, policy="naive")
    assert any(p > 1e-6 for p in result["p_batt_series"])

    batt_fixed = Battery(
        capacity_kwh=150.0,
        soc_init=0.6,
        soc_min=0.2,
        soc_max=0.95,
        p_charge_max_kw=0.0,
        p_discharge_max_kw=0.0,
    )
    system_fixed = HybridPowerSystem(generator=generator, battery=batt_fixed)
    result_fixed = system_fixed.simulate(loads, dt_s=5.0, policy="naive")
    assert all(abs(p) < 1e-9 for p in result_fixed["p_batt_series"])


def test_soc_monotone_with_sign_convention():
    system = make_system(smoothing_band=(600.0, 1000.0))
    loads = [400.0, 1100.0, 300.0, 1200.0] * 3
    initial_soc = system.battery.soc
    result = system.simulate(loads, dt_s=10.0, policy="load_smoothing", reset_state=True)
    soc_series = result["soc_series"]
    p_batt_series = result["p_batt_series"]
    assert soc_series, "expected SOC samples"
    prev_soc = initial_soc
    for soc, p_batt in zip(soc_series, p_batt_series):
        if p_batt > 1e-6:
            assert soc <= prev_soc + 1e-9
        elif p_batt < -1e-6:
            assert soc >= prev_soc - 1e-9
        prev_soc = soc
