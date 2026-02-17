import math

from my_vessel.energy.battery_model import BatteryModel


def make_battery(**kwargs):
    params = dict(
        capacity_kwh=100.0,
        soc_init=0.5,
        soc_min=0.2,
        soc_max=0.9,
        p_charge_max_kw=50.0,
        p_discharge_max_kw=60.0,
        eta_charge=0.95,
        eta_discharge=0.9,
    )
    params.update(kwargs)
    return BatteryModel(**params)


def test_soc_stays_within_bounds():
    batt = make_battery()
    # discharge heavily
    for _ in range(5):
        batt.step(1000.0, 3600.0)
    assert math.isclose(batt.soc, batt.cfg.soc_min, rel_tol=1e-6)

    # charge heavily
    for _ in range(5):
        batt.step(-1000.0, 3600.0)
    assert math.isclose(batt.soc, batt.cfg.soc_max, rel_tol=1e-6)


def test_charge_increases_soc_discharge_decreases():
    batt = make_battery(soc_init=0.6)
    res_charge = batt.step(-20.0, 3600.0)
    soc_after_charge = res_charge["soc"]
    res_discharge = batt.step(15.0, 3600.0)
    soc_after_discharge = res_discharge["soc"]
    assert soc_after_charge > 0.6
    assert soc_after_discharge < soc_after_charge


def test_power_clamp_applied():
    batt = make_battery(p_charge_max_kw=25.0, p_discharge_max_kw=30.0)
    res = batt.step(100.0, 60.0)
    assert math.isclose(res["applied_power_kw"], 30.0)
    res2 = batt.step(-100.0, 60.0)
    assert math.isclose(res2["applied_power_kw"], -25.0)


def test_energy_tracking_with_efficiency():
    batt = make_battery(soc_init=0.7)
    res = batt.step(20.0, 3600.0)
    # Energy drawn from battery should be load/eta
    expected_kwh = -(20.0 / batt.cfg.eta_discharge)
    assert math.isclose(res["energy_delta_kwh"], expected_kwh, rel_tol=1e-6)

    res2 = batt.step(-10.0, 3600.0)
    expected_charge = 10.0 * batt.cfg.eta_charge
    assert math.isclose(res2["energy_delta_kwh"], expected_charge, rel_tol=1e-6)
