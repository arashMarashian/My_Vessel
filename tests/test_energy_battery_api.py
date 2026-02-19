import math

import pytest

from my_vessel.energy import Battery


def test_soc_init_fraction_sets_state():
    batt = Battery(capacity_kwh=200.0, soc_init=0.6, soc_min=0.2, soc_max=0.9)
    assert math.isclose(batt.soc_kwh, 120.0, rel_tol=1e-6)


def test_soc_kwh_alias_supported():
    batt = Battery(capacity_kwh=150.0, soc_kwh=45.0)
    assert math.isclose(batt.soc_kwh, 45.0, rel_tol=1e-6)


def test_conflicting_soc_args_raise():
    with pytest.raises(ValueError):
        Battery(capacity_kwh=100.0, soc_init=0.4, soc_kwh=30.0)


def test_power_limits_clamp():
    batt = Battery(
        capacity_kwh=100.0,
        soc_init=0.5,
        p_charge_max_kw=10.0,
        p_discharge_max_kw=5.0,
    )
    applied_discharge = batt.step(power_w=20_000.0, dt_h=0.5)
    assert math.isclose(applied_discharge, 5_000.0)
    applied_charge = batt.step(power_w=-20_000.0, dt_h=0.5)
    assert math.isclose(applied_charge, -10_000.0)


def test_soc_moves_with_power_sign():
    batt = Battery(
        capacity_kwh=80.0,
        soc_init=0.6,
        soc_min=0.2,
        soc_max=0.95,
        p_charge_max_kw=40.0,
        p_discharge_max_kw=40.0,
    )
    dt_h = 0.25
    soc_before = batt.soc
    applied = batt.step(power_w=30_000.0, dt_h=dt_h)
    assert applied > 0
    assert batt.soc < soc_before
    soc_mid = batt.soc
    applied = batt.step(power_w=-20_000.0, dt_h=dt_h)
    assert applied < 0
    assert batt.soc > soc_mid
