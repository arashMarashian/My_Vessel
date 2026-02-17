import my_vessel.energy.battery_model as battery_module
from my_vessel.energy import Battery


def test_smoke_imports():
    assert hasattr(battery_module, "BatteryModel")
    batt = Battery(capacity_kwh=100.0, soc_kwh=50.0)
    res = batt.step(power_w=1000.0, dt_h=1.0)
    assert isinstance(res, float)
