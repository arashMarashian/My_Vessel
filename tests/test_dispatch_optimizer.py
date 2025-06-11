import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from engine_loader import load_engines_from_yaml
from energy.opt_control import DispatchOptimizer


def test_dispatch_optimizer_build():
    path = "data/engine_data.yaml"
    engines = load_engines_from_yaml(path)[:1]
    with open(path) as f:
        data = yaml.safe_load(f)
    curve = data["engines"][0]["sfoc"]["HFO"]
    env = [{"wind_speed": 5.0, "wind_angle_diff": 0.0, "wave_height": 1.0} for _ in range(2)]
    opt = DispatchOptimizer(
        engines,
        horizon=2,
        env=env,
        dt_hours=1.0,
        battery_capacity_kwh=1000.0,
        sfoc_curves=[{float(k): v for k, v in curve.items()}],
    )
    assert opt.model is not None
