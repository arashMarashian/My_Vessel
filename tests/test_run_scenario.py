from pathlib import Path
from typing import Any, Dict

from my_vessel.pipeline.run_scenario import run_scenario


def test_run_scenario_returns_expected_keys():
    repo_root = Path(__file__).resolve().parents[1]
    config = {
        "_base_dir": str(repo_root),
        "bathy": {
            "array": [
                [-5.0, -5.0, -5.0, -5.0],
                [-5.0, 1.0, 1.0, -5.0],
                [-5.0, -5.0, -5.0, -5.0],
                [-5.0, -5.0, -5.0, -5.0],
            ],
            "bounds": [0.0, 0.0, 4.0, 4.0],
            "min_depth_m": 1.0,
        },
        "start": [0.5, 0.5],
        "goal": [3.5, 3.5],
        "planning": {
            "densify_pts": 2,
            "smoothness": 0.25,
            "iterations": 60,
            "snap_radius": 2,
        },
        "energy": {
            "mode": "hybrid",
            "policy": "naive",
            "target_speed_kn": 6.0,
            "dt_s": 30,
            "generator": {"p_max_kw": 800.0, "p_min_kw": 200.0},
            "battery": {
                "capacity_kwh": 400.0,
                "soc_init": 0.5,
                "soc_min": 0.2,
                "soc_max": 0.95,
                "p_charge_max_kw": 150.0,
                "p_discharge_max_kw": 150.0,
                "eta_charge": 0.95,
                "eta_discharge": 0.95,
            },
            "hotel_load_kw": 50.0,
        },
        "environment": {
            "mode": "constant",
            "values": {
                "wind_speed": 1.5,
                "wind_angle_diff": 10.0,
                "wave_height": 0.2,
            },
        },
    }

    result = run_scenario(config)

    for key in ("grid", "path", "smoothed_path", "metrics"):
        assert key in result

    assert result["path"], "planner should return at least one waypoint"
    assert len(result["smoothed_path"]) >= 2

    metrics = result["metrics"]
    assert metrics["distance_nm"] > 0
    assert metrics["runtime_s"] >= 0

    env_samples = result.get("environment_samples")
    assert env_samples is not None
    assert "wind_speed" in env_samples


def _base_config() -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    return {
        "_base_dir": str(repo_root),
        "bathy": {
            "array": [
                [-5.0, -5.0, -5.0, -5.0],
                [-5.0, 1.0, 1.0, -5.0],
                [-5.0, -5.0, -5.0, -5.0],
                [-5.0, -5.0, -5.0, -5.0],
            ],
            "bounds": [0.0, 0.0, 4.0, 4.0],
            "min_depth_m": 1.0,
        },
        "start": [0.5, 0.5],
        "goal": [3.5, 3.5],
        "planning": {"densify_pts": 2, "smoothness": 0.25, "iterations": 60, "snap_radius": 2},
        "environment": {
            "mode": "constant",
            "values": {"wind_speed": 1.0, "wind_angle_diff": 5.0, "wave_height": 0.2},
        },
        "energy": {
            "mode": "hybrid",
            "policy": "load_smoothing",
            "dt_s": 20,
            "target_speed_kn": 6.0,
            "generator": {"p_max_kw": 900.0, "p_min_kw": 250.0},
            "battery": {
                "capacity_kwh": 300.0,
                "soc_init": 0.6,
                "soc_min": 0.2,
                "soc_max": 0.9,
                "p_charge_max_kw": 120.0,
                "p_discharge_max_kw": 120.0,
                "eta_charge": 0.96,
                "eta_discharge": 0.95,
            },
            "hotel_load_kw": 40.0,
        },
    }


def test_energy_result_contains_required_keys():
    config = _base_config()
    result = run_scenario(config)
    energy = result.get("energy_result")
    assert energy is not None
    for key in ("fuel_kg_total", "soc_series", "p_gen_series", "p_batt_series", "load_series"):
        assert key in energy
    assert len(energy["load_series"]) == len(energy["p_gen_series"]) == len(energy["p_batt_series"])
    if energy["soc_series"]:
        assert energy["soc_min"] == min(energy["soc_series"])


def test_diesel_only_mode_smoke():
    config = _base_config()
    config["energy"]["mode"] = "diesel_only"
    config["energy"].pop("battery", None)
    result = run_scenario(config)
    energy = result.get("energy_result")
    assert energy is not None
    assert energy["mode"] == "diesel_only"
    assert len(energy["p_batt_series"]) == len(energy["load_series"])


def test_synthetic_peaks_uses_battery():
    config = _base_config()
    config["energy"]["load_profile"] = "synthetic_peaks"
    config["energy"]["policy"] = "load_smoothing"
    result = run_scenario(config)
    energy = result.get("energy_result")
    assert energy is not None
    batt_series = energy.get("p_batt_series", [])
    soc_series = energy.get("soc_series", [])
    assert batt_series
    assert max(abs(p) for p in batt_series) > 1e-3
    assert soc_series
    initial_soc = soc_series[0]
    assert any(abs(s - initial_soc) > 1e-4 for s in soc_series[1:])
