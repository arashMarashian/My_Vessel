from pathlib import Path

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
            "engine_yaml": "data/engine_data.yaml",
            "max_engines": 1,
            "battery": {
                "capacity_kwh": 500.0,
                "soc_kwh": 250.0,
            },
            "target_speed_kn": 6.0,
            "dt_s": 30,
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
