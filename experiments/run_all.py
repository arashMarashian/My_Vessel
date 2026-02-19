from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from my_vessel.pipeline.run_scenario import run_scenario
from my_vessel.bathy.grid import latlon_to_rc


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    keys = set(base) | set(override)
    for key in keys:
        if key in base and key in override:
            if isinstance(base[key], dict) and isinstance(override[key], dict):
                result[key] = _deep_merge(base[key], override[key])
            else:
                result[key] = override[key]
        elif key in override:
            result[key] = override[key]
        else:
            result[key] = base[key]
    return result


def _slugify(name: str) -> str:
    safe = [c if c.isalnum() or c in ("-", "_") else "_" for c in name.lower()]
    return "".join(safe).strip("_") or "scenario"


def _plot_path(result: Dict[str, Any], out_path: Path) -> None:
    grid = np.asarray(result["grid"])
    bounds = tuple(result["grid_bounds"])
    path_rc = result.get("path", [])
    smoothed = result.get("smoothed_path", [])
    start = result.get("start")
    goal = result.get("goal")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap="Greys", origin="upper")

    if path_rc:
        rows, cols = zip(*path_rc)
        ax.plot(cols, rows, "r.-", label="grid path", linewidth=1.5)

    if smoothed:
        smooth_rc = [latlon_to_rc(lat, lon, bounds, grid.shape) for lat, lon in smoothed]
        s_rows, s_cols = zip(*smooth_rc)
        ax.plot(s_cols, s_rows, "c-", label="smoothed (collision-free)", linewidth=2.0)

    if start:
        sr, sc = latlon_to_rc(*start, bounds, grid.shape)
        ax.scatter([sc], [sr], c="green", marker="o", label="start")
    if goal:
        gr, gc = latlon_to_rc(*goal, bounds, grid.shape)
        ax.scatter([gc], [gr], c="orange", marker="x", label="goal")

    ax.set_title("Planned Route")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_summary(rows: List[Dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "scenario",
        "distance_nm",
        "travel_time_s",
        "fuel_kg",
        "runtime_s",
        "path_cells",
        "smoothed_points",
        "energy_mode",
        "energy_policy",
        "energy_fuel_kg_total",
        "energy_min_soc",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_energy_timeseries(slug: str, energy: Dict[str, Any], out_dir: Path) -> None:
    load_series = energy.get("load_series") or []
    if not load_series:
        return
    dt_s = float(energy.get("dt_s", 0.0))
    if dt_s <= 0:
        return
    p_gen = energy.get("p_gen_series") or [0.0] * len(load_series)
    p_batt = energy.get("p_batt_series") or [0.0] * len(load_series)
    soc = energy.get("soc_series") or []
    times = [dt_s * i for i in range(len(load_series))]

    rows = []
    for idx, (t, load_kw) in enumerate(zip(times, load_series)):
        rows.append(
            {
                "time_s": t,
                "load_kw": load_kw,
                "p_gen_kw": p_gen[idx] if idx < len(p_gen) else 0.0,
                "p_batt_kw": p_batt[idx] if idx < len(p_batt) else 0.0,
                "soc": soc[idx] if idx < len(soc) else "",
            }
        )

    csv_path = out_dir / f"{slug}_energy_timeseries.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time_s", "load_kw", "p_gen_kw", "p_batt_kw", "soc"])
        writer.writeheader()
        writer.writerows(rows)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(times, load_series, label="Load (kW)", color="tab:blue")
    ax1.plot(times, p_gen[: len(times)], label="Generator (kW)", color="tab:orange")
    if any(abs(x) > 1e-6 for x in p_batt):
        ax1.plot(times, p_batt[: len(times)], label="Battery (kW)", color="tab:green")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Power [kW]")
    ax1.grid(True, alpha=0.3)
    handles, labels = ax1.get_legend_handles_labels()

    if soc:
        ax2 = ax1.twinx()
        ax2.plot(times[: len(soc)], soc, label="SOC", color="tab:red", linestyle="--")
        ax2.set_ylabel("State of Charge")
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2
    if handles:
        ax1.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"{slug}_energy_timeseries.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scenario experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/paper_minimal.yaml",
        help="Path to the YAML experiment config",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory where timestamped results folders will be created",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    with config_path.open() as f:
        raw_cfg = yaml.safe_load(f) or {}

    if "scenarios" in raw_cfg:
        defaults = raw_cfg.get("defaults", {})
        scenarios_raw = raw_cfg.get("scenarios", [])
    else:
        defaults = {}
        scenarios_raw = [raw_cfg]

    if not scenarios_raw:
        raise SystemExit("No scenarios defined in config")

    base_dir = config_path.parent.resolve()
    scenarios: List[Dict[str, Any]] = []
    for idx, entry in enumerate(scenarios_raw, start=1):
        cfg = _deep_merge(defaults, entry)
        cfg.setdefault("name", f"scenario_{idx}")
        cfg["_base_dir"] = str(base_dir)
        scenarios.append(cfg)

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.results_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []

    for cfg in scenarios:
        name = cfg["name"]
        slug = _slugify(name)
        result = run_scenario(cfg)
        metrics = result.get("metrics", {})
        energy = result.get("energy_result") or {}
        summary_rows.append(
            {
                "scenario": name,
                "distance_nm": metrics.get("distance_nm", 0.0),
                "travel_time_s": metrics.get("travel_time_s", 0.0),
                "fuel_kg": metrics.get("fuel_kg", 0.0),
                "runtime_s": metrics.get("runtime_s", 0.0),
                "path_cells": metrics.get("path_cells", 0),
                "smoothed_points": metrics.get("smoothed_points", 0),
                "energy_mode": energy.get("mode"),
                "energy_policy": energy.get("policy"),
                "energy_fuel_kg_total": energy.get("fuel_kg_total"),
                "energy_min_soc": energy.get("soc_min"),
            }
        )

        plot_path = out_dir / f"{slug}_path.png"
        _plot_path(result, plot_path)

        detail_path = out_dir / f"{slug}_result.json"
        with detail_path.open("w") as f:
            json.dump(
                {
                    "name": name,
                    "metrics": metrics,
                    "environment": result.get("environment_samples"),
                    "energy_result": energy or None,
                },
                f,
                indent=2,
            )
        if energy:
            _write_energy_timeseries(slug, energy, out_dir)

    _write_summary(summary_rows, out_dir / "summary.csv")
    print(f"Wrote {len(summary_rows)} scenario(s) to {out_dir}")


if __name__ == "__main__":
    main()
