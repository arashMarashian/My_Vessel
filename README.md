# My_Vessel

My_Vessel is a modular research and simulation repository for developing,
testing, and analyzing hydrodynamic models and control strategies related to
marine vessels. The framework is written in Python and focuses on high-speed
planing hull dynamics, porpoising mitigation, trim control, and actuator-based
stabilization (flaps, tabs, interceptors).

## Features

- Nonlinear and linearized models for high-speed planing hulls inspired by
  classical approaches (Savitsky, Troesch, Jing Sun).
- Data-driven handling of added mass and damping terms.
- Feedback control strategies including simple feedback linearization.
- Tools for extrapolating hydrodynamic coefficients from experiments.
- Example notebooks for simulation and visualization.
- Grid-based path planning utilities with an A* demo and optional path
  smoothing for trajectory following.

## Repository Structure

```
models/        # hydrodynamic models
controllers/   # control design algorithms
utils/         # plotting and helper functions
data/          # placeholder for experimental data files
notebooks/     # research notebooks and examples
path_planner/  # simple planners for obstacle avoidance
```

Each directory is initialized as a Python package so additional modules can be
easily added.

## Getting Started

1. Install the project dependencies:

```bash
pip install -r requirements.txt
```

2. Run the demo script to verify the environment:

```bash
python test.py
```

The script runs a simple simulation of a planing hull with a feedback
linearization controller and prints the final state vector.

To try the A* planner, you can either open
`notebooks/a_star_demo.ipynb` in Jupyter or run the provided script:

```bash
python examples/a_star_example.py
```

Both approaches create a small occupancy grid and use
`AStarPlanner` to find a path while visualizing the result.

## Bathymetry routing

The repository includes an optional pipeline for planning vessel routes using
real bathymetry data. To fetch remote DEM tiles you need an
OpenTopography API key. Create a `.env` file or export the variable in your
shell:

```bash
export OPENTOPO_API_KEY=YOUR_KEY_HERE
```

Example command to download bathymetry, plan a route, and compute a speed
profile:

```bash
bathy-route --bbox 60.1 20.1 60.5 20.7 --draft 6.5 --ukc 1.0 \
  --start 60.45 20.15 --goal 60.15 20.65 --target-speed-kn 12 \
  --dilate-cells 1 --downsample 2 --out-prefix demo \
  --map-html demo_map.html --verbose
```

By default, the command writes `demo.geojson`, `demo.csv`, and `demo.png` to
`Results/bathy_route/` and, if `--map-html` is provided, an interactive Leaflet map
(`demo_map.html`). Additional time-series plots such as fuel usage and speed vs
time are also saved in the same directory.

### Conda environment (myvessel311)

If you prefer using conda, create and activate a Python 3.11 environment, then install dependencies:

```bash
conda create -n myvessel311 python=3.11 -y
conda activate myvessel311
pip install -r requirements.txt
```

If `rasterio` fails to build from source on your system, consider installing it from conda-forge instead:

```bash
conda install -n myvessel311 -c conda-forge rasterio
```

### Bathymetry quickstart (example command)

Below is a ready-to-run example using the built-in CLI module. It activates the `myvessel311` conda environment, sets `OPENTOPO_API_KEY`, selects a non-interactive Matplotlib backend (`Agg`), and computes a route with environmental data from Open-Meteo.

```bash
conda activate myvessel311

OPENTOPO_API_KEY=23016192f2637c9b8fc6137bcfc852df \
MPLBACKEND=Agg \
python -m cli.bathy_route \
  --bbox 52 -1 66 26 \
  --draft 3.0 \
  --ukc 0.5 \
  --start 60.1699 24.9384 \
  --goal 64.1466 -21.9426 \
  --target-speed-kn 20 \
  --dilate-cells 0 \
  --downsample 16 \
  --dem-type SRTM15Plus \
  --env-source openmeteo \
  --env-sample-stride 20 \
  --depart-iso '2025-08-27T10:00Z' \
  --out-prefix hel_to_rey \
  --map-html hel_to_rey.html \
  --verbose
```

What the flags mean (high level):

- `--bbox 52 -1 66 26`: search area as `S W N E` in degrees (lat/lon).
- `--draft 3.0`, `--ukc 0.5`: vessel draft and under-keel clearance in meters.
- `--start 60.1699 24.9384`, `--goal 64.1466 -21.9426`: start/goal coordinates (lat lon).
- `--target-speed-kn 20`: target cruising speed in knots for energy/profile estimates.
- `--dilate-cells 0`: obstacle dilation (grid safety buffer) in pixels.
- `--downsample 16`: resampling factor for the DEM grid (bigger → faster, coarser).
- `--dem-type SRTM15Plus`: OpenTopography dataset to fetch.
- `--env-source openmeteo`: sample wind/wave along route from Open-Meteo (requires `--depart-iso`).
- `--env-sample-stride 20`: sample every Nth waypoint to reduce network calls.
- `--depart-iso '2025-08-27T10:00Z'`: departure time in UTC ISO8601 used for environment sampling.
- `--out-prefix hel_to_rey`: prefix for output files (CSV, GeoJSON, PNG, plots).
- `--map-html hel_to_rey.html`: write an interactive HTML map to this filename.
- `--verbose`: print extra debug information during the run.

Outputs are written under `Results/bathy_route/` by default, including:

- `hel_to_rey.geojson`: planned route geometry and summary properties.
- `hel_to_rey.csv`: per-segment time, speed, power, SFOC, fuel usage, and environment.
- `hel_to_rey.png`: static route plot; plus additional time-series PNGs.

## Streamlit App

An interactive UI is available to draw the bounding box and start/goal on a map, set numerical parameters (draft, UKC, target speed, etc.), run the routing pipeline, and view interactive plots.

Run the app:

```bash
conda activate myvessel311
pip install -r requirements.txt  # ensure Streamlit deps are installed
streamlit run app/streamlit_app.py
```

What you can do in the UI:

- Draw a rectangle on the map to set `--bbox` (S W N E)
- Place two markers and select which is start and goal
- Set numeric inputs: `draft`, `ukc`, `downsample`, `dilate-cells`, `target-speed-kn`, `dt-s`
- Choose environment source: `openmeteo` (requires `depart-iso`) or `constant`
- Provide `OPENTOPO_API_KEY` (in the sidebar) or via environment variable
- Always-shown plots: cumulative fuel, speed profile, total power profile, and the route on a map
- Extra plots: select from a list (engine power/SFOC, battery SOC/power, wind, waves) and render interactively

Outputs are also written to `Results/bathy_route/` with the chosen `out-prefix` for later analysis.

### Deploying to Streamlit Community Cloud

If you deploy this repo to Streamlit Cloud and see build failures for `scipy` or `rasterio` on Python 3.13, pin the Python runtime to 3.11. This repository includes `runtime.txt` with:

```
3.11
```

Streamlit Cloud will honor this and use Python 3.11, which has prebuilt wheels for `scipy==1.13.1` and `rasterio==1.3.10`, avoiding gfortran/GDAL source builds. If you still see errors:

- Ensure `requirements.txt` is used (no conflicting `pyproject.toml`).
- Restart the app from the Streamlit Cloud dashboard after pushing changes.
- Optionally, switch to a smaller bbox or upload a local GeoTIFF in the app to validate the pipeline independent of remote DEM fetch.

## Project Workflow
![output (1)](https://github.com/user-attachments/assets/887d7af9-b24a-4adf-8b18-df29376f93bf)

## Energy Architecture For Cruise Vessel Simulation
![output (2)](https://github.com/user-attachments/assets/6ba6b003-a737-4548-bcce-5edae61bc879)
