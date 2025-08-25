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
  --map-html demo.html --verbose
```

The command writes `demo.geojson`, `demo.csv`, and `demo.png` to the current
directory and, if `--map-html` is provided, an interactive Leaflet map overlaying
the bathymetry and planned route.

## Project Workflow
![output (1)](https://github.com/user-attachments/assets/887d7af9-b24a-4adf-8b18-df29376f93bf)

## Energy Architecture For Cruise Vessel Simulation
![output (2)](https://github.com/user-attachments/assets/6ba6b003-a737-4548-bcce-5edae61bc879)


