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

## Project Workflow
![output (1)](https://github.com/user-attachments/assets/887d7af9-b24a-4adf-8b18-df29376f93bf)



