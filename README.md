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

## Repository Structure

```
models/        # hydrodynamic models
controllers/   # control design algorithms
utils/         # plotting and helper functions
data/          # placeholder for experimental data files
notebooks/     # research notebooks and examples
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

