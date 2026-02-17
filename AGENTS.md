# Agent Onboarding: My_Vessel

## Project Snapshot
- Purpose: simulate and plan high-speed vessel behavior, including hydrodynamics, routing, and energy analysis.
- Languages/stack: Python 3.11+, `pip install -r requirements.txt`; optional Streamlit UI.
- Entry points: `python test.py` (smoke), `pytest` (unit tests), `bathy-route` / `python -m cli.bathy_route` (routing CLI), `streamlit run app/streamlit_app.py` (UI).
- Secrets: load from `.env` (copy from `.env.example`). Never commit `.env` or raw keys.

## Directory Guide
- `my_vessel/`: core package (hydrodynamics, bathymetry fetch, routing, energy, utilities). Subpackages are Python packages; keep new modules package-aware.
- `controllers/`, `models/`: control schemes and physics models underpinning `test.py` and planners.
- `path_planner/`: grid planners (A*, smoothing) consumed by CLI + tests.
- `energy/`, `environment/`: energy system models and weather abstractions used by planners and notebooks.
- `cli/`: `bathy_route.py` exposes the CLI/console script.
- `app/`: Streamlit UI (`streamlit_app.py`) for interactive routing/simulation. Reads env vars dynamically and persists results under `Results/`.
- `examples/`: runnable scripts per subsystem (A*, dispatch, energy, plotting). Use them as execution references.
- `tests/`: pytest suite covering planners, bathymetry, snapping, power models, etc. Extend tests alongside new code.
- `data/`, `notebooks/`: user-supplied datasets and exploratory notebooks (never commit sensitive files here).
- `Results/`: runtime outputs/cache; gitignored.

## Conventions
- Python formatting: follow existing style (PEP8-ish, standard typing). Keep modules importable (no notebook-only patterns inside packages).
- Environment variables: read via `os.getenv` or `my_vessel.config`. Required ones belong in `.env.example`; load them with `set -a; source .env; set +a` or `dotenv`.
- Secrets & data: never embed API keys in code/docs. Use `.env.example` placeholders and keep `.env` out of git (already ignored).
- Caching/output: use `utils.paths.ensure_results_subdir` so artifacts land under `Results/` or package-defined caches.
- Tests should stay deterministic (offline by default). Introduce network calls only behind explicit flags/env guards.

## Common Commands
```bash
# Install deps
pip install -r requirements.txt

# (Recommended) create conda env if needed
conda create -n myvessel311 python=3.11 -y
conda activate myvessel311
pip install -r requirements.txt

# Secrets setup
cp .env.example .env   # then edit OPENTOPO_API_KEY, optional DEM_TYPE/MAX_PIXELS
set -a; source .env; set +a

# Run fast smoke test
python test.py

# Run full suite
pytest

# Specific tests
pytest tests/test_astar_planner.py -q

# Routing CLI (requires OPENTOPO_API_KEY when fetching DEM)
MPLBACKEND=Agg python -m cli.bathy_route --bbox 60.1 20.1 60.5 20.7 \
  --draft 6.5 --ukc 1.0 --start 60.45 20.15 --goal 60.15 20.65 \
  --target-speed-kn 12 --dilate-cells 1 --downsample 2 \
  --out-prefix demo --map-html demo_map.html --verbose

# Streamlit UI
streamlit run app/streamlit_app.py

# Selected examples
python examples/a_star_example.py
python examples/dispatch_optimizer_example.py
python examples/power_model_example.py
```

## Testing & QA Notes
- `pytest` discovers tests in `tests/`; keep fixtures fast and offline.
- `test.py` exercises the planing hull + controller loop for sanity.
- Add new tests for each new module; prefer property/unit tests over notebooks.

## Data & Secrets Handling
- Remote DEM downloads hit OpenTopography; failures usually mean missing `OPENTOPO_API_KEY`.
- Large downloads/cache stored in `Results/` (already ignored). Clean with manual deletion when space is needed.
- When adding new env vars (e.g., API tokens, dataset toggles), document them in `.env.example` and README.

## Agent Tips
- Prefer `rg` for code searches (fast, used in repo tooling).
- CLI uses `argparse`; re-run `python -m cli.bathy_route --help` after adding flags.
- Streamlit app expects Matplotlib backend set to `Agg` in headless runs; `app/streamlit_app.py` sets defaults if unset.
- Keep docs (README.md + my_vessel.egg-info/PKG-INFO) aligned whenever you change user-facing instructions.
