from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root directory (assumes this file lives in utils/)."""
    return Path(__file__).resolve().parent.parent


def results_root() -> Path:
    """Return the root directory to store all run outputs: <repo>/Results."""
    root = project_root() / "Results"
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_results_subdir(name: str) -> Path:
    """Ensure and return a subdirectory under Results for a specific script/module."""
    subdir = results_root() / name
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir

