import os
import numpy as np


def load_environment_data(path="data/environment/environment_fields.npz"):
    """Load saved environment fields from an NPZ file.

    Parameters
    ----------
    path : str, optional
        Path to the ``.npz`` file containing environment arrays.

    Returns
    -------
    dict
        Dictionary mapping field names to ``numpy.ndarray`` objects.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Environment file not found: {path}")
    return {k: v for k, v in np.load(path).items()}


def extract_environment_along_path(env_data, path):
    """Sample environment values along a path.

    Parameters
    ----------
    env_data : dict
        Dictionary of 2D arrays representing environment fields.
    path : Iterable[Tuple[int, int]]
        Sequence of ``(x, y)`` grid coordinates.

    Returns
    -------
    dict
        Dictionary with the same keys as ``env_data`` containing lists of
        sampled values along the path.
    """
    sampled = {key: [] for key in env_data}
    for x, y in path:
        for key, field in env_data.items():
            sampled[key].append(field[y, x])
    return sampled
