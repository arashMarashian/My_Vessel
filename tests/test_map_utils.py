import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.map_utils import load_environment_data, extract_environment_along_path


def test_extract_environment_along_path(tmp_path):
    # create temporary environment npz
    data = {
        "field1": np.arange(9).reshape(3, 3),
        "field2": np.arange(9, 18).reshape(3, 3),
    }
    save_path = tmp_path / "env.npz"
    np.savez(save_path, **data)

    env = load_environment_data(str(save_path))
    path = [(0, 0), (2, 2)]
    sampled = extract_environment_along_path(env, path)

    assert sampled["field1"] == [0, 8]
    assert sampled["field2"] == [9, 17]
