import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.map_generator import generate_environment_grid, plot_field
from utils.paths import ensure_results_subdir


def main():
    # Step 1: Generate environmental fields
    env_data = generate_environment_grid(size=(50, 50))

    # Step 2: Save to file under Results/env_map_example
    save_dir = ensure_results_subdir('env_map_example')
    save_path = os.path.join(str(save_dir), 'environment_fields.npz')
    np.savez(save_path, **env_data)
    print(f"Saved environment fields to {save_path}")

    # Step 3: Plot each field
    for key, field in env_data.items():
        plot_field(field, title=key)


if __name__ == '__main__':
    main()
