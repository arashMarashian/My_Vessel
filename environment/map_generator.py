import numpy as np


def generate_environment_grid(rows: int = 10, cols: int = 10) -> dict:
    """Generate a simple occupancy grid environment.

    Parameters
    ----------
    rows : int, optional
        Number of grid rows.
    cols : int, optional
        Number of grid columns.

    Returns
    -------
    dict
        Dictionary containing ``grid`` (np.ndarray), ``start`` (tuple),
        and ``goal`` (tuple).
    """
    grid = np.zeros((rows, cols), dtype=int)
    # Create a simple obstacle wall across the center row
    if rows > 2 and cols > 2:
        grid[rows // 2, 1:-1] = 1

    start = (0, 0)
    goal = (rows - 1, cols - 1)
    return {"grid": grid, "start": start, "goal": goal}
