from __future__ import annotations
import base64, io
from typing import Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def make_overlay_data_url(arr: np.ndarray, bounds: Tuple[float, float, float, float],
                          vmin: Optional[float] = None, vmax: Optional[float] = None,
                          cmap_name: str = "viridis", opacity: float = 0.7) -> Tuple[str, Tuple[float, float, float, float], int]:
    """
    Returns (data_url_png, (south, west, north, east), visible_px)
    - Land (arr >= 0) and NaNs are fully transparent
    - Only water depths (arr < 0) are colorized
    """
    S, W, N, E = bounds
    a = arr.astype("float32")
    water = np.isfinite(a) & (a < 0.0)
    if vmin is None:
        vmin = float(np.nanpercentile(a[water], 5)) if np.any(water) else -1.0
    if vmax is None:
        vmax = float(np.nanpercentile(a[water], 95)) if np.any(water) else 0.0
    # clip to [vmin, vmax] for color scaling
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.get_cmap(cmap_name)

    rgba = np.zeros((a.shape[0], a.shape[1], 4), dtype="float32")
    if np.any(water):
        # Colorize all water (clip colors to [vmin, vmax]) and keep land/NaN fully transparent
        colored = cmap(norm(np.where(water, a, np.nan)))
        rgba[...] = 0.0
        rgba[water] = colored[water]
        # Make all water visible with the chosen opacity so there are no "holes" outside percentile range
        rgba[..., 3] = 0.0
        rgba[water, 3] = opacity
    # everything else (land or NaN) stays alpha=0

    h, w = rgba.shape[:2]
    visible = int((rgba[..., 3] > 0).sum())

    fig = plt.figure(frameon=False)
    fig.set_size_inches(w / 100.0, h / 100.0)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(rgba, origin="upper")
    buf = io.BytesIO()
    fig.canvas.print_png(buf)
    plt.close(fig)
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return data_url, (S, W, N, E), visible
