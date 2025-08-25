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
    * NaNs are fully transparent
    * Values outside [vmin, vmax] are fully transparent
    """
    S, W, N, E = bounds
    a = arr.astype("float32")
    mask = ~np.isfinite(a)
    if vmin is None:
        vmin = float(np.nanpercentile(a, 5)) if np.isfinite(a).any() else 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(a, 95)) if np.isfinite(a).any() else 1.0
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm(a))
    # Transparent where NaN or outside range
    outside = (a < vmin) | (a > vmax) | ~np.isfinite(a)
    rgba[..., 3] = np.where(outside, 0.0, opacity).astype("float32")
    h, w = rgba.shape[:2]
    visible = int((~outside).sum())

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
