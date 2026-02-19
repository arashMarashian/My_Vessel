"""Simple diesel generator fuel model using SFOC curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


def _validate_curve(curve: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts = sorted((float(x), float(y)) for x, y in curve)
    if not pts:
        raise ValueError("sfoc_curve must be non-empty")
    for x, y in pts:
        if not (0.0 <= x <= 1.0):
            raise ValueError("load ratio breakpoints must lie within [0, 1]")
        if y <= 0:
            raise ValueError("SFOC values must be positive")
    # Ensure unique x values by keeping last occurrence.
    dedup: List[Tuple[float, float]] = []
    for x, y in pts:
        if dedup and abs(dedup[-1][0] - x) < 1e-9:
            dedup[-1] = (x, y)
        else:
            dedup.append((x, y))
    return dedup


@dataclass
class DieselGenerator:
    """Diesel generator with a piecewise-linear SFOC fuel model.

    Parameters
    ----------
    p_max_kw:
        Rated electrical power in kilowatts.
    p_min_kw:
        Minimum sustainable power. Any non-zero dispatch is clamped to at
        least this level to avoid the highly inefficient low-load regime.
    sfoc_curve:
        Optional iterable of ``(load_ratio, sfoc_g_kwh)`` breakpoints. If
        omitted, a representative medium-speed generator curve is used:

        ===========  =============
        Load ratio    SFOC (g/kWh)
        -----------  -------------
        0.20          250
        0.40          215
        0.70          185
        0.90          205
        1.00          225
        ===========  =============

        The curve is linearly interpolated between points and extrapolated as
        flat below the first and above the last breakpoint.
    """

    p_max_kw: float
    p_min_kw: float
    sfoc_curve: List[Tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        if self.p_max_kw <= 0:
            raise ValueError("p_max_kw must be positive")
        if not (0 < self.p_min_kw <= self.p_max_kw):
            raise ValueError("p_min_kw must be within (0, p_max_kw]")
        if self.sfoc_curve is None:
            # Representative curve roughly matching OEM datasheets.
            self.sfoc_curve = _validate_curve(
                [
                    (0.2, 250.0),
                    (0.4, 215.0),
                    (0.7, 185.0),
                    (0.9, 205.0),
                    (1.0, 225.0),
                ]
            )
        else:
            self.sfoc_curve = _validate_curve(self.sfoc_curve)

    def clamp_power(self, requested_kw: float) -> float:
        """Return the realizable generator power for a requested setpoint."""

        req = float(requested_kw)
        if req <= 0:
            return 0.0
        if req < self.p_min_kw:
            return self.p_min_kw
        return min(req, self.p_max_kw)

    def fuel_kg_per_s(self, requested_kw: float) -> float:
        """Approximate instantaneous fuel flow in kg/s for ``requested_kw``.

        ``requested_kw`` is clamped to ``[0, p_max_kw]``. Any non-zero value
        is further raised to ``p_min_kw`` to represent the minimum efficient
        loading rule. The fuel flow is the commanded power multiplied by the
        interpolated SFOC curve and converted from g/h to kg/s.
        """

        p_kw = self.clamp_power(requested_kw)
        if p_kw <= 0:
            return 0.0
        load_ratio = p_kw / self.p_max_kw
        sfoc = self._interp_sfoc(load_ratio)
        fuel_g_per_h = sfoc * p_kw
        return fuel_g_per_h / 3_600_000.0  # 3600 s/h * 1000 g/kg

    # Internal helpers -------------------------------------------------
    def _interp_sfoc(self, load_ratio: float) -> float:
        pts = self.sfoc_curve or []
        if not pts:
            raise RuntimeError("SFOC curve is missing")
        r = max(0.0, min(1.0, float(load_ratio)))
        prev_x, prev_y = pts[0]
        if r <= prev_x:
            return prev_y
        for x, y in pts[1:]:
            if r <= x:
                span = x - prev_x
                if span <= 0:
                    return y
                frac = (r - prev_x) / span
                return prev_y + frac * (y - prev_y)
            prev_x, prev_y = x, y
        return pts[-1][1]


__all__ = ["DieselGenerator"]
