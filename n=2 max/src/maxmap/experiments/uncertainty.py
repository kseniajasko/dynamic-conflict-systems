r"""Part 4 of the checks: the uncertainty exponent of the colour boundaries.

:math:`f(\varepsilon)` is the fraction of random initial points whose phase changes under a
displacement of length :math:`\varepsilon` in a uniformly random direction.  For boundaries
of box-counting dimension :math:`D` one has :math:`f(\varepsilon)\sim\varepsilon^{2-D}`
(McDonald, Grebogi, Ott and Yorke, Physica D 17 (1985)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from maxmap.experiments.phases import classify_phases

__all__ = ["UncertaintyRow", "fitted_exponent", "local_slopes", "uncertainty_rows"]

_Z95 = 1.959963984540054  # standard normal quantile of 0.975


@dataclass(frozen=True, slots=True)
class UncertaintyRow:
    """Counts at one displacement length."""

    eps: float
    changed: int
    valid: int
    not_converged: int

    @property
    def fraction(self) -> float:
        r"""The fraction :math:`f(\varepsilon)`."""
        return self.changed / self.valid

    @property
    def ratio(self) -> float:
        r"""The ratio :math:`f(\varepsilon)/\varepsilon`."""
        return self.fraction / self.eps

    def ratio_interval(self, z: float = _Z95) -> tuple[float, float]:
        r"""Wilson score interval of :math:`f(\varepsilon)/\varepsilon` (95 % by default)."""
        n, p = self.valid, self.fraction
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return (centre - half) / self.eps, (centre + half) / self.eps


def uncertainty_rows(
    alpha: float,
    rng: np.random.Generator,
    *,
    n_points: int,
    eps: tuple[float, ...],
    max_steps: int,
    tol: float = 1e-6,
) -> list[UncertaintyRow]:
    r"""Compute :math:`f(\varepsilon)` for several displacement lengths.

    The random numbers are drawn in the order ``x0``, ``y0``, direction angle (as for the
    published results); the same base points and directions are used for all lengths.
    Displaced points outside the open square, and points that do not converge, are excluded.
    """
    x0 = rng.uniform(0.5, 1.0, n_points)
    y0 = rng.uniform(0.5, 1.0, n_points)
    theta = rng.uniform(0.0, 2.0 * np.pi, n_points)
    base = classify_phases(x0, y0, alpha, max_steps=max_steps, tol=tol)
    base_ok = base >= 0
    rows = []
    for e in eps:
        x1 = x0 + e * np.cos(theta)
        y1 = y0 + e * np.sin(theta)
        inside = (x1 > 0.5) & (x1 < 1.0) & (y1 > 0.5) & (y1 < 1.0) & base_ok
        moved = classify_phases(x1[inside], y1[inside], alpha, max_steps=max_steps, tol=tol)
        good = moved >= 0
        changed = int(np.sum(moved[good] != base[inside][good]))
        not_converged = int((~good).sum()) + int((~base_ok).sum())
        rows.append(UncertaintyRow(e, changed, int(good.sum()), not_converged))
    return rows


def local_slopes(rows: list[UncertaintyRow]) -> list[float]:
    r"""Slopes :math:`\Delta\log f/\Delta\log\varepsilon` between consecutive lengths."""
    log_f = np.log([row.fraction for row in rows])
    log_e = np.log([row.eps for row in rows])
    return [float(s) for s in np.diff(log_f) / np.diff(log_e)]


def fitted_exponent(rows: list[UncertaintyRow]) -> tuple[float, float]:
    r"""Weighted least-squares slope of :math:`\log f` against :math:`\log\varepsilon`.

    The weights are the inverse binomial variances of :math:`\log f`,
    :math:`\operatorname{Var}\log f\approx(1-f)/(n f)`.

    Returns
    -------
    tuple of float
        The exponent :math:`2-D` and its standard error.
    """
    log_e = np.log([row.eps for row in rows])
    log_f = np.log([row.fraction for row in rows])
    weights = np.array([row.changed / (1.0 - row.fraction) for row in rows])
    mean_e = np.average(log_e, weights=weights)
    mean_f = np.average(log_f, weights=weights)
    sxx = float(np.sum(weights * (log_e - mean_e) ** 2))
    slope = float(np.sum(weights * (log_e - mean_e) * (log_f - mean_f)) / sxx)
    return slope, math.sqrt(1.0 / sxx)
