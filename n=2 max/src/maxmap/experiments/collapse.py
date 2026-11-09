r"""Collapse at :math:`\alpha=2` (Section 5.2 and Fig. 9)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from maxmap._types import FloatArray
from maxmap.analysis import ALPHA_CRITICAL
from maxmap.model import phi

__all__ = [
    "FIGURE_SEED",
    "FIGURE_TRAJECTORIES",
    "CollapseStatistics",
    "TrajectoryRate",
    "collapse_deviation",
    "collapse_statistics",
    "figure_initial_points",
    "trajectory_rates",
]

FIGURE_SEED: Final = 7
"""Seed of the random initial points of Fig. 9."""

FIGURE_TRAJECTORIES: Final = 5
"""Number of trajectories in Fig. 9."""


def collapse_deviation(x0: FloatArray, y0: FloatArray, steps: int) -> FloatArray:
    r"""Return the deviations :math:`G_t-\tfrac12`, :math:`G_t=\max(x^t,y^t)`, at :math:`\alpha=2`.

    Parameters
    ----------
    x0, y0
        One-dimensional arrays of initial points.
    steps
        Number of time points ``t = 0, ..., steps - 1``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(steps, len(x0))``.
    """
    x = np.array(x0, dtype=np.float64)
    y = np.array(y0, dtype=np.float64)
    out = np.empty((steps, x.size))
    for t in range(steps):
        out[t] = np.maximum(x, y) - 0.5
        x, y = phi(x, y, ALPHA_CRITICAL)
    return out


@dataclass(frozen=True, slots=True)
class CollapseStatistics:
    r"""Two-step ratios :math:`(G_{t+2}-\tfrac12)/(G_t-\tfrac12)` over a random sample."""

    n_points: int
    times: tuple[int, ...]
    max_abs_deviation: tuple[float, ...]
    r""":math:`\max|\text{ratio}-3/4|` over the sample at each time."""
    median_gap: tuple[float, ...]
    r"""Median of :math:`G_t-\tfrac12` at each time (the scale of the rounding errors)."""


def collapse_statistics(
    rng: np.random.Generator, *, n_points: int, times: tuple[int, ...]
) -> CollapseStatistics:
    """Ratios over two steps for uniform random initial points (with ``x0 != y0``)."""
    x0 = rng.uniform(0.5, 1.0, n_points)
    y0 = rng.uniform(0.5, 1.0, n_points)
    keep = x0 != y0
    dev = collapse_deviation(x0[keep], y0[keep], max(times) + 3)
    worst = []
    median = []
    for t in times:
        ratio = dev[t + 2] / dev[t]
        worst.append(float(np.max(np.abs(ratio - 0.75))))
        median.append(float(np.median(dev[t])))
    return CollapseStatistics(int(keep.sum()), times, tuple(worst), tuple(median))


def figure_initial_points() -> FloatArray:
    """Return the random initial points of Fig. 9, an array of shape ``(5, 2)``."""
    rng = np.random.default_rng(FIGURE_SEED)
    points: FloatArray = rng.uniform(0.5, 1.0, (FIGURE_TRAJECTORIES, 2))
    return points


@dataclass(frozen=True, slots=True)
class TrajectoryRate:
    r"""Rate of collapse along one trajectory at :math:`\alpha=2`."""

    start: tuple[float, float]
    ratio_deviation: float
    r""":math:`|(G_{t+2}-\tfrac12)/(G_t-\tfrac12)-\tfrac34|` at the ratio time."""
    one_step_factors: tuple[float, float]
    r"""The factors :math:`(G_{t+1}-\tfrac12)/(G_t-\tfrac12)` and
    :math:`(G_{t+2}-\tfrac12)/(G_{t+1}-\tfrac12)` at the factor time."""


def trajectory_rates(
    points: FloatArray, *, ratio_time: int, factor_time: int
) -> list[TrajectoryRate]:
    """Two-step ratios and one-step factors along the trajectories of given initial points.

    Parameters
    ----------
    points
        Initial points, an array of shape ``(m, 2)``.
    ratio_time
        Time ``t`` of the two-step ratio.
    factor_time
        Time ``t`` of the two consecutive one-step factors.
    """
    dev = collapse_deviation(points[:, 0], points[:, 1], max(ratio_time, factor_time) + 3)
    ratio = dev[ratio_time + 2] / dev[ratio_time]
    first = dev[factor_time + 1] / dev[factor_time]
    second = dev[factor_time + 2] / dev[factor_time + 1]
    return [
        TrajectoryRate(
            (float(x0), float(y0)),
            float(abs(ratio[k] - 0.75)),
            (float(first[k]), float(second[k])),
        )
        for k, (x0, y0) in enumerate(points)
    ]
