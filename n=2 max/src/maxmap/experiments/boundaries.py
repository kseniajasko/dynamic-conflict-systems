"""Part 3 of the checks: colour boundaries of Fig. 6 located by bisection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maxmap._types import FloatArray, IntArray
from maxmap.analysis import x_sym
from maxmap.experiments.phases import classify_phases
from maxmap.model import phi

__all__ = ["BoundaryPairs", "bisect_boundary", "closest_approach", "sample_boundary_pairs"]


@dataclass(frozen=True, slots=True, eq=False)
class BoundaryPairs:
    """Horizontally neighbouring grid points of different phases."""

    x_left: FloatArray
    x_right: FloatArray
    y: FloatArray


def sample_boundary_pairs(
    grid_x: FloatArray,
    grid_y: FloatArray,
    phases: IntArray,
    rng: np.random.Generator,
    size: int,
) -> BoundaryPairs:
    """Sample (without replacement) pairs of horizontal neighbours with different phases.

    Pairs in which a point has not converged are excluded.  The sample is drawn with
    ``rng.choice(n_pairs, size=min(size, n_pairs), replace=False)``.
    """
    left, right = phases[:, :-1], phases[:, 1:]
    rows, cols = np.nonzero((left != right) & (left >= 0) & (right >= 0))
    pick = rng.choice(rows.size, size=min(size, rows.size), replace=False)
    rows, cols = rows[pick], cols[pick]
    return BoundaryPairs(grid_x[rows, cols], grid_x[rows, cols + 1], grid_y[rows, cols])


def bisect_boundary(
    pairs: BoundaryPairs, alpha: float | FloatArray, *, bisections: int, max_steps: int, tol: float
) -> FloatArray:
    """Locate a point of the phase boundary on each segment by bisection.

    Returns
    -------
    numpy.ndarray
        Midpoints of the final brackets (the ordinates are ``pairs.y``).
    """
    x_left, x_right, y = pairs.x_left, pairs.x_right, pairs.y
    phase_left = classify_phases(x_left, y, alpha, max_steps=max_steps, tol=tol)
    for _ in range(bisections):
        x_mid = (x_left + x_right) / 2.0
        same = classify_phases(x_mid, y, alpha, max_steps=max_steps, tol=tol) == phase_left
        x_left, x_right = np.where(same, x_mid, x_left), np.where(same, x_right, x_mid)
    result: FloatArray = (x_left + x_right) / 2.0
    return result


def _x_sym_per_point(alpha: float | FloatArray, shape: tuple[int, ...]) -> FloatArray:
    a = np.broadcast_to(np.asarray(alpha, dtype=np.float64), shape)
    s = np.empty(shape)
    for value in np.unique(a):
        s[a == value] = x_sym(float(value))
    return s


def closest_approach(
    x: FloatArray, y: FloatArray, alpha: float | FloatArray, *, steps: int
) -> tuple[FloatArray, FloatArray]:
    r"""Closest approach of trajectories to the diagonal and to the symmetric saddle.

    Returns
    -------
    tuple of numpy.ndarray
        ``(to_diagonal, to_saddle)``: :math:`\min_t|x^t-y^t|` and the minimal Euclidean
        distance to :math:`(x_{\mathrm{sym}},x_{\mathrm{sym}})` over ``t = 0, ..., steps``.
    """
    s = _x_sym_per_point(alpha, np.shape(x))
    to_diagonal = np.abs(x - y)
    to_saddle = np.hypot(x - s, y - s)
    for _ in range(steps):
        x, y = phi(x, y, alpha)
        to_diagonal = np.minimum(to_diagonal, np.abs(x - y))
        to_saddle = np.minimum(to_saddle, np.hypot(x - s, y - s))
    return to_diagonal, to_saddle
