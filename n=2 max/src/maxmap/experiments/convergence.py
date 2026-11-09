"""Part 1 of the checks: convergence of random initial points to the asymmetric attractor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maxmap._types import BoolArray, FloatArray, IntArray
from maxmap.analysis import ALPHA_CRITICAL, asymmetric_attractor
from maxmap.model import phi

__all__ = ["ConvergenceResult", "capture_steps", "random_convergence"]


@dataclass(frozen=True, slots=True)
class ConvergenceResult:
    """Statistics of the convergence of a random sample at one value of the parameter."""

    alpha: float
    n_points: int
    converged: int
    median_steps: int | None
    max_steps: int | None
    wrong_winner: int | None
    """For ``alpha > 2``: number of trajectories whose limit is not the fixed point with the
    larger coordinate in the position of the larger initial maximum (must be zero by
    Lemma 2(2)); ``None`` for ``alpha < 2``."""

    @property
    def converged_fraction(self) -> float:
        """Fraction of convergent trajectories."""
        return self.converged / self.n_points


def capture_steps(
    x0: FloatArray, y0: FloatArray, alpha: float, *, max_steps: int, tol: float
) -> tuple[IntArray, BoolArray]:
    """First even step at which each trajectory is within ``tol`` of the attractor.

    The distance is the Euclidean distance to the nearer of the points :math:`(x^*,y^*)`,
    :math:`(y^*,x^*)`.

    Returns
    -------
    tuple of numpy.ndarray
        ``(steps, at_first)``: the capture step (``-1`` if not captured within
        ``max_steps``) and whether the nearer point is :math:`(x^*,y^*)`.
    """
    u, v = asymmetric_attractor(alpha)
    x = np.array(x0, dtype=np.float64)
    y = np.array(y0, dtype=np.float64)
    steps = np.full(x.size, -1, dtype=np.int64)
    at_first = np.zeros(x.size, dtype=np.bool_)
    idx = np.arange(x.size)
    for step in range(1, max_steps + 1):
        x, y = phi(x, y, alpha)
        if step % 2:
            continue
        dist_first = np.hypot(x - u, y - v)
        dist_second = np.hypot(x - v, y - u)
        hit = np.minimum(dist_first, dist_second) < tol
        if not hit.any():
            continue
        steps[idx[hit]] = step
        at_first[idx[hit]] = dist_first[hit] < dist_second[hit]
        keep = ~hit
        x, y, idx = x[keep], y[keep], idx[keep]
        if idx.size == 0:
            break
    return steps, at_first


def random_convergence(
    alpha: float, rng: np.random.Generator, *, n_points: int, max_steps: int, tol: float
) -> ConvergenceResult:
    r"""Iterate ``n_points`` uniform random initial points of :math:`(\tfrac12,1)^2`.

    The initial points are drawn as ``x = rng.uniform(0.5, 1, n)``, then
    ``y = rng.uniform(0.5, 1, n)`` (the order of the published checks).
    """
    x0 = rng.uniform(0.5, 1.0, n_points)
    y0 = rng.uniform(0.5, 1.0, n_points)
    steps, at_first = capture_steps(x0, y0, alpha, max_steps=max_steps, tol=tol)
    converged = steps > 0
    count = int(converged.sum())
    wrong = None
    if alpha > ALPHA_CRITICAL:
        wrong = int(np.sum(converged & (at_first != (x0 > y0))))
    median = int(np.median(steps[converged])) if count else None
    largest = int(np.max(steps[converged])) if count else None
    return ConvergenceResult(alpha, n_points, count, median, largest, wrong)
