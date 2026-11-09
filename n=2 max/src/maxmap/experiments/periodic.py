r"""Search for periodic orbits of the map of maxima by Newton's method.

For a period :math:`n`, Newton's method is applied to :math:`\Phi^n(w)-w` from many random
starting points, with the Jacobian of :math:`\Phi^n` given by the chain rule over the
branches visited.  Accepted solutions stay at a positive distance from the switching curves
(so that :math:`\Phi^n` is smooth there); orbits are reduced to their minimal period and
deduplicated.  This is a search, not a proof: it can miss orbits with small Newton basins
and does not consider orbits through the switching curves (these lie in the exceptional set
:math:`E_\alpha`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maxmap._types import FloatArray
from maxmap.model import jacobian, phi

__all__ = ["PeriodicOrbit", "find_periodic_orbits"]

_RESIDUAL_TOL = 1e-12
_MARGIN_TOL = 1e-9
_DEDUP_DECIMALS = 8
_MAX_NEWTON_STEP = 0.05


@dataclass(frozen=True, slots=True)
class PeriodicOrbit:
    """A periodic orbit found by the search."""

    period: int
    points: tuple[tuple[float, float], ...]
    multipliers: tuple[complex, complex]

    @property
    def is_attracting(self) -> bool:
        """Whether both multipliers lie in the open unit disc."""
        return max(abs(m) for m in self.multipliers) < 1.0

    @property
    def is_symmetric(self) -> bool:
        """Whether the orbit lies on the diagonal."""
        return all(abs(x - y) < 1e-9 for x, y in self.points)


def _branch_margin(x: FloatArray, y: FloatArray, alpha: float) -> FloatArray:
    """Distance of the image coordinates from 1/2, i.e. from the switching curves."""
    xn, yn = phi(x, y, alpha)
    result: FloatArray = np.minimum(xn - 0.5, yn - 0.5)
    return result


def _iterate_with_jacobian(
    x: FloatArray, y: FloatArray, alpha: float, period: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    mono = np.broadcast_to(np.eye(2), (*x.shape, 2, 2)).copy()
    margin = np.full(x.shape, np.inf)
    for _ in range(period):
        mono = jacobian(x, y, alpha) @ mono
        margin = np.minimum(margin, _branch_margin(x, y, alpha))
        x, y = phi(x, y, alpha)
    return x, y, mono, margin


def _orbit_points(x: float, y: float, alpha: float, period: int) -> list[tuple[float, float]]:
    points = [(x, y)]
    for _ in range(period - 1):
        xn, yn = phi(np.array([x]), np.array([y]), alpha)
        x, y = float(xn[0]), float(yn[0])
        points.append((x, y))
    return points


def find_periodic_orbits(
    alpha: float,
    rng: np.random.Generator,
    *,
    max_period: int,
    starts: int,
    newton_steps: int,
) -> list[PeriodicOrbit]:
    """Find periodic orbits of periods ``1..max_period`` (deduplicated, minimal periods)."""
    found: dict[tuple[int, tuple[tuple[float, float], ...]], PeriodicOrbit] = {}
    for period in range(1, max_period + 1):
        x = rng.uniform(0.5, 1.0, starts)
        y = rng.uniform(0.5, 1.0, starts)
        for _ in range(newton_steps):
            xn, yn, mono, _ = _iterate_with_jacobian(x, y, alpha, period)
            fx, fy = xn - x, yn - y
            a = mono - np.eye(2)
            det = a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]
            det = np.where(np.abs(det) < 1e-300, 1e-300, det)
            dx = (a[..., 1, 1] * fx - a[..., 0, 1] * fy) / det
            dy = (-a[..., 1, 0] * fx + a[..., 0, 0] * fy) / det
            scale = np.minimum(1.0, _MAX_NEWTON_STEP / np.maximum(np.hypot(dx, dy), 1e-300))
            x = np.clip(x - scale * dx, 0.5 + 1e-12, 1.0 - 1e-12)
            y = np.clip(y - scale * dy, 0.5 + 1e-12, 1.0 - 1e-12)
        xn, yn, mono, margin = _iterate_with_jacobian(x, y, alpha, period)
        ok = (np.hypot(xn - x, yn - y) < _RESIDUAL_TOL) & (margin > _MARGIN_TOL)
        for xi, yi in zip(x[ok], y[ok], strict=True):
            points = _orbit_points(float(xi), float(yi), alpha, period)
            minimal = next(
                d
                for d in range(1, period + 1)
                if period % d == 0
                and (d == period or np.hypot(*np.subtract(points[d], points[0])) < 1e-9)
            )
            orbit_points = points[:minimal]
            key = (
                minimal,
                tuple(
                    sorted(
                        (round(p, _DEDUP_DECIMALS), round(q, _DEDUP_DECIMALS))
                        for p, q in orbit_points
                    )
                ),
            )
            if key in found:
                continue
            _, _, mono_min, _ = _iterate_with_jacobian(
                np.array([orbit_points[0][0]]), np.array([orbit_points[0][1]]), alpha, minimal
            )
            eig = np.linalg.eigvals(mono_min[0])
            found[key] = PeriodicOrbit(
                minimal, tuple(orbit_points), (complex(eig[0]), complex(eig[1]))
            )
    return sorted(found.values(), key=lambda o: (o.period, o.points))
