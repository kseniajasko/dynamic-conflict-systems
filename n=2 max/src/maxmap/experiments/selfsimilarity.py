r"""Self-similarity of the phase pattern near the corners of the square.

Part 5 of the checks compares two windows with a common corner point by point on offset grids
with the same relative positions; for windows ``[lo, 1]^2`` this is a comparison under scaling
about :math:`(1,1)`, and for windows ``[1/2, hi]^2`` under scaling about
:math:`(\tfrac12,\tfrac12)`.  The additional checks quantify the scaling by :math:`\tfrac34`
about :math:`(\tfrac12,\tfrac12)`, the number of layers near this corner, and the position of
the differing points near :math:`(1,1)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from maxmap._types import IntArray
from maxmap.experiments.config import Window
from maxmap.experiments.phases import classify_phases, offset_grid, window_phases

__all__ = [
    "CornerComparison",
    "colour_changes_along_ray",
    "consecutive_disagreements",
    "corner_one_comparisons",
    "disagreement",
    "scaling_disagreement",
]


def disagreement(phases_a: IntArray, phases_b: IntArray) -> float:
    """Share of grid points with different phases."""
    if phases_a.shape != phases_b.shape:
        raise ValueError("the grids must have the same shape")
    return float(np.mean(phases_a != phases_b))


def consecutive_disagreements(
    windows: tuple[Window, ...], n: int, *, max_steps: int, tol: float
) -> tuple[list[float], bool]:
    """Disagreements between consecutive windows of a sequence.

    Returns
    -------
    tuple
        The list of shares and whether all grid points converged.
    """
    phases = [window_phases(w, n, max_steps=max_steps, tol=tol) for w in windows]
    all_converged = all(bool((p >= 0).all()) for p in phases)
    shares = [disagreement(a, b) for a, b in pairwise(phases)]
    return shares, all_converged


def scaling_disagreement(
    window: Window, factor: float, n: int, *, max_steps: int, tol: float
) -> tuple[float, bool]:
    r"""Share of grid points whose phase differs from that of their image under a scaling.

    Every point :math:`w` of the offset grid of ``window`` is compared with
    :math:`c+\text{factor}\,(w-c)`, where :math:`c=(\text{lo},\text{lo})` is the lower left
    corner of the window.

    Returns
    -------
    tuple
        The share and whether all points (and their images) converged.
    """
    grid_x, grid_y = offset_grid(window.lo, window.hi, n)
    corner = window.lo
    base = classify_phases(grid_x, grid_y, window.alpha, max_steps=max_steps, tol=tol)
    scaled = classify_phases(
        corner + factor * (grid_x - corner),
        corner + factor * (grid_y - corner),
        window.alpha,
        max_steps=max_steps,
        tol=tol,
    )
    converged = bool((base >= 0).all() and (scaled >= 0).all())
    return disagreement(base, scaled), converged


def colour_changes_along_ray(
    alpha: float,
    slope: float,
    *,
    u_min: float,
    u_max: float,
    samples: int,
    max_steps: int,
    tol: float,
) -> tuple[int, bool]:
    r"""Count the phase changes along a ray from the corner :math:`(\tfrac12,\tfrac12)`.

    The ray is :math:`v=su` with ``s = slope``, where :math:`x=\tfrac12+u`,
    :math:`y=\tfrac12+v`; it is sampled at ``samples`` points with :math:`u` geometrically
    spaced in ``[u_min, u_max]``.

    Returns
    -------
    tuple
        The number of changes between consecutive samples and whether all samples converged.
    """
    u = np.geomspace(u_min, u_max, samples)
    phases = classify_phases(0.5 + u, 0.5 + slope * u, alpha, max_steps=max_steps, tol=tol)
    return int(np.count_nonzero(phases[1:] != phases[:-1])), bool((phases >= 0).all())


@dataclass(frozen=True, slots=True)
class CornerComparison:
    """Comparison of a window ``[lo, 1]^2`` with a reference window at the corner (1, 1)."""

    window: Window
    reference: Window
    share: float
    """Share of grid points with different phases."""
    near_diagonal: float
    """Share of the differing points that lie in the sector around the diagonal."""


def corner_one_comparisons(
    reference: Window,
    windows: tuple[Window, ...],
    n: int,
    *,
    sector_deg: float,
    max_steps: int,
    tol: float,
) -> tuple[list[CornerComparison], float, bool]:
    """Compare windows ``[lo, 1]^2`` with a reference window on grids of the same shape.

    The grids have the same relative positions, so the comparison is one under scaling about
    the corner :math:`(1,1)`.  A grid point is in the sector around the diagonal if the ray
    from :math:`(1,1)` through it makes an angle less than ``sector_deg`` with the diagonal.

    Returns
    -------
    tuple
        The comparisons, the share of grid points in the sector, and whether all grid points
        converged.
    """
    for window in (reference, *windows):
        if window.hi != 1.0:
            raise ValueError(f"window {window.label} does not have the corner (1,1)")
    grid_x, grid_y = offset_grid(reference.lo, reference.hi, n)
    angle = np.degrees(np.arctan2(1.0 - grid_y, 1.0 - grid_x))  # 45 degrees: the diagonal
    in_sector = np.abs(angle - 45.0) < sector_deg
    base = window_phases(reference, n, max_steps=max_steps, tol=tol)
    converged = bool((base >= 0).all())
    out = []
    for window in windows:
        phases = window_phases(window, n, max_steps=max_steps, tol=tol)
        converged = converged and bool((phases >= 0).all())
        differ = phases != base
        near = float(np.mean(in_sector[differ])) if differ.any() else 0.0
        out.append(CornerComparison(window, reference, float(np.mean(differ)), near))
    return out, float(np.mean(in_sector)), converged
