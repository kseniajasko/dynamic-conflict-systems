r"""Phases of the asymmetric attractor: the point of it occupied at even times.

For :math:`0<\alpha<2` the asymmetric attractor is the exchange 2-cycle
:math:`\{(x^*,y^*),(y^*,x^*)\}`; the *phase* of an initial point is the point of the cycle
that its trajectory approaches at even times.  For :math:`\alpha>2` the two phases are the
basins of the two winner--loser fixed points.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from maxmap._types import FloatArray, FloatOrArray, IntArray
from maxmap.analysis import asymmetric_attractor
from maxmap.experiments.config import Window
from maxmap.model import phi

__all__ = [
    "CAPTURE_FRACTION",
    "NOT_CONVERGED",
    "PHASE_FIRST",
    "PHASE_SECOND",
    "classify_phases",
    "offset_grid",
    "window_phases",
]

PHASE_FIRST: Final = 0
r"""Phase of :math:`(x^*,y^*)` (light blue in Fig. 6)."""

PHASE_SECOND: Final = 1
r"""Phase of :math:`(y^*,x^*)` (light red in Fig. 6)."""

NOT_CONVERGED: Final = -1
"""Marker of initial points that do not reach the attractor within the step limit."""

CAPTURE_FRACTION: Final = 1e-6
r"""Capture radius of the early stop, relative to :math:`|x^*-y^*|`.

A point closer than this to a point of the attractor at an even step stays in the region in
which :math:`\Phi^2` is a smooth contraction, so its phase is final.  On all grids of the
paper the early stop gives the same phases as 4000 iterations.
"""


def offset_grid(lo: float, hi: float, n: int) -> tuple[FloatArray, FloatArray]:
    r"""Return the offset :math:`n\times(n-1)` grid of a window, which contains no diagonal points.

    The abscissae are :math:`lo+(hi-lo)i/(n+1)`, :math:`i=1,\dots,n`, and the ordinates
    :math:`lo+(hi-lo)j/n`, :math:`j=1,\dots,n-1`; equality :math:`i/(n+1)=j/n` is impossible
    for these indices since :math:`\gcd(n,n+1)=1`.

    Returns
    -------
    tuple of numpy.ndarray
        Coordinate arrays of shape ``(n - 1, n)`` as returned by :func:`numpy.meshgrid`.
    """
    gx = np.linspace(lo, hi, n + 2)[1:-1]
    gy = np.linspace(lo, hi, n + 1)[1:]
    gy = gy[gy < hi]
    grid_x, grid_y = np.meshgrid(gx, gy)
    return grid_x, grid_y


def _attractor_per_point(
    alpha: FloatArray, tol: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Attractor coordinates and capture radii for an array of parameter values."""
    u = np.empty_like(alpha)
    v = np.empty_like(alpha)
    cap = np.empty_like(alpha)
    for value in np.unique(alpha):
        mask = alpha == value
        x_star, y_star = asymmetric_attractor(float(value))
        u[mask], v[mask] = x_star, y_star
        cap[mask] = min(tol, CAPTURE_FRACTION * abs(x_star - y_star))
    return u, v, cap


def classify_phases(
    x0: FloatArray, y0: FloatArray, alpha: FloatOrArray, *, max_steps: int, tol: float
) -> IntArray:
    r"""Phase of the asymmetric attractor reached from each initial point.

    Parameters
    ----------
    x0, y0
        Initial points (arrays of a common shape).
    alpha
        Parameter, :math:`\alpha\neq2`; a scalar or an array of the shape of ``x0``.
    max_steps
        Even step limit.
    tol
        Tolerance (max-norm) of the final test at ``max_steps``, applied to points that have
        not been captured earlier; it is also an upper bound of the capture radius.

    Returns
    -------
    numpy.ndarray
        Integer array of the shape of ``x0`` with values :data:`PHASE_FIRST`,
        :data:`PHASE_SECOND` or :data:`NOT_CONVERGED`.

    Notes
    -----
    Only the points that have not been classified yet are iterated.  A point is classified
    at the first even step at which it lies within the capture radius
    ``min(tol, CAPTURE_FRACTION * |x* - y*|)`` of a point of the attractor; the points left
    at ``max_steps`` are classified with the tolerance ``tol``, as in the computations of the
    paper.
    Since :func:`maxmap.model.phi` acts elementwise, the trajectories are bit-identical to
    those of a full iteration.
    """
    if max_steps % 2:
        raise ValueError("max_steps must be even: phases are defined at even times")
    shape = np.shape(x0)
    x = np.array(x0, dtype=np.float64).ravel()
    y = np.array(y0, dtype=np.float64).ravel()
    scalar_alpha = np.ndim(alpha) == 0
    a_all = np.broadcast_to(np.asarray(alpha, dtype=np.float64), shape).ravel()
    u, v, cap = _attractor_per_point(a_all, tol)
    a: FloatOrArray = float(np.asarray(alpha)) if scalar_alpha else a_all.copy()
    phase = np.full(x.size, NOT_CONVERGED, dtype=np.int64)
    idx = np.arange(x.size)
    for step in range(1, max_steps + 1):
        x, y = phi(x, y, a)
        if step % 2 or step == max_steps:
            continue
        first = (np.abs(x - u) < cap) & (np.abs(y - v) < cap)
        second = (np.abs(x - v) < cap) & (np.abs(y - u) < cap)
        done = first | second
        if not done.any():
            continue
        phase[idx[first]] = PHASE_FIRST
        phase[idx[second]] = PHASE_SECOND
        keep = ~done
        x, y, idx, u, v, cap = x[keep], y[keep], idx[keep], u[keep], v[keep], cap[keep]
        if not scalar_alpha:
            a = a_all[idx]
        if idx.size == 0:
            return phase.reshape(shape)
    first = (np.abs(x - u) < tol) & (np.abs(y - v) < tol)
    second = (np.abs(x - v) < tol) & (np.abs(y - u) < tol)
    phase[idx[first]] = PHASE_FIRST
    phase[idx[second]] = PHASE_SECOND
    return phase.reshape(shape)


def window_phases(window: Window, n: int, *, max_steps: int, tol: float) -> IntArray:
    """Phases on the offset grid (:func:`offset_grid`) of a window."""
    grid_x, grid_y = offset_grid(window.lo, window.hi, n)
    return classify_phases(grid_x, grid_y, window.alpha, max_steps=max_steps, tol=tol)
