r"""The model: the map of maxima, its branches and switching curves, and the full system.

Notation follows the paper.  For :math:`\alpha>0` the map of maxima is

.. math::

    \Phi(x, y) = \left(\frac{\max\{\alpha x(1-y),\,1-x\}}{\alpha x(1-y)+1-x},\;
                       \frac{\max\{\alpha y(1-x),\,1-y\}}{\alpha y(1-x)+1-y}\right)

on :math:`[\tfrac12, 1)^2`.  A coordinate is *active* if the maximum is attained at the
first argument and *passive* otherwise.

All functions are vectorized over NumPy arrays.  The order of the floating-point operations
in :func:`phi` is fixed: it is the order used for the published results, so that every
number of the paper is reproduced bit for bit.  Since ``1 - x`` is computed exactly for
:math:`x\in[\tfrac12,1]` (Sterbenz lemma), :func:`phi` is accurate to a few units in the
last place on the whole square.
"""

from __future__ import annotations

import numpy as np

from maxmap._types import BoolArray, FloatArray, FloatOrArray

__all__ = [
    "branches",
    "coordinate_iterates",
    "corner_map",
    "curve_s",
    "eta",
    "full_step",
    "iterate",
    "jacobian",
    "logit",
    "orbit",
    "phi",
    "phi_point",
    "psi",
    "swap",
    "switching_curve_tp",
    "switching_curve_tr",
]


def phi(x: FloatArray, y: FloatArray, alpha: FloatOrArray) -> tuple[FloatArray, FloatArray]:
    r"""Evaluate the map of maxima :math:`\Phi` (equation (1) of the paper).

    Parameters
    ----------
    x, y
        Maxima of the two stochastic vectors, in :math:`[\tfrac12, 1)`; arrays of a common
        shape.
    alpha
        Parameter :math:`\alpha>0`; a scalar or an array broadcastable to ``x``.

    Returns
    -------
    tuple of numpy.ndarray
        The new maxima :math:`(x', y')`.

    Notes
    -----
    The map is exactly equivariant also in floating point: ``phi(y, x)`` is ``phi(x, y)``
    with the outputs interchanged, so that points of the diagonal stay on it.
    """
    pa = alpha * x * (1.0 - y)  # active numerator of the first coordinate
    pp = 1.0 - x  # passive numerator of the first coordinate
    ra = alpha * y * (1.0 - x)
    rp = 1.0 - y
    return np.maximum(pa, pp) / (pa + pp), np.maximum(ra, rp) / (ra + rp)


def phi_point(x: float, y: float, alpha: float) -> tuple[float, float]:
    """Evaluate :func:`phi` at a single point and return Python floats.

    The arithmetic is the same IEEE double-precision arithmetic as in :func:`phi`, so the
    result coincides bit for bit with the vectorized evaluation.
    """
    pa = alpha * x * (1.0 - y)
    pp = 1.0 - x
    ra = alpha * y * (1.0 - x)
    rp = 1.0 - y
    return max(pa, pp) / (pa + pp), max(ra, rp) / (ra + rp)


def iterate(
    x: FloatArray, y: FloatArray, alpha: FloatOrArray, steps: int
) -> tuple[FloatArray, FloatArray]:
    """Apply :func:`phi` ``steps`` times to arrays of initial points."""
    for _ in range(steps):
        x, y = phi(x, y, alpha)
    return x, y


def orbit(x0: float, y0: float, alpha: float, steps: int) -> FloatArray:
    """Return the orbit ``[(x^0, y^0), ..., (x^steps, y^steps)]`` of one point.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(steps + 1, 2)``.
    """
    out = np.empty((steps + 1, 2))
    x, y = x0, y0
    out[0] = x, y
    for t in range(1, steps + 1):
        x, y = phi_point(x, y, alpha)
        out[t] = x, y
    return out


def swap(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    r"""Apply the permutation :math:`\sigma(x, y) = (y, x)`."""
    return y, x


def branches(x: FloatArray, y: FloatArray, alpha: FloatOrArray) -> tuple[BoolArray, BoolArray]:
    r"""Masks of the active branches of the two coordinates of :math:`\Phi`.

    Returns
    -------
    tuple of numpy.ndarray
        ``(active_p, active_r)`` with ``active_p`` true where
        :math:`\alpha x(1-y) > 1-x` and ``active_r`` true where :math:`\alpha y(1-x) > 1-y`.
        On a switching curve the corresponding mask is false.
    """
    active_p = alpha * x * (1.0 - y) > 1.0 - x
    active_r = alpha * y * (1.0 - x) > 1.0 - y
    return active_p, active_r


def jacobian(x: FloatArray, y: FloatArray, alpha: FloatOrArray) -> FloatArray:
    r"""Jacobian matrix of :math:`\Phi` on the branch that contains each point.

    On the active branch of the first coordinate
    :math:`\partial x'/\partial x = \alpha(1-y)/z_p^2` and
    :math:`\partial x'/\partial y = -\alpha x(1-x)/z_p^2`; the passive branch equals one minus
    the active one, so its derivatives have the opposite signs.  The second coordinate is
    obtained by the symmetry :math:`x\leftrightarrow y`.

    Returns
    -------
    numpy.ndarray
        Array of shape ``x.shape + (2, 2)``.
    """
    zp = alpha * x * (1.0 - y) + (1.0 - x)
    zr = alpha * y * (1.0 - x) + (1.0 - y)
    active_p, active_r = branches(x, y, alpha)
    sign_p = np.where(active_p, 1.0, -1.0)
    sign_r = np.where(active_r, 1.0, -1.0)
    jac = np.empty((*np.shape(x), 2, 2))
    jac[..., 0, 0] = sign_p * alpha * (1.0 - y) / zp**2
    jac[..., 0, 1] = -sign_p * alpha * x * (1.0 - x) / zp**2
    jac[..., 1, 0] = -sign_r * alpha * y * (1.0 - y) / zr**2
    jac[..., 1, 1] = sign_r * alpha * (1.0 - x) / zr**2
    return jac


def switching_curve_tp(x: FloatArray, alpha: float) -> FloatArray:
    r"""Return the curve :math:`T_p=\{\alpha x(1-y)=1-x\}` as a graph :math:`y(x)`.

    The first coordinate of :math:`\Phi` is active below this graph.
    """
    result: FloatArray = 1.0 - (1.0 - x) / (alpha * x)
    return result


def switching_curve_tr(x: FloatArray, alpha: float) -> FloatArray:
    r"""Return the curve :math:`T_r=\{\alpha y(1-x)=1-y\}` as a graph :math:`y(x)`.

    The second coordinate of :math:`\Phi` is active above this graph.
    """
    result: FloatArray = 1.0 / (1.0 + alpha * (1.0 - x))
    return result


def curve_s(x: FloatArray, alpha: float) -> FloatArray:
    r"""Return the curve :math:`S=\{\alpha^2 xy=1\}` as a graph :math:`y(x)`."""
    result: FloatArray = 1.0 / (alpha * alpha * x)
    return result


def logit(s: FloatArray) -> FloatArray:
    r"""Return the logarithmic coordinate :math:`\ell(s)=\ln\frac{s}{1-s}` of a maximum."""
    result: FloatArray = np.log(s / (1.0 - s))
    return result


def eta(u: FloatArray, v: FloatArray, alpha: float) -> FloatArray:
    r"""Evaluate :math:`\eta(u,v)=\ln\alpha+u-\ln(1+e^{v})` (Lemma 3 of the paper)."""
    result: FloatArray = np.log(alpha) + u - np.logaddexp(0.0, v)
    return result


def psi(u: FloatArray, v: FloatArray, alpha: float) -> tuple[FloatArray, FloatArray]:
    r"""Evaluate :math:`\Psi(u,v)=(|\eta(u,v)|,|\eta(v,u)|)`, conjugate to :math:`\Phi`."""
    return np.abs(eta(u, v, alpha)), np.abs(eta(v, u, alpha))


def corner_map(u: FloatArray, v: FloatArray) -> tuple[FloatArray, FloatArray]:
    r"""Evaluate the parameter-free rescaled map near the corner :math:`(\tfrac12,\tfrac12)`.

    In the variables :math:`U = 8u/(2-\alpha)`, :math:`V = 8v/(2-\alpha)` the leading-order
    part of :math:`\Phi` is :math:`(U,V)\mapsto(|U-V/2-1|,\,|V-U/2-1|)`.
    """
    return np.abs(u - 0.5 * v - 1.0), np.abs(v - 0.5 * u - 1.0)


def full_step(p: FloatArray, r: FloatArray, alpha: FloatOrArray) -> tuple[FloatArray, FloatArray]:
    r"""One step of the full system (2) for stochastic vectors on two positions.

    Parameters
    ----------
    p, r
        Arrays of shape ``(..., 2)``: the two stochastic vectors (vectorized over the
        leading axes).
    alpha
        Parameter; a scalar or an array of shape ``p.shape[:-1]``.

    Returns
    -------
    tuple of numpy.ndarray
        The vectors at the next step.

    Notes
    -----
    The dominant position is the first index of the maximal coordinate (as
    :func:`numpy.argmax`).  The normalizing denominators are computed, as in the definition
    of the law, as the sums of the numerators; the new vectors are therefore stochastic up
    to one rounding and need no renormalization.  (If the denominators are computed from
    the explicit form :math:`z_p=\alpha x(1-y)+1-x`, which presupposes :math:`p_1+p_2=1`
    exactly, the deviation of :math:`p_1+p_2` from one is multiplied by
    :math:`x^*/(1-x^*)>1` over two steps near the 4-cycle and rounding errors grow.)
    """
    first_p = p[..., 0] >= p[..., 1]  # ties -> first position, as numpy.argmax
    first_r = r[..., 0] >= r[..., 1]
    p_max = np.where(first_p, p[..., 0], p[..., 1])
    p_other = np.where(first_p, p[..., 1], p[..., 0])
    r_max = np.where(first_r, r[..., 0], r[..., 1])
    r_other = np.where(first_r, r[..., 1], r[..., 0])
    num_p = alpha * p_max * (1.0 - r_max)  # active interaction at the dominant position
    num_r = alpha * r_max * (1.0 - p_max)
    zp = num_p + p_other
    zr = num_r + r_other
    new_p_max, new_p_other = num_p / zp, p_other / zp
    new_r_max, new_r_other = num_r / zr, r_other / zr
    p_new = np.stack(
        [np.where(first_p, new_p_max, new_p_other), np.where(first_p, new_p_other, new_p_max)],
        axis=-1,
    )
    r_new = np.stack(
        [np.where(first_r, new_r_max, new_r_other), np.where(first_r, new_r_other, new_r_max)],
        axis=-1,
    )
    return p_new, r_new


def coordinate_iterates(
    p0: tuple[float, float],
    r0: tuple[float, float],
    alphas: FloatArray,
    n_transient: int,
    n_keep: int,
) -> tuple[FloatArray, FloatArray]:
    """Iterate the full system from one initial state for many values of the parameter.

    Parameters
    ----------
    p0, r0
        Initial stochastic vectors.
    alphas
        One-dimensional array of parameter values.
    n_transient
        Number of discarded iterations.
    n_keep
        Number of subsequent iterations whose first coordinates are returned.

    Returns
    -------
    tuple of numpy.ndarray
        Arrays ``(p_1, r_1)`` of shape ``(len(alphas), n_keep)``.
    """
    p = np.broadcast_to(np.asarray(p0, dtype=np.float64), (alphas.size, 2)).copy()
    r = np.broadcast_to(np.asarray(r0, dtype=np.float64), (alphas.size, 2)).copy()
    for _ in range(n_transient):
        p, r = full_step(p, r, alphas)
    kept_p = np.empty((alphas.size, n_keep))
    kept_r = np.empty((alphas.size, n_keep))
    for k in range(n_keep):
        p, r = full_step(p, r, alphas)
        kept_p[:, k] = p[:, 0]
        kept_r[:, k] = r[:, 0]
    return kept_p, kept_r
