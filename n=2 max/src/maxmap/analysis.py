r"""Closed-form objects of the paper: symmetric state, exchange 2-cycle, fixed points, multipliers.

References to numbered results are to the paper: Proposition 1 (symmetric 2-cycles),
Theorem 1 (exchange 2-cycle), Theorem 3 (winner--loser fixed point), Proposition 3 (saddle),
Remarks 2 and 3 (constants at :math:`\alpha=1` and the limit :math:`\alpha\to2`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from maxmap._types import FloatArray
from maxmap.model import jacobian

__all__ = [
    "ALPHA_CRITICAL",
    "CORNER_MAP_CYCLE",
    "GOLDEN_RATIO",
    "PLASTIC_NUMBER",
    "Multipliers",
    "asymmetric_attractor",
    "collapse_factor",
    "collapse_s_map",
    "cycle_cubic",
    "cycle_modulus",
    "cycle_monodromy",
    "cycle_multipliers",
    "exchange_cycle",
    "exchange_cycle_near_critical",
    "fixed_point_jacobian",
    "rho",
    "rho_squared",
    "saddle_multipliers",
    "two_step_contraction",
    "winner_loser_fixed_point",
    "x_sym",
]

ALPHA_CRITICAL: float = 2.0
"""The bifurcation value of the parameter."""

GOLDEN_RATIO: float = (1.0 + math.sqrt(5.0)) / 2.0
"""The golden ratio; ``x_sym(1) == 1 / GOLDEN_RATIO``."""

PLASTIC_NUMBER: float = ((9.0 + math.sqrt(69.0)) / 18.0) ** (1.0 / 3.0) + (
    (9.0 - math.sqrt(69.0)) / 18.0
) ** (1.0 / 3.0)
"""The plastic number, the real root of :math:`t^3=t+1` (Cardano's formula)."""

CORNER_MAP_CYCLE: tuple[float, float] = (10.0 / 7.0, 2.0 / 7.0)
"""Point of the exchange 2-cycle of the rescaled corner map :func:`maxmap.model.corner_map`."""

_CUBIC_IMAG_TOL = 1e-12


def _check_alpha(alpha: float, *, below: bool | None = None) -> None:
    if not alpha > 0.0:
        raise ValueError(f"alpha must be positive, got {alpha!r}")
    if below is True and not alpha < ALPHA_CRITICAL:
        raise ValueError(f"expected 0 < alpha < 2, got {alpha!r}")
    if below is False and not alpha > ALPHA_CRITICAL:
        raise ValueError(f"expected alpha > 2, got {alpha!r}")


def x_sym(alpha: float) -> float:
    r"""Value :math:`x_{\mathrm{sym}}(\alpha)` of the symmetric fixed point of :math:`\Phi`.

    It is the root of :math:`\alpha x^2+x-1=0` in :math:`(\tfrac12,1)` for
    :math:`0<\alpha<2`, equals :math:`1-1/\alpha` for :math:`\alpha>2` and is extended by
    :math:`x_{\mathrm{sym}}(2)=\tfrac12` (Lemma 6).
    """
    _check_alpha(alpha)
    if alpha < ALPHA_CRITICAL:
        return float((-1.0 + np.sqrt(1.0 + 4.0 * alpha)) / (2.0 * alpha))
    return 1.0 - 1.0 / alpha


def cycle_cubic(x: float, alpha: float) -> float:
    r"""Evaluate the cubic :math:`P(x)=\alpha^3x^3+\alpha x^2+(1-\alpha)x-1` of Theorem 1."""
    return alpha**3 * x**3 + alpha * x**2 + (1.0 - alpha) * x - 1.0


def exchange_cycle(alpha: float) -> tuple[float, float]:
    r"""Point :math:`(x^*,y^*)` of the exchange 2-cycle for :math:`0<\alpha<2` (Theorem 1).

    :math:`x^*` is the unique root of the cubic :func:`cycle_cubic` in :math:`(\tfrac12,1)`
    and :math:`y^*=1/(1+\alpha x^*)`.  The root is taken from the companion-matrix
    eigenvalues (:func:`numpy.roots`), as for the published results; its accuracy (a few
    units in the last place) is checked against high-precision values in the test suite.

    Raises
    ------
    ValueError
        If ``alpha`` is not in :math:`(0, 2)`.
    """
    _check_alpha(alpha, below=True)
    roots = np.roots([alpha**3, alpha, 1.0 - alpha, -1.0])
    real = [
        float(root.real)
        for root in roots
        if abs(root.imag) < _CUBIC_IMAG_TOL and 0.5 < root.real < 1.0
    ]
    if len(real) != 1:  # pragma: no cover - excluded by Theorem 1
        raise ArithmeticError(f"expected one root of the cubic in (1/2, 1), got {real}")
    x_star = real[0]
    return x_star, 1.0 / (1.0 + alpha * x_star)


def exchange_cycle_near_critical(alpha: float) -> tuple[float, float]:
    r"""First-order expansion of :math:`(x^*,y^*)` as :math:`\alpha\to2^-` (Remark 3).

    :math:`x^*=\tfrac12+\tfrac5{28}(2-\alpha)+O((2-\alpha)^2)` and
    :math:`y^*=\tfrac12+\tfrac1{28}(2-\alpha)+O((2-\alpha)^2)`.
    """
    delta = ALPHA_CRITICAL - alpha
    return 0.5 + 5.0 * delta / 28.0, 0.5 + delta / 28.0


def winner_loser_fixed_point(alpha: float) -> tuple[float, float]:
    r"""Return the winner--loser fixed point :math:`(x^*,y^*)`, :math:`\alpha>2` (Theorem 3)."""
    _check_alpha(alpha, below=False)
    return 1.0 - 1.0 / (alpha * (alpha - 1.0) ** 2), 1.0 - 1.0 / alpha


def asymmetric_attractor(alpha: float) -> tuple[float, float]:
    r"""Return the point :math:`(x^*,y^*)` of the asymmetric attractor for :math:`\alpha\neq2`.

    The attractor is the exchange 2-cycle :math:`\{(x^*,y^*),(y^*,x^*)\}` for
    :math:`\alpha<2` and the pair of fixed points :math:`(x^*,y^*)`, :math:`(y^*,x^*)` for
    :math:`\alpha>2`.

    Raises
    ------
    ValueError
        If ``alpha == 2``: the asymmetric attractor does not exist there.
    """
    if alpha == ALPHA_CRITICAL:
        raise ValueError("there is no asymmetric attractor at alpha = 2")
    return exchange_cycle(alpha) if alpha < ALPHA_CRITICAL else winner_loser_fixed_point(alpha)


def cycle_modulus(alpha: float) -> float:
    r"""Return the common modulus :math:`1-x^*y^*` of the multipliers of the exchange 2-cycle."""
    x_star, y_star = exchange_cycle(alpha)
    return 1.0 - x_star * y_star


def rho_squared(alpha: float) -> float:
    r""":math:`\rho^2(\alpha)=(\alpha^2-\alpha+1)/(\alpha^2(\alpha-1))` of Theorem 3."""
    return (alpha * alpha - alpha + 1.0) / (alpha * alpha * (alpha - 1.0))


def rho(alpha: float) -> float:
    r"""Modulus :math:`\rho(\alpha)` of the eigenvalues at the winner--loser fixed point."""
    return math.sqrt(rho_squared(alpha))


def two_step_contraction(alpha: float) -> float:
    r"""Asymptotic contraction over two steps of the asymmetric attractor (Fig. 3(b))."""
    return cycle_modulus(alpha) if alpha < ALPHA_CRITICAL else rho_squared(alpha)


@dataclass(frozen=True, slots=True)
class Multipliers:
    """Eigenvalues of a monodromy (or Jacobian) matrix."""

    values: tuple[complex, complex]

    @property
    def modulus(self) -> float:
        """Largest modulus."""
        return max(abs(v) for v in self.values)

    @property
    def argument_degrees(self) -> float:
        """Argument of the eigenvalue with nonnegative imaginary part, in degrees."""
        upper = max(self.values, key=lambda v: v.imag)
        return math.degrees(math.atan2(upper.imag, upper.real))


def cycle_monodromy(alpha: float) -> FloatArray:
    r"""Monodromy matrix :math:`M=J_\Phi(y^*,x^*)J_\Phi(x^*,y^*)` of the exchange 2-cycle."""
    x_star, y_star = exchange_cycle(alpha)
    jac_a = jacobian(np.array(x_star), np.array(y_star), alpha)
    jac_b = jacobian(np.array(y_star), np.array(x_star), alpha)
    result: FloatArray = jac_b @ jac_a
    return result


def cycle_multipliers(alpha: float) -> Multipliers:
    """Multipliers of the exchange 2-cycle (computed from the monodromy matrix)."""
    eig = np.linalg.eigvals(cycle_monodromy(alpha))
    return Multipliers((complex(eig[0]), complex(eig[1])))


def fixed_point_jacobian(alpha: float) -> FloatArray:
    r"""Jacobian of :math:`\Phi` at the winner--loser fixed point (Theorem 3)."""
    x_star, y_star = winner_loser_fixed_point(alpha)
    result: FloatArray = jacobian(np.array(x_star), np.array(y_star), alpha)
    return result


def saddle_multipliers(alpha: float) -> tuple[float, float]:
    r"""Eigenvalues :math:`(\lambda_\parallel,\lambda_\perp)` of the symmetric saddle (Prop. 3).

    :math:`\lambda_\parallel=s-1`, :math:`\lambda_\perp=-(1+s)` for :math:`0<\alpha<2` and
    :math:`\lambda_\parallel=1-s`, :math:`\lambda_\perp=1+s` for :math:`\alpha>2`, where
    :math:`s=x_{\mathrm{sym}}(\alpha)`.
    """
    if alpha == ALPHA_CRITICAL:
        raise ValueError("the symmetric state is not a hyperbolic fixed point at alpha = 2")
    s = x_sym(alpha)
    if alpha < ALPHA_CRITICAL:
        return s - 1.0, -(1.0 + s)
    return 1.0 - s, 1.0 + s


def collapse_s_map(s: float) -> float:
    r"""Leading-order map :math:`f(s)=|s-\tfrac12|/(1-\tfrac s2)` of the ratio :math:`s=v/u`.

    At :math:`\alpha=2` and to leading order near :math:`(\tfrac12,\tfrac12)`, the ratio of
    the deviations of the smaller and the larger maximum from :math:`\tfrac12` evolves by this
    map (Section 5.2).
    """
    return abs(s - 0.5) / (1.0 - 0.5 * s)


def collapse_factor(s: float) -> float:
    r"""Return the one-step factor :math:`1-s/2` of :math:`G_t-\tfrac12` at :math:`\alpha=2`."""
    return 1.0 - 0.5 * s
