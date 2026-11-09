"""Data of the bifurcation diagrams (Figs. 2--5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maxmap._types import FloatArray
from maxmap.analysis import exchange_cycle, rho_squared, winner_loser_fixed_point, x_sym
from maxmap.model import coordinate_iterates

__all__ = [
    "ALPHAS_ABOVE",
    "ALPHAS_BELOW",
    "ALPHAS_COORDINATE",
    "COORDINATE_KEEP",
    "COORDINATE_TRANSIENT",
    "BifurcationBranches",
    "bifurcation_branches",
    "coordinate_diagram",
]

ALPHAS_BELOW: FloatArray = np.linspace(0.04, 1.999, 500)
"""Parameter values of the oscillatory branch in Figs. 2 and 3."""

ALPHAS_ABOVE: FloatArray = np.linspace(2.0005, 5.0, 500)
"""Parameter values of the symmetry-broken branch in Figs. 2 and 3."""

ALPHAS_COORDINATE: FloatArray = np.linspace(0.02, 5.0, 600)
"""Parameter values of the coordinate-wise diagrams (Figs. 4 and 5)."""

COORDINATE_TRANSIENT = 3000
"""Discarded iterations of the full system in Figs. 4 and 5."""

COORDINATE_KEEP = 8
"""Plotted iterations of the full system in Figs. 4 and 5."""


@dataclass(frozen=True, slots=True, eq=False)
class BifurcationBranches:
    """Closed-form branches of the bifurcation diagram."""

    alphas_below: FloatArray
    cycle: FloatArray
    """Shape ``(n, 2)``: :math:`(x^*, y^*)` of the exchange 2-cycle."""
    sym_below: FloatArray
    alphas_above: FloatArray
    fixed: FloatArray
    """Shape ``(m, 2)``: :math:`(x^*, y^*)` of the winner--loser fixed point."""
    sym_above: FloatArray
    rho2_above: FloatArray

    @property
    def cycle_modulus(self) -> FloatArray:
        """The modulus :math:`1-x^*y^*` along the oscillatory branch."""
        result: FloatArray = 1.0 - self.cycle[:, 0] * self.cycle[:, 1]
        return result


def bifurcation_branches(
    alphas_below: FloatArray = ALPHAS_BELOW, alphas_above: FloatArray = ALPHAS_ABOVE
) -> BifurcationBranches:
    """Evaluate the closed-form branches on the parameter grids of Figs. 2 and 3."""
    return BifurcationBranches(
        alphas_below=alphas_below,
        cycle=np.array([exchange_cycle(float(a)) for a in alphas_below]),
        sym_below=np.array([x_sym(float(a)) for a in alphas_below]),
        alphas_above=alphas_above,
        fixed=np.array([winner_loser_fixed_point(float(a)) for a in alphas_above]),
        sym_above=np.array([x_sym(float(a)) for a in alphas_above]),
        rho2_above=np.array([rho_squared(float(a)) for a in alphas_above]),
    )


def coordinate_diagram(
    p0: tuple[float, float],
    r0: tuple[float, float],
    alphas: FloatArray = ALPHAS_COORDINATE,
    *,
    n_transient: int = COORDINATE_TRANSIENT,
    n_keep: int = COORDINATE_KEEP,
) -> tuple[FloatArray, FloatArray]:
    """First coordinates of the full system after a transient, for each parameter value."""
    return coordinate_iterates(p0, r0, alphas, n_transient, n_keep)
