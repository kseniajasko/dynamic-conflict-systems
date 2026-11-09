r"""Numerical experiments and figures for the map of maxima.

The package accompanies the paper *From oscillations to symmetry breaking in a
two-dimensional piecewise-smooth dynamical system*.  It contains

* :mod:`maxmap.model` -- the map of maxima :math:`\Phi`, its branches, switching curves and
  logarithmic form, and the full system of two stochastic vectors;
* :mod:`maxmap.analysis` -- closed forms: symmetric state, exchange 2-cycle, winner--loser
  fixed point, Jacobians and multipliers;
* :mod:`maxmap.experiments` -- the checks of Section 5;
* :mod:`maxmap.plotting` -- one function per figure of the paper;
* :mod:`maxmap.cli` -- the command line interface ``maxmap`` (``python -m maxmap``).

Importing the package has no side effects (in particular, no Matplotlib state is changed).
"""

from maxmap.analysis import (
    asymmetric_attractor,
    exchange_cycle,
    winner_loser_fixed_point,
    x_sym,
)
from maxmap.model import full_step, phi

__all__ = [
    "__version__",
    "asymmetric_attractor",
    "exchange_cycle",
    "full_step",
    "phi",
    "winner_loser_fixed_point",
    "x_sym",
]

__version__ = "2.1.0"
