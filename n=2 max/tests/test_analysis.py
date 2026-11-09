"""Closed forms of Sections 3 and 4 (and the constants of Remark 2)."""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import pairwise

import numpy as np
import pytest

from conftest import cycle_root_decimal
from maxmap.analysis import (
    CORNER_MAP_CYCLE,
    GOLDEN_RATIO,
    PLASTIC_NUMBER,
    collapse_factor,
    collapse_s_map,
    cycle_cubic,
    cycle_modulus,
    cycle_monodromy,
    cycle_multipliers,
    exchange_cycle,
    exchange_cycle_near_critical,
    fixed_point_jacobian,
    rho,
    rho_squared,
    saddle_multipliers,
    winner_loser_fixed_point,
    x_sym,
)
from maxmap.model import corner_map, jacobian

ALPHAS_BELOW = (0.04, 0.2, 0.5, 1.0, 1.5, 1.9, 1.99, 1.999)
ALPHAS_ABOVE = (2.0005, 2.01, 2.5, 3.0, 5.0, 10.0)


def test_golden_ratio_at_alpha_one() -> None:
    assert x_sym(1.0) == pytest.approx(1.0 / GOLDEN_RATIO, rel=1e-15)
    assert saddle_multipliers(1.0)[1] == pytest.approx(-GOLDEN_RATIO, rel=1e-15)


def test_plastic_number_at_alpha_one() -> None:
    psi = PLASTIC_NUMBER
    assert psi**3 == pytest.approx(psi + 1.0, rel=1e-15)
    x_star, y_star = exchange_cycle(1.0)
    assert x_star == pytest.approx(1.0 / psi, rel=1e-15)
    assert y_star == pytest.approx(psi**-2, rel=1e-15)
    assert y_star == pytest.approx(x_star**2, rel=1e-15)
    assert cycle_modulus(1.0) == pytest.approx(y_star, rel=1e-14)  # Remark 2


@pytest.mark.parametrize("alpha", ["0.04", "0.2", "1", "1.5", "1.9", "1.99", "1.9999"])
def test_cycle_root_is_accurate(alpha: str) -> None:
    exact = float(cycle_root_decimal(alpha))
    # numpy.roots (companion matrix) is accurate to a few units in the last place
    assert abs(exchange_cycle(float(alpha))[0] - exact) <= 8 * np.spacing(exact)


@pytest.mark.parametrize("alpha", ALPHAS_BELOW)
def test_exchange_cycle_identities_and_regimes(alpha: float) -> None:
    x, y = exchange_cycle(alpha)
    assert 0.5 < y < x < 1.0
    assert cycle_cubic(x, alpha) == pytest.approx(0.0, abs=1e-14)
    assert 1.0 - y == pytest.approx(alpha * x * y, rel=1e-13)
    assert 1.0 - x == pytest.approx(alpha**3 * x**3 * y, rel=1e-12)
    assert alpha * x < 1.0
    assert alpha * x * (1.0 - y) > 1.0 - x  # first coordinate active
    assert alpha * y * (1.0 - x) < 1.0 - y  # second coordinate passive


@pytest.mark.parametrize("alpha", ALPHAS_BELOW)
def test_cycle_multipliers_are_a_stable_focus(alpha: float) -> None:
    x, y = exchange_cycle(alpha)
    mono = cycle_monodromy(alpha)
    assert np.trace(mono) == pytest.approx(x * x + y * y - 2.0, abs=1e-12)
    assert np.linalg.det(mono) == pytest.approx((1.0 - x * y) ** 2, abs=1e-12)
    mult = cycle_multipliers(alpha)
    assert all(abs(m.imag) > 0 for m in mult.values)
    assert mult.modulus == pytest.approx(1.0 - x * y, rel=1e-9)
    assert mult.modulus < 0.75


def test_multiplier_argument_at_alpha_one_and_a_half() -> None:
    assert cycle_multipliers(1.5).argument_degrees == pytest.approx(174.2786, abs=1e-3)


def test_contraction_tends_to_three_quarters_from_both_sides() -> None:
    assert cycle_modulus(2.0 - 1e-7) == pytest.approx(0.75, abs=1e-6)
    assert rho_squared(2.0 + 1e-9) == pytest.approx(0.75, abs=1e-8)
    assert rho(2.0 + 1e-12) == pytest.approx(math.sqrt(3.0) / 2.0, abs=1e-10)


def test_rho_is_decreasing() -> None:
    values = [rho(a) for a in np.linspace(2.0001, 50.0, 400)]
    assert all(b < a for a, b in pairwise(values))
    assert rho(1e6) < 1e-2


@pytest.mark.parametrize("alpha", ALPHAS_ABOVE)
def test_winner_loser_fixed_point(alpha: float) -> None:
    x, y = winner_loser_fixed_point(alpha)
    assert 0.5 < y < x < 1.0
    jac = fixed_point_jacobian(alpha)
    assert np.trace(jac) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(jac @ jac, rho_squared(alpha) * np.eye(2), atol=1e-12)
    eig = np.sort(np.linalg.eigvals(jac).real)
    np.testing.assert_allclose(eig, [-rho(alpha), rho(alpha)], atol=1e-12)


@pytest.mark.parametrize("alpha", ALPHAS_BELOW + ALPHAS_ABOVE)
def test_saddle_eigenvectors(alpha: float) -> None:
    s = x_sym(alpha)
    jac = jacobian(np.array(s), np.array(s), alpha)
    lam_par, lam_perp = saddle_multipliers(alpha)
    np.testing.assert_allclose(jac @ [1.0, 1.0], lam_par * np.array([1.0, 1.0]), atol=1e-12)
    np.testing.assert_allclose(jac @ [1.0, -1.0], lam_perp * np.array([1.0, -1.0]), atol=1e-12)
    assert abs(lam_par) < 1.0
    assert abs(lam_perp) > 1.5


def test_x_sym_is_v_shaped_and_continuous() -> None:
    below = [x_sym(a) for a in np.linspace(0.01, 1.9999, 300)]
    above = [x_sym(a) for a in np.linspace(2.0001, 20.0, 300)]
    assert all(b < a for a, b in pairwise(below))
    assert all(b > a for a, b in pairwise(above))
    assert x_sym(2.0) == 0.5
    assert x_sym(2.0 - 1e-9) == pytest.approx(0.5, abs=1e-9)


def test_expansion_near_critical_value() -> None:
    for delta in (1e-3, 1e-4, 1e-5):
        x, y = exchange_cycle(2.0 - delta)
        x1, y1 = exchange_cycle_near_critical(2.0 - delta)
        assert abs(x - x1) < 2 * delta**2
        assert abs(y - y1) < 2 * delta**2
        assert (x - y) / delta == pytest.approx(1.0 / 7.0, abs=2 * delta)


def test_corner_map_cycle_exactly() -> None:
    u, v = (Fraction(10, 7), Fraction(2, 7))
    assert (abs(u - v / 2 - 1), abs(v - u / 2 - 1)) == (v, u)
    image = corner_map(np.array(CORNER_MAP_CYCLE[0]), np.array(CORNER_MAP_CYCLE[1]))
    np.testing.assert_allclose(image, CORNER_MAP_CYCLE[::-1], rtol=1e-15)


def test_collapse_map_is_an_involution_with_constant_product() -> None:
    for s in np.linspace(0.0, 0.5, 101):
        assert collapse_s_map(collapse_s_map(s)) == pytest.approx(s, abs=1e-15)
        product = collapse_factor(s) * collapse_factor(collapse_s_map(s))
        assert product == pytest.approx(0.75, abs=1e-15)
    for s in np.linspace(0.51, 0.99, 49):
        assert collapse_s_map(s) < s


@pytest.mark.parametrize(
    ("func", "alpha"),
    [(exchange_cycle, 2.0), (exchange_cycle, 2.5), (winner_loser_fixed_point, 1.5), (x_sym, -1.0)],
)
def test_invalid_parameters_raise(func: object, alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        func(alpha)  # type: ignore[operator]
