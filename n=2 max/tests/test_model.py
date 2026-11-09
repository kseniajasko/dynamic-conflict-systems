"""Properties of the map of maxima and of the full system (Sections 2 and 3)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from maxmap.model import (
    branches,
    coordinate_iterates,
    full_step,
    jacobian,
    logit,
    phi,
    phi_point,
    psi,
)

maxima = st.floats(min_value=0.5, max_value=1.0, exclude_max=True, allow_nan=False)
open_maxima = st.floats(min_value=0.5, max_value=1.0, exclude_min=True, exclude_max=True)
alphas = st.floats(min_value=0.05, max_value=20.0, allow_nan=False)


def _phi1(x: float, y: float, alpha: float) -> tuple[float, float]:
    xn, yn = phi(np.array([x]), np.array([y]), alpha)
    return float(xn[0]), float(yn[0])


@given(maxima, maxima, alphas)
def test_phi_maps_the_square_into_itself(x: float, y: float, alpha: float) -> None:
    assume(max(x, y) < 1.0 - 1e-12)
    xn, yn = _phi1(x, y, alpha)
    assert 0.5 <= xn < 1.0
    assert 0.5 <= yn < 1.0


@given(maxima, maxima, alphas)
def test_exchange_symmetry_is_exact(x: float, y: float, alpha: float) -> None:
    xn, yn = _phi1(x, y, alpha)
    assert _phi1(y, x, alpha) == (yn, xn)


@given(maxima, maxima, alphas)
def test_scalar_and_vector_evaluations_coincide(x: float, y: float, alpha: float) -> None:
    assert phi_point(x, y, alpha) == _phi1(x, y, alpha)


@given(open_maxima, open_maxima, alphas)
def test_conjugacy_with_the_logarithmic_form(x: float, y: float, alpha: float) -> None:
    assume(max(x, y) < 1.0 - 1e-9)
    xn, yn = _phi1(x, y, alpha)
    u, v = psi(np.array(logit(np.array(x))), np.array(logit(np.array(y))), alpha)
    assert float(logit(np.array(xn))) == pytest.approx(float(u), rel=1e-8, abs=1e-8)
    assert float(logit(np.array(yn))) == pytest.approx(float(v), rel=1e-8, abs=1e-8)


@given(open_maxima, open_maxima, alphas)
def test_lemma2_bounds(x: float, y: float, alpha: float) -> None:
    assume(max(x, y) < 1.0 - 1e-9)
    c_alpha = 2.0 * abs(math.log(alpha)) + 2.0 * math.log(2.0)
    x1, y1 = _phi1(x, y, alpha)
    lx1, ly1 = float(logit(np.array(x1))), float(logit(np.array(y1)))
    assert abs(lx1 - ly1) <= c_alpha + 1e-9
    x2, y2 = _phi1(x1, y1, alpha)
    lx2, ly2 = float(logit(np.array(x2))), float(logit(np.array(y2)))
    assert lx2 + ly2 <= 2.0 * c_alpha + 1e-9  # the compact set K_alpha after two steps


@given(open_maxima, open_maxima, st.floats(min_value=2.0, max_value=20.0))
def test_order_is_preserved_for_alpha_at_least_two(x: float, y: float, alpha: float) -> None:
    # Away from the side x = 1, where doubles cannot resolve 1 - x (all images round to
    # 1 - 2**-52), and from ties at the level of rounding errors.
    assume(x > y + 1e-12)
    assume(x < 1.0 - 1e-9)
    xn, yn = _phi1(x, y, alpha)
    assert xn > yn
    assert branches(np.array(x), np.array(y), alpha)[0]


@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 1.9, 2.0, 2.5, 5.0])
def test_jacobian_matches_finite_differences_and_determinant(alpha: float) -> None:
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0.52, 0.98, (2, 200))
    h = 1e-7
    jac = jacobian(x, y, alpha)
    for col, (dx, dy) in enumerate(((h, 0.0), (0.0, h))):
        plus = np.stack(phi(x + dx, y + dy, alpha), axis=-1)
        minus = np.stack(phi(x - dx, y - dy, alpha), axis=-1)
        fd = (plus - minus) / (2 * h)
        same_branch = np.all(
            [
                np.equal(*[branches(x + s * dx, y + s * dy, alpha)[k] for s in (1, -1)])
                for k in (0, 1)
            ],
            axis=0,
        )
        np.testing.assert_allclose(fd[same_branch], jac[same_branch][:, :, col], atol=1e-6)
    zp = alpha * x * (1 - y) + 1 - x
    zr = alpha * y * (1 - x) + 1 - y
    det_abs = alpha**2 * (1 - x) * (1 - y) * (1 - x * y) / (zp**2 * zr**2)
    np.testing.assert_allclose(np.abs(np.linalg.det(jac)), det_abs, rtol=1e-10)


simplex_points = st.floats(min_value=0.01, max_value=0.99).filter(lambda t: abs(t - 0.5) > 1e-6)


@given(simplex_points, simplex_points, alphas)
def test_full_system_closes_on_the_map_of_maxima(p1: float, r1: float, alpha: float) -> None:
    p = np.array([p1, 1.0 - p1])
    r = np.array([r1, 1.0 - r1])
    p_new, r_new = full_step(p, r, alpha)
    two_ulp = 2 * np.spacing(1.0)
    assert abs(p_new.sum() - 1.0) <= two_ulp
    assert abs(r_new.sum() - 1.0) <= two_ulp
    xn, yn = _phi1(float(np.max(p)), float(np.max(r)), alpha)
    assert float(np.max(p_new)) == pytest.approx(xn, abs=1e-15)
    assert float(np.max(r_new)) == pytest.approx(yn, abs=1e-15)


def test_full_system_stays_stochastic_without_renormalization() -> None:
    alphas_grid = np.linspace(0.02, 5.0, 300)
    p = np.tile([0.81, 0.19], (alphas_grid.size, 1))
    r = np.tile([0.11, 0.89], (alphas_grid.size, 1))
    for _ in range(3000):
        p, r = full_step(p, r, alphas_grid)
    two_ulp = 2 * np.spacing(1.0)
    assert np.max(np.abs(p.sum(axis=1) - 1.0)) <= two_ulp
    assert np.max(np.abs(r.sum(axis=1) - 1.0)) <= two_ulp


def test_symmetric_and_mirror_states_are_invariant() -> None:
    alphas_grid = np.linspace(0.02, 5.0, 101)
    p_sym, r_sym = coordinate_iterates((0.81, 0.19), (0.81, 0.19), alphas_grid, 200, 8)
    assert np.array_equal(p_sym, r_sym)
    p_mir, r_mir = coordinate_iterates((0.81, 0.19), (0.19, 0.81), alphas_grid, 200, 8)
    np.testing.assert_allclose(r_mir, 1.0 - p_mir, atol=1e-15)


def test_coordinate_diagram_shows_the_four_cycle_and_the_winner() -> None:
    p, r = coordinate_iterates((0.81, 0.19), (0.11, 0.89), np.array([1.5, 3.0]), 3000, 8)
    assert len(np.unique(np.round(p[0], 12))) == 4  # alpha = 1.5: four values
    assert len(np.unique(np.round(r[1], 12))) == 1  # alpha = 3: r is the winner
    assert len(np.unique(np.round(p[1], 12))) == 2  # the loser switches at every step
