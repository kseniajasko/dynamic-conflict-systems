"""Building blocks of the experiments of Section 5 (small sizes)."""

from __future__ import annotations

import numpy as np
import pytest

from maxmap.analysis import asymmetric_attractor, exchange_cycle
from maxmap.experiments.boundaries import bisect_boundary, closest_approach, sample_boundary_pairs
from maxmap.experiments.collapse import (
    FIGURE_TRAJECTORIES,
    collapse_deviation,
    collapse_statistics,
    figure_initial_points,
    trajectory_rates,
)
from maxmap.experiments.config import Window
from maxmap.experiments.convergence import random_convergence
from maxmap.experiments.periodic import find_periodic_orbits
from maxmap.experiments.phases import (
    NOT_CONVERGED,
    PHASE_FIRST,
    PHASE_SECOND,
    classify_phases,
    offset_grid,
)
from maxmap.experiments.selfsimilarity import (
    colour_changes_along_ray,
    corner_one_comparisons,
    scaling_disagreement,
)
from maxmap.experiments.uncertainty import fitted_exponent, local_slopes, uncertainty_rows
from maxmap.model import iterate


@pytest.mark.parametrize(("lo", "hi", "n"), [(0.5, 1.0, 600), (0.99, 1.0, 400), (0.5, 0.505, 37)])
def test_offset_grid_avoids_the_diagonal(lo: float, hi: float, n: int) -> None:
    grid_x, grid_y = offset_grid(lo, hi, n)
    assert grid_x.shape == (n - 1, n)
    assert not np.any(grid_x == grid_y)
    assert np.min(grid_x) > lo
    assert np.max(grid_y) < hi


@pytest.mark.parametrize("alpha", [1.5, 1.9, 2.5])
def test_early_stopping_gives_the_brute_force_phases(alpha: float) -> None:
    grid_x, grid_y = offset_grid(0.5, 1.0, 80)
    fast = classify_phases(grid_x, grid_y, alpha, max_steps=4000, tol=1e-6)
    x, y = iterate(grid_x, grid_y, alpha, 1000)
    u, v = asymmetric_attractor(alpha)
    slow = np.full(grid_x.shape, NOT_CONVERGED)
    slow[(np.abs(x - u) < 1e-6) & (np.abs(y - v) < 1e-6)] = PHASE_FIRST
    slow[(np.abs(x - v) < 1e-6) & (np.abs(y - u) < 1e-6)] = PHASE_SECOND
    assert np.array_equal(fast, slow)


def test_phase_with_array_parameter_equals_scalar_calls() -> None:
    grid_x, grid_y = offset_grid(0.5, 1.0, 30)
    alpha = np.where(grid_x > 0.75, 1.5, 1.9)
    mixed = classify_phases(grid_x, grid_y, alpha, max_steps=2000, tol=1e-6)
    for value in (1.5, 1.9):
        mask = alpha == value
        single = classify_phases(grid_x[mask], grid_y[mask], value, max_steps=2000, tol=1e-6)
        assert np.array_equal(mixed[mask], single)


def test_odd_step_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="even"):
        classify_phases(np.array([0.7]), np.array([0.6]), 1.5, max_steps=11, tol=1e-6)


@pytest.mark.parametrize("alpha", [0.2, 0.5, 1.0])
def test_phase_is_the_sign_of_the_difference_for_alpha_at_most_one(alpha: float) -> None:
    grid_x, grid_y = offset_grid(0.5, 1.0, 60)
    phases = classify_phases(grid_x, grid_y, alpha, max_steps=4000, tol=1e-6)
    assert np.array_equal(phases, np.where(grid_x > grid_y, PHASE_FIRST, PHASE_SECOND))


def test_random_convergence_small_sample() -> None:
    rng = np.random.default_rng(3)
    res = random_convergence(1.5, rng, n_points=2000, max_steps=6000, tol=1e-10)
    assert res.converged == res.n_points
    assert res.median_steps is not None
    assert 90 <= res.median_steps <= 120
    assert res.wrong_winner is None
    res_above = random_convergence(2.5, rng, n_points=2000, max_steps=6000, tol=1e-10)
    assert res_above.wrong_winner == 0


def test_boundary_points_lie_on_the_stable_set_of_the_saddle() -> None:
    alpha = 1.5
    grid_x, grid_y = offset_grid(0.5, 1.0, 60)
    phases = classify_phases(grid_x, grid_y, alpha, max_steps=1500, tol=1e-6)
    pairs = sample_boundary_pairs(grid_x, grid_y, phases, np.random.default_rng(0), 10)
    x_b = bisect_boundary(pairs, alpha, bisections=50, max_steps=1500, tol=1e-6)
    to_diag, to_saddle = closest_approach(x_b, pairs.y, alpha, steps=400)
    assert np.max(to_diag) < 1e-12
    assert np.max(to_saddle) < 1e-8
    control_diag, control_saddle = closest_approach(
        grid_x[::7, ::7], grid_y[::7, ::7], alpha, steps=400
    )
    assert np.median(control_saddle) > 1e-4
    assert np.median(control_diag) > 1e-4


def test_uncertainty_ratio_and_exponent() -> None:
    rows = uncertainty_rows(
        1.5, np.random.default_rng(5), n_points=40_000, eps=(1e-2, 1e-3), max_steps=1000
    )
    assert rows[0].ratio == pytest.approx(6.2, rel=0.1)
    low, high = rows[0].ratio_interval()
    assert low < rows[0].ratio < high
    assert local_slopes(rows)[0] == pytest.approx(1.0, abs=0.1)
    slope, err = fitted_exponent(rows)
    assert slope == pytest.approx(1.0, abs=5 * err + 0.05)


def test_collapse_ratio_tends_to_three_quarters() -> None:
    dev = collapse_deviation(np.array([0.9, 0.7]), np.array([0.6, 0.95]), 90)
    np.testing.assert_allclose(dev[82] / dev[80], 0.75, atol=1e-5)
    on_diagonal = collapse_deviation(np.array([0.9]), np.array([0.9]), 40)
    assert on_diagonal[39, 0] / on_diagonal[38, 0] == pytest.approx(0.5, abs=1e-6)
    stats = collapse_statistics(np.random.default_rng(2), n_points=2000, times=(80,))
    assert stats.max_abs_deviation[0] < 1e-5


def test_figure_trajectories_collapse_with_two_step_factor_three_quarters() -> None:
    points = figure_initial_points()
    assert points.shape == (FIGURE_TRAJECTORIES, 2)
    assert np.all((points > 0.5) & (points < 1.0))
    np.testing.assert_array_equal(points, figure_initial_points())  # fixed seed
    for rate in trajectory_rates(points, ratio_time=80, factor_time=100):
        assert rate.ratio_deviation < 2e-6
        first, second = rate.one_step_factors
        assert first * second == pytest.approx(0.75, abs=1e-6)
        assert 0.75 < min(first, second) <= max(first, second) < 1.0


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_scaling_and_rays_see_no_structure_for_alpha_at_most_one(alpha: float) -> None:
    # For alpha <= 1 the phase is the sign of x - y, which is invariant under scaling
    # about (1/2, 1/2) and constant along rays below the diagonal.
    share, converged = scaling_disagreement(
        Window(alpha, 0.5, 0.6), 0.75, 40, max_steps=4000, tol=1e-6
    )
    assert converged
    assert share == 0.0
    changes, converged = colour_changes_along_ray(
        alpha, 0.25, u_min=1e-6, u_max=0.2, samples=500, max_steps=4000, tol=1e-6
    )
    assert converged
    assert changes == 0


def test_layers_near_the_corner_grow_logarithmically() -> None:
    counts = [
        colour_changes_along_ray(
            alpha, 0.25, u_min=1e-6, u_max=0.2, samples=4000, max_steps=4000, tol=1e-6
        )[0]
        for alpha in (1.99, 1.999)
    ]
    # Two colour changes per layer of relative width 4/3: 2 log_{4/3} 10 = 16 per decade.
    assert 13 <= counts[1] - counts[0] <= 19


def test_corner_one_comparisons() -> None:
    comparisons, sector, converged = corner_one_comparisons(
        Window(1.99, 0.99, 1.0),
        (Window(1.99, 0.9, 1.0),),
        120,
        sector_deg=10.0,
        max_steps=4000,
        tol=1e-6,
    )
    assert converged
    assert sector == pytest.approx(1 - np.tan(np.radians(35.0)), abs=0.02)
    (comp,) = comparisons
    assert 0.0 < comp.share < 0.2
    assert sector < comp.near_diagonal <= 1.0
    with pytest.raises(ValueError, match="corner"):
        corner_one_comparisons(
            Window(1.99, 0.5, 0.55), (), 10, sector_deg=10.0, max_steps=40, tol=1e-6
        )


def test_periodic_orbit_search_finds_the_known_orbits() -> None:
    orbits = find_periodic_orbits(
        1.5, np.random.default_rng(1), max_period=2, starts=400, newton_steps=60
    )
    kinds = sorted((o.period, o.is_symmetric, o.is_attracting) for o in orbits)
    assert kinds == [(1, True, False), (2, False, True)]
    cycle = next(o for o in orbits if o.period == 2)
    expected = sorted([exchange_cycle(1.5), exchange_cycle(1.5)[::-1]])
    np.testing.assert_allclose(sorted(cycle.points), expected, atol=1e-12)
