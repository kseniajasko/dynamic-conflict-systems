"""Regression against the published numbers of Section 5.1 (``reference_output``)."""

from __future__ import annotations

import numpy as np
import pytest

from maxmap.experiments.config import PAPER_SEED, PHASE_PANELS, RandomConvergenceConfig, Window
from maxmap.experiments.convergence import random_convergence
from maxmap.experiments.phases import window_phases
from maxmap.experiments.selfsimilarity import consecutive_disagreements

# Published: alpha -> (median steps, max steps); 100 % converged; no wrong winner.
PUBLISHED_PART1 = {
    0.2: (28, 44),
    0.5: (44, 66),
    1.0: (76, 98),
    1.5: (106, 132),
    1.9: (126, 152),
    1.99: (126, 156),
    2.01: (146, 148),
    2.1: (112, 128),
    2.5: (64, 88),
    3.0: (46, 72),
    5.0: (30, 50),
    10.0: (22, 38),
}


@pytest.mark.slow
def test_part1_reproduces_the_published_table() -> None:
    cfg = RandomConvergenceConfig()
    rng = np.random.default_rng(PAPER_SEED)
    for alpha in cfg.alphas:  # the stream is shared, so the order of alphas matters
        res = random_convergence(
            alpha, rng, n_points=cfg.n_points, max_steps=cfg.max_steps, tol=cfg.tol
        )
        assert res.converged == cfg.n_points
        assert (res.median_steps, res.max_steps) == PUBLISHED_PART1[alpha]
        assert res.wrong_winner in (None, 0)


@pytest.mark.slow
@pytest.mark.parametrize("window", [PHASE_PANELS[0], PHASE_PANELS[5]])
def test_part2_every_grid_point_converges(window: Window) -> None:
    phases = window_phases(window, 600, max_steps=4000, tol=1e-6)
    assert phases.shape == (599, 600)
    assert int((phases < 0).sum()) == 0


@pytest.mark.slow
def test_part5_corner_shares() -> None:
    windows_one = (Window(1.99, 0.9, 1.0), Window(1.99, 0.99, 1.0), Window(1.99, 0.999, 1.0))
    windows_half = (
        Window(1.99, 0.5, 0.505),
        Window(1.999, 0.5, 0.5005),
        Window(1.9999, 0.5, 0.50005),
    )
    one, ok_one = consecutive_disagreements(windows_one, 400, max_steps=4000, tol=1e-6)
    half, ok_half = consecutive_disagreements(windows_half, 400, max_steps=4000, tol=1e-6)
    assert ok_one and ok_half
    assert [f"{s:.4f}" for s in one] == ["0.0462", "0.0177"]
    assert [f"{s:.4f}" for s in half] == ["0.0067", "0.0005"]
