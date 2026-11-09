# maxmap — numerical experiments for the map of maxima

Supplementary code for the paper *From oscillations to symmetry breaking in a
two-dimensional piecewise-smooth dynamical system*. It reproduces all nine figures
and every number of Section 5 ("Numerical experiments") of the paper.

Author: Oksana Satur, Institute of Mathematics of the National Academy of Sciences of
Ukraine. Repository: https://github.com/kseniajasko/dynamic-conflict-systems (folder
`maxmap`).

The package implements the map of maxima

    Phi(x, y) = ( max{a x (1-y), 1-x} / (a x (1-y) + 1-x),
                  max{a y (1-x), 1-y} / (a y (1-x) + 1-y) ),   (x, y) in [1/2, 1)^2,

the full system of two stochastic vectors on two positions, the closed forms of the paper
(symmetric state, exchange 2-cycle, winner–loser fixed point, multipliers), and the
numerical checks of global convergence.

## Installation

Python ≥ 3.10. Runtime dependencies: NumPy ≥ 1.26 and Matplotlib ≥ 3.8 (nothing else).

```
pip install .              # the package and the `maxmap` command
pip install -e ".[dev]"    # editable install with pytest, hypothesis, ruff, mypy
```

## Usage

```
maxmap figures --out out                      # Figures 1-9 (PDF + PNG previews)
maxmap figures --out out --only fig_basins    # selected figures
maxmap figures --out preview --fast           # coarse previews (not the published figures)
maxmap checks --out out                       # the checks of Section 5.1
maxmap checks --out out --extended            # plus the additional checks (see below)
maxmap checks --out out --seed 1              # the same checks with another random sample
maxmap checks --out out --fast                # smoke test with small samples
maxmap list                                   # figure names and numbers
```

`python -m maxmap ...` is equivalent to `maxmap ...`. Progress is reported through
`logging` on standard error (`-v` for debug output, `-q` for warnings only).

`maxmap checks` writes `convergence_check.txt` (the report quoted in the paper) and
`convergence_check.json` (the same results in machine-readable form); with `--extended`
also `convergence_check_extended.txt`/`.json`. The exit status is 0 if all pass criteria
hold and 1 otherwise. The folder `reference_output/` contains both text reports as obtained
with the default seed, for comparison.

The functions can also be used directly, for example:

```python
from maxmap import exchange_cycle
from maxmap.experiments.phases import classify_phases, offset_grid

x_star, y_star = exchange_cycle(1.0)  # (1/psi, 1/psi^2), psi = plastic number
gx, gy = offset_grid(0.5, 1.0, 600)  # the 600 x 599 grid of Fig. 6
phases = classify_phases(gx, gy, 1.5, max_steps=4000, tol=1e-6)
```

## What reproduces what

| Paper | Output | Function |
|---|---|---|
| Fig. 1 — switching curves, branch regions, attractors (α = 1, 1.5, 2, 2.5) | `fig_switching_curves.pdf` | `maxmap.plotting.figures.fig_switching_curves` |
| Fig. 2 — bifurcation diagram of the maxima | `fig_bif_main.pdf` | `fig_bif_main` |
| Fig. 3 — symmetric branch, two-step contraction | `fig_bif_mech.pdf` | `fig_bif_mech` |
| Fig. 4 — coordinate-wise diagram, asymmetric start | `fig_bif_coord.pdf` | `fig_bif_coord` |
| Fig. 5 — coordinate-wise diagram, symmetric and mirror starts | `fig_bif_coord_sym_mirror.pdf` | `fig_bif_coord_sym_mirror` |
| Fig. 6 — phases of the exchange 2-cycle (whole square, corner windows) | `fig_basins.pdf` | `fig_basins` |
| Fig. 7 — trajectories in the plane | `fig_trajectories.pdf` | `fig_trajectories` |
| Fig. 8 — time series of the maxima | `fig_time_series.pdf` | `fig_time_series` |
| Fig. 9 — collapse at α = 2 | `fig_collapse.pdf` | `fig_collapse` |
| Section 5.1, 10^5 random points per α (convergence, medians, maxima, winner) | `convergence_check.txt`, part 1 | `experiments.convergence.random_convergence` |
| Section 5.1 and Fig. 6, 600 × 599 grids | part 2 | `experiments.phases.classify_phases` |
| Section 5.1, colour boundaries by bisection | part 3 | `experiments.boundaries` |
| Section 5.1, uncertainty exponent f(ε)/ε | part 4 | `experiments.uncertainty` |
| Section 5.1, self-similarity near the corners | part 5 | `experiments.selfsimilarity` |
| Section 5.1, the seeds 1–8 | `maxmap checks --seed k`, k = 1, …, 8 | |
| Section 5.1, additional checks | `convergence_check_extended.txt`, parts B–H | see below |
| Section 5.2 and Fig. 9, collapse at α = 2 | figure; extended parts A and I | `experiments.collapse` |

Additional checks (`--extended`, report `convergence_check_extended.txt`):

* A — ratio over two steps at α = 2 for 10^5 random initial points;
* B — control sample for part 3 (closest approach of random, non-boundary initial points);
* C — weighted fit of the uncertainty exponent 2 − D, with standard errors;
* D — Newton search for periodic orbits of periods 1–8;
* E — the phase rule for α ≤ 1 (the phase is determined by the sign of x0 − y0);
* F — corner (1,1): the windows [0.9,1]^2 and [0.999,1]^2 compared with panel (d) of Fig. 6
  on 600 × 599 grids, and the share of the differing points near the diagonal;
* G — scaling by 3/4 about (1/2,1/2): panel (e) of Fig. 6 and the window [0.5,0.505]^2
  for α = 1.99999;
* H — colour changes along the rays v = s u near (1/2,1/2) for α = 1.99, 1.999, 1.9999;
* I — the five trajectories of Fig. 9: ratio over two steps at t = 80 and one-step factors
  at t = 100.

## Expected runtime

Measured on two cores (Python 3.13, NumPy 2.5.3, Matplotlib 3.11.2):

| Command | Time |
|---|---|
| `maxmap figures` | about 7 s |
| `maxmap checks` | about 45 s (part 4 with 10^6 points per α takes most of it) |
| `maxmap checks --extended` | about 1 min more (periodic-orbit search) |
| `pytest` | about 12 s |

## Reproducibility

* **Random numbers.** All random numbers come from explicitly seeded
  `numpy.random.default_rng` generators: the checks use `seed` (part 1), `seed + 1`
  (part 3), `seed + 2` (part 4) and `seed + 3`…`seed + 5` (extended checks), with
  `seed = 2026` by default; Fig. 9 uses seed 7. In part 1 one stream is shared by all
  values of α, so the results depend on the order of the list of α.
* **Bit reproducibility.** The order of the floating-point operations in `maxmap.model.phi`
  is that of the code used for the paper, so the report `convergence_check.txt` is
  identical, character for character, to the published one. Since `1 - x` is exact for
  x in [1/2, 1] (Sterbenz lemma), `phi` is accurate to a few units in the last place on the
  whole square, and it is exactly symmetric: points of the diagonal stay on it.
* **Early stopping.** Phases are classified at the first even step at which a trajectory is
  within 10^-6 |x* − y*| of a point of the attractor, and only unclassified points are
  iterated further; the step limits (4000, 1500, 1000) are upper bounds. On every grid of the
  paper this gives the same phases as a fixed number of 4000 iterations (all grid points
  are classified within 140 steps).
* **Full system.** The normalizing denominators are computed, as in the definition of the
  law, as the sums of the numerators, so the vectors stay stochastic to one rounding and no
  renormalization is needed. (Computing them as `a x (1 - y) + 1 - x`, which presupposes
  p1 + p2 = 1 exactly, amplifies the deviation of p1 + p2 from 1 by x*/(1 − x*) over two
  steps near the 4-cycle.)
* **Exchange 2-cycle.** x* is computed with `numpy.roots`; its error is at most a few units
  in the last place (checked against 50-digit values in the tests).
* **Figures.** Figures are drawn without `matplotlib.pyplot` and without changing global
  Matplotlib settings. The published figures use the font Inter (the default of
  Matplotlib 3.11 when the font is installed); if Inter is not available, DejaVu Sans
  (shipped with Matplotlib) is used and a warning is logged. Pixel-identical figures
  require the same Matplotlib version (3.11.2) and fonts; with other versions the content
  is the same, but text metrics and tick labelling may differ slightly. PDFs are written
  without a creation date, so repeated runs give byte-identical files.

## Development

```
python -m pytest            # unit, property-based and regression tests (about 12 s)
ruff check . && ruff format --check .
python -m mypy              # strict mode, configured in pyproject.toml
```

## Layout

```
src/maxmap/
  model.py            map of maxima, branches, switching curves, logarithmic form, full system
  analysis.py         closed forms: x_sym, exchange 2-cycle, fixed point, multipliers
  experiments/
    config.py         dataclasses with the parameters of the paper (and a fast variant)
    phases.py         offset grids and phase classification (early stopping)
    convergence.py    part 1: random initial points
    boundaries.py     part 3: bisection and closest approach
    uncertainty.py    part 4: f(eps), local slopes, confidence intervals, fitted exponent
    selfsimilarity.py part 5 and extended F-H: windows, scaling and rays near the corners
    collapse.py       alpha = 2: G_t - 1/2, two-step ratios, trajectories of Fig. 9
    periodic.py       Newton search for periodic orbits
    diagrams.py       data of Figs. 2-5
    checks.py         orchestration and the reports
  plotting/
    style.py          colours, style context, figure creation and saving
    figures.py        one function per figure and the registry FIGURES
  cli.py              command line interface
tests/                pytest suite
reference_output/     reports of `maxmap checks --extended` with the default seed
```
