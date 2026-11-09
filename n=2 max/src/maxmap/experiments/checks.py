"""The checks of Section 5.1 and the report ``convergence_check.txt``.

:func:`run_checks` reproduces, with the default configuration, every number of the published
report; :func:`format_report` produces the same text.  :func:`run_extended_checks` performs
additional checks (command-line option ``--extended``).
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from maxmap.analysis import x_sym
from maxmap.experiments.boundaries import bisect_boundary, closest_approach, sample_boundary_pairs
from maxmap.experiments.collapse import (
    FIGURE_SEED,
    CollapseStatistics,
    TrajectoryRate,
    collapse_statistics,
    figure_initial_points,
    trajectory_rates,
)
from maxmap.experiments.config import ChecksConfig, Window
from maxmap.experiments.convergence import ConvergenceResult, random_convergence
from maxmap.experiments.periodic import PeriodicOrbit, find_periodic_orbits
from maxmap.experiments.phases import (
    PHASE_FIRST,
    PHASE_SECOND,
    classify_phases,
    offset_grid,
)
from maxmap.experiments.selfsimilarity import (
    CornerComparison,
    colour_changes_along_ray,
    consecutive_disagreements,
    corner_one_comparisons,
    scaling_disagreement,
)
from maxmap.experiments.uncertainty import (
    UncertaintyRow,
    fitted_exponent,
    local_slopes,
    uncertainty_rows,
)

__all__ = [
    "BoundaryResult",
    "ChecksReport",
    "ExtendedReport",
    "GridResult",
    "extended_report_files",
    "format_extended_report",
    "format_report",
    "report_files",
    "run_checks",
    "run_extended_checks",
    "write_report",
]

logger = logging.getLogger(__name__)

REPORT_NAME = "convergence_check.txt"
EXTENDED_REPORT_NAME = "convergence_check_extended.txt"

EXPONENT_FIT_MAX_EPS = 1e-4
"""Largest displacement used in the fit of the uncertainty exponent.

For alpha close to 2, f(eps)/eps still grows down to eps of about 1e-4 (the layered structure
near the corner (1/2, 1/2)), so larger displacements are excluded from the fit.
"""


@dataclass(frozen=True, slots=True)
class GridResult:
    """Part 2: one panel of Fig. 6."""

    window: Window
    points: int
    not_converged: int


@dataclass(frozen=True, slots=True)
class BoundaryResult:
    """Part 3: boundary points of one panel."""

    window: Window
    points: int
    max_to_diagonal: float
    max_to_saddle: float


@dataclass(slots=True)
class ChecksReport:
    """Results of the five checks."""

    config: ChecksConfig
    random: list[ConvergenceResult] = field(default_factory=list)
    grids: list[GridResult] = field(default_factory=list)
    boundaries: list[BoundaryResult] = field(default_factory=list)
    uncertainty: dict[float, list[UncertaintyRow]] = field(default_factory=dict)
    corner_one: list[float] = field(default_factory=list)
    corner_half: list[float] = field(default_factory=list)
    corners_converged: bool = True
    seconds: float = 0.0

    @property
    def passed(self) -> bool:
        """Whether all pass criteria of the published checks hold."""
        cfg = self.config.boundaries
        return (
            all(r.converged == r.n_points and r.wrong_winner in (None, 0) for r in self.random)
            and all(g.not_converged == 0 for g in self.grids)
            and all(
                b.max_to_diagonal < cfg.max_diagonal_distance
                and b.max_to_saddle < cfg.max_saddle_distance
                for b in self.boundaries
            )
            and self.corners_converged
        )


def _timed(label: str, func: Callable[[], None]) -> None:
    start = time.perf_counter()
    func()
    logger.info("%s done in %.1f s", label, time.perf_counter() - start)


def run_checks(config: ChecksConfig | None = None) -> ChecksReport:
    """Run the five checks of Section 5.1."""
    config = config or ChecksConfig()
    report = ChecksReport(config)
    start = time.perf_counter()

    def part1() -> None:
        rng = np.random.default_rng(config.seed)
        cfg = config.random
        for alpha in cfg.alphas:
            report.random.append(
                random_convergence(
                    alpha, rng, n_points=cfg.n_points, max_steps=cfg.max_steps, tol=cfg.tol
                )
            )

    grids: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def part2() -> None:
        cfg = config.grids
        for window in cfg.panels:
            grid_x, grid_y = offset_grid(window.lo, window.hi, cfg.n)
            phases = classify_phases(
                grid_x, grid_y, window.alpha, max_steps=cfg.max_steps, tol=cfg.tol
            )
            grids.append((grid_x, grid_y, phases))
            report.grids.append(GridResult(window, grid_x.size, int((phases < 0).sum())))

    def part3() -> None:
        cfg = config.boundaries
        rng = np.random.default_rng(config.seed + 1)
        for window, (grid_x, grid_y, phases) in zip(config.grids.panels, grids, strict=True):
            pairs = sample_boundary_pairs(grid_x, grid_y, phases, rng, cfg.per_panel)
            x_b = bisect_boundary(
                pairs,
                window.alpha,
                bisections=cfg.bisections,
                max_steps=cfg.colour_steps,
                tol=config.grids.tol,
            )
            to_diag, to_saddle = closest_approach(
                x_b, pairs.y, window.alpha, steps=cfg.approach_steps
            )
            report.boundaries.append(
                BoundaryResult(window, x_b.size, float(np.max(to_diag)), float(np.max(to_saddle)))
            )

    def part4() -> None:
        cfg = config.uncertainty
        rng = np.random.default_rng(config.seed + 2)
        for alpha in cfg.alphas:
            report.uncertainty[alpha] = uncertainty_rows(
                alpha, rng, n_points=cfg.n_points, eps=cfg.eps, max_steps=cfg.max_steps
            )

    def part5() -> None:
        cfg = config.selfsimilarity
        one, ok_one = consecutive_disagreements(
            cfg.corner_one, cfg.n, max_steps=cfg.max_steps, tol=config.grids.tol
        )
        half, ok_half = consecutive_disagreements(
            cfg.corner_half, cfg.n, max_steps=cfg.max_steps, tol=config.grids.tol
        )
        report.corner_one, report.corner_half = one, half
        report.corners_converged = ok_one and ok_half

    _timed("part 1 (random initial points)", part1)
    _timed("part 2 (regular grids)", part2)
    _timed("part 3 (colour boundaries)", part3)
    _timed("part 4 (uncertainty exponent)", part4)
    _timed("part 5 (self-similarity)", part5)
    report.seconds = time.perf_counter() - start
    return report


def format_report(report: ChecksReport) -> list[str]:
    """Lines of ``convergence_check.txt`` (identical to the published report)."""
    cfg = report.config
    lines = [
        f"1. Random initial points: N = {cfg.random.n_points} per alpha, seed = {cfg.seed}, "
        f"up to {cfg.random.max_steps} steps,",
        f"   tolerance {cfg.random.tol:g} at even steps",
        "",
        "   alpha   converged   median steps   max steps   wrong winner (alpha > 2)",
    ]
    for r in report.random:
        wrong = "--" if r.wrong_winner is None else str(r.wrong_winner)
        lines.append(
            f"   {r.alpha:5g}   {100 * r.converged_fraction:8.3f} %   {r.median_steps!s:>12}   "
            f"{r.max_steps!s:>9}   {wrong}"
        )
    g = cfg.grids
    lines += [
        "",
        f"2. Regular {g.n} x {g.n - 1} grids, {g.max_steps} steps, tolerance {g.tol:g}",
        "",
    ]
    for grid in report.grids:
        w = grid.window
        lines.append(
            f"   alpha = {w.alpha:g}, window {w.label}: grid points = {grid.points}, "
            f"not converged to the exchange 2-cycle = {grid.not_converged}"
        )
    lines += [
        "",
        "3. Colour boundaries located by bisection (closest approach of their trajectories)",
        "",
    ]
    for b in report.boundaries:
        w = b.window
        lines.append(
            f"   alpha = {w.alpha:g}, window {w.label}: {b.points} points; max distance to the "
            f"diagonal {b.max_to_diagonal:.1e}, to the saddle {b.max_to_saddle:.1e}"
        )
    u = cfg.uncertainty
    lines += [
        "",
        f"4. Uncertainty exponent: f(eps)/eps, {u.n_points} random points, displacement in a "
        "random direction",
        "",
    ]
    for alpha, rows in report.uncertainty.items():
        cells = "  ".join(f"eps={row.eps:.0e}: {row.ratio:6.2f} ({row.changed})" for row in rows)
        missing = sum(row.not_converged for row in rows)
        lines.append(f"   alpha = {alpha:<4g} {cells}   not converged: {missing}")
        slopes = "  ".join(f"{s:.2f}" for s in local_slopes(rows))
        lines.append(f"                local slopes d(log f)/d(log eps): {slopes}")
    s = cfg.selfsimilarity
    one = s.corner_one
    lines += [
        "",
        "5. Self-similarity near the corners: share of differing points of "
        f"{s.n} x {s.n - 1} grids",
        "",
        f"   corner (1,1), alpha = {one[0].alpha:g}: "
        + "   ".join(
            f"{a.label} vs {b.label}: {share:.4f}"
            for a, b, share in zip(one[:-1], one[1:], report.corner_one, strict=True)
        ),
        "   corner (1/2,1/2): "
        + "   ".join(
            f"alpha = {a.alpha:g} vs {b.alpha:g}: {share:.4f}"
            for a, b, share in zip(
                s.corner_half[:-1], s.corner_half[1:], report.corner_half, strict=True
            )
        ),
        "",
        "All checks passed." if report.passed else "SOME CHECKS FAILED.",
    ]
    return lines


def _report_json(report: ChecksReport) -> dict[str, object]:
    return {
        "config": asdict(report.config),
        "random": [asdict(r) for r in report.random],
        "grids": [asdict(g) for g in report.grids],
        "boundaries": [asdict(b) for b in report.boundaries],
        "uncertainty": {
            str(alpha): [asdict(row) | {"ratio": row.ratio} for row in rows]
            for alpha, rows in report.uncertainty.items()
        },
        "corner_one": report.corner_one,
        "corner_half": report.corner_half,
        "passed": report.passed,
        "seconds": report.seconds,
    }


def write_report(lines: list[str], out_dir: Path, name: str, data: dict[str, object]) -> Path:
    """Write a text report and its JSON companion; return the path of the text file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def report_files(report: ChecksReport, out_dir: Path) -> Path:
    """Write ``convergence_check.txt`` and ``convergence_check.json``."""
    return write_report(format_report(report), out_dir, REPORT_NAME, _report_json(report))


# --------------------------------------------------------------------------- extended checks


@dataclass(slots=True)
class ExtendedReport:
    """Results of the additional checks."""

    collapse: CollapseStatistics | None = None
    control_to_diagonal: dict[float, tuple[float, float]] = field(default_factory=dict)
    """Per alpha: (median, minimum) over the sample of the closest approach to the diagonal."""
    control_to_saddle: dict[float, tuple[float, float]] = field(default_factory=dict)
    """Per alpha: (median, minimum) of the closest approach to the symmetric saddle."""
    exponents: dict[float, tuple[float, float]] = field(default_factory=dict)
    periodic: dict[float, list[PeriodicOrbit]] = field(default_factory=dict)
    sign_rule: dict[float, tuple[int, int]] = field(default_factory=dict)
    corner_one: list[CornerComparison] = field(default_factory=list)
    sector_share: float = 0.0
    """Share of the grid points in the sector around the diagonal (part F)."""
    scaling: list[tuple[Window, float]] = field(default_factory=list)
    rays: dict[float, list[int]] = field(default_factory=dict)
    figure_rates: list[TrajectoryRate] = field(default_factory=list)
    corners_converged: bool = True


def run_extended_checks(config: ChecksConfig, base: ChecksReport | None = None) -> ExtendedReport:
    """Additional checks: collapse, controls, exponent fits, periodic orbits, corners.

    Parameters
    ----------
    config
        Configuration; its ``seed`` offsets ``+3`` to ``+5`` are used for the new random
        streams, so that the published streams are not affected.
    base
        Report of :func:`run_checks`, whose uncertainty counts are used for the fits.
    """
    ext = ExtendedReport()
    cfg = config.extended

    def collapse() -> None:
        rng = np.random.default_rng(config.seed + 3)
        ext.collapse = collapse_statistics(
            rng, n_points=cfg.collapse.n_points, times=cfg.collapse.times
        )

    def control() -> None:
        rng = np.random.default_rng(config.seed + 4)
        for alpha in sorted({w.alpha for w in config.grids.panels}):
            x0 = rng.uniform(0.5, 1.0, cfg.control_points)
            y0 = rng.uniform(0.5, 1.0, cfg.control_points)
            to_diag, to_saddle = closest_approach(
                x0, y0, alpha, steps=config.boundaries.approach_steps
            )
            ext.control_to_diagonal[alpha] = (float(np.median(to_diag)), float(np.min(to_diag)))
            ext.control_to_saddle[alpha] = (float(np.median(to_saddle)), float(np.min(to_saddle)))

    def exponents() -> None:
        if base is None:
            return
        for alpha, rows in base.uncertainty.items():
            small = [row for row in rows if row.eps <= EXPONENT_FIT_MAX_EPS]
            if len(small) >= 2 and all(row.changed > 0 for row in small):
                ext.exponents[alpha] = fitted_exponent(small)

    def periodic() -> None:
        rng = np.random.default_rng(config.seed + 5)
        p = cfg.periodic
        for alpha in p.alphas:
            ext.periodic[alpha] = find_periodic_orbits(
                alpha, rng, max_period=p.max_period, starts=p.starts, newton_steps=p.newton_steps
            )

    def sign_rule() -> None:
        for alpha in cfg.sign_rule_alphas:
            grid_x, grid_y = offset_grid(0.5, 1.0, cfg.sign_rule_n)
            phases = classify_phases(
                grid_x, grid_y, alpha, max_steps=config.grids.max_steps, tol=config.grids.tol
            )
            expected = np.where(grid_x > grid_y, PHASE_FIRST, PHASE_SECOND)
            ext.sign_rule[alpha] = (int(np.sum(phases != expected)), phases.size)

    steps, tol = config.grids.max_steps, config.grids.tol
    corners = cfg.corners

    def corner_one() -> None:
        ext.corner_one, ext.sector_share, ok = corner_one_comparisons(
            corners.corner_one_reference,
            corners.corner_one_windows,
            corners.n,
            sector_deg=corners.sector_deg,
            max_steps=steps,
            tol=tol,
        )
        ext.corners_converged = ext.corners_converged and ok

    def scaling() -> None:
        for window in corners.scaling_windows:
            share, ok = scaling_disagreement(
                window, corners.scaling_factor, corners.n, max_steps=steps, tol=tol
            )
            ext.scaling.append((window, share))
            ext.corners_converged = ext.corners_converged and ok

    def rays() -> None:
        for alpha in corners.ray_alphas:
            counts = []
            for slope in corners.ray_slopes:
                count, ok = colour_changes_along_ray(
                    alpha,
                    slope,
                    u_min=corners.ray_u_min,
                    u_max=corners.ray_u_max,
                    samples=corners.ray_samples,
                    max_steps=steps,
                    tol=tol,
                )
                counts.append(count)
                ext.corners_converged = ext.corners_converged and ok
            ext.rays[alpha] = counts

    def figure_rates() -> None:
        ext.figure_rates = trajectory_rates(
            figure_initial_points(),
            ratio_time=cfg.figure_ratio_time,
            factor_time=cfg.figure_factor_time,
        )

    _timed("extended: collapse at alpha = 2", collapse)
    _timed("extended: control points", control)
    _timed("extended: exponent fits", exponents)
    _timed("extended: periodic orbits", periodic)
    _timed("extended: sign rule for alpha <= 1", sign_rule)
    _timed("extended: corner (1,1) at the resolution of Fig. 6", corner_one)
    _timed("extended: scaling about (1/2,1/2)", scaling)
    _timed("extended: colour changes along rays", rays)
    _timed("extended: trajectories of Fig. 9", figure_rates)
    return ext


def format_extended_report(ext: ExtendedReport, config: ChecksConfig) -> list[str]:
    """Lines of ``convergence_check_extended.txt``."""
    cfg = config.extended
    lines = ["A. Collapse at alpha = 2: ratio (G_{t+2} - 1/2) / (G_t - 1/2)", ""]
    if ext.collapse is not None:
        c = ext.collapse
        lines.append(f"   {c.n_points} random initial points (seed {config.seed + 3})")
        for t, worst, gap in zip(c.times, c.max_abs_deviation, c.median_gap, strict=True):
            lines.append(
                f"   t = {t:4d}: max |ratio - 3/4| = {worst:.2e}   median G_t - 1/2 = {gap:.1e}"
            )
    lines += [
        "",
        "B. Control for part 3: closest approach of random (non-boundary) initial points",
        f"   ({cfg.control_points} points per alpha, {config.boundaries.approach_steps} steps; "
        "median and minimum over the sample)",
        "",
    ]
    for alpha, (diag_median, diag_min) in ext.control_to_diagonal.items():
        saddle_median, saddle_min = ext.control_to_saddle[alpha]
        lines.append(
            f"   alpha = {alpha:g}: to the diagonal {diag_median:.1e} (min {diag_min:.1e}), "
            f"to the saddle {saddle_median:.1e} (min {saddle_min:.1e}); "
            f"x_sym = {x_sym(alpha):.6f}"
        )
    lines += [
        "",
        f"C. Uncertainty exponent 2 - D: weighted fit over eps <= {EXPONENT_FIT_MAX_EPS:.0e} "
        "(+- 1 s.e.)",
        "",
    ]
    for alpha, (slope, err) in ext.exponents.items():
        lines.append(
            f"   alpha = {alpha:<5g} 2 - D = {slope:.3f} +- {err:.3f}   D = {2 - slope:.3f}"
        )
    p = cfg.periodic
    lines += [
        "",
        f"D. Periodic orbits of Phi of periods 1..{p.max_period} found by Newton's method "
        f"({p.starts} starts per period)",
        "",
    ]
    for alpha, orbits in ext.periodic.items():
        lines.append(f"   alpha = {alpha:g}: {len(orbits)} orbit(s)")
        for orbit in orbits:
            kind = "attracting" if orbit.is_attracting else "saddle"
            where = "diagonal" if orbit.is_symmetric else "off the diagonal"
            pts = ", ".join(f"({x:.6f}, {y:.6f})" for x, y in orbit.points)
            mods = ", ".join(f"{abs(m):.4f}" for m in orbit.multipliers)
            lines.append(
                f"      period {orbit.period}, {kind}, {where}: {pts}; |multipliers| {mods}"
            )
    lines += [
        "",
        "E. Phase determined by the sign of x0 - y0 for alpha <= 1 (whole square)",
        "",
    ]
    for alpha, (bad, total) in ext.sign_rule.items():
        lines.append(
            f"   alpha = {alpha:g}: {total} grid points, phase (x*,y*) iff x0 > y0 fails at {bad}"
        )
    lines += _format_corners(ext, config)
    return lines


def _format_corners(ext: ExtendedReport, config: ChecksConfig) -> list[str]:
    """Sections F--I of ``convergence_check_extended.txt``."""
    cfg = config.extended
    c = cfg.corners
    grid = f"{c.n} x {c.n - 1}"
    lines = []
    if ext.corner_one:
        ref = ext.corner_one[0].reference
        lines += [
            "",
            f"F. Corner (1,1), alpha = {ref.alpha:g}: windows compared with {ref.label} "
            f"on {grid} grids",
            f"   (sector: rays from (1,1) within {c.sector_deg:g} degrees of the diagonal, "
            f"{100 * ext.sector_share:.0f}% of the grid)",
            "",
        ]
        for comp in ext.corner_one:
            lines.append(
                f"   {comp.window.label} vs {ref.label}: differing points {comp.share:.4f}, "
                f"of them in the sector {100 * comp.near_diagonal:.0f}%"
            )
    lines += [
        "",
        f"G. Scaling by {c.scaling_factor:g} about (1/2,1/2): share of the points w of {grid} "
        f"grids with colour(w) != colour(c + {c.scaling_factor:g} (w - c)), c = (1/2,1/2)",
        "",
    ]
    for window, share in ext.scaling:
        lines.append(f"   alpha = {window.alpha:g}, window {window.label}: {share:.4f}")
    lines += [
        "",
        f"H. Colour changes along the rays v = s u, {c.ray_u_min:g} <= u <= {c.ray_u_max:g} "
        f"({c.ray_samples} points, geometric spacing), x = 1/2 + u, y = 1/2 + v",
        "",
    ]
    for alpha, counts in ext.rays.items():
        cells = ", ".join(f"s = {s:g}: {n}" for s, n in zip(c.ray_slopes, counts, strict=True))
        layers = math.log(1.0 / (2.0 - alpha)) / math.log(4.0 / 3.0)
        lines.append(f"   alpha = {alpha:<7g} {cells};  log_(4/3) (1/(2-alpha)) = {layers:.1f}")
    lines += [
        "",
        f"I. Collapse at alpha = 2: the trajectories of Fig. 9 (seed {FIGURE_SEED})",
        "",
    ]
    for rate in ext.figure_rates:
        first, second = rate.one_step_factors
        lines.append(
            f"   start ({rate.start[0]:.4f}, {rate.start[1]:.4f}): "
            f"|ratio - 3/4| at t = {cfg.figure_ratio_time}: {rate.ratio_deviation:.1e}; "
            f"one-step factors at t = {cfg.figure_factor_time}: {first:.4f}, {second:.4f} "
            f"(product {first * second:.6f})"
        )
    if not ext.corners_converged:
        lines += ["", "WARNING: some points of sections F--H did not converge."]
    return lines


def extended_report_files(ext: ExtendedReport, config: ChecksConfig, out_dir: Path) -> Path:
    """Write ``convergence_check_extended.txt`` and its JSON companion."""
    data: dict[str, object] = {
        "collapse": None if ext.collapse is None else asdict(ext.collapse),
        "control_to_diagonal": {str(a): list(v) for a, v in ext.control_to_diagonal.items()},
        "control_to_saddle": {str(a): list(v) for a, v in ext.control_to_saddle.items()},
        "exponents": {str(a): list(v) for a, v in ext.exponents.items()},
        "periodic": {
            str(a): [
                {
                    "period": o.period,
                    "points": o.points,
                    "multipliers": [[m.real, m.imag] for m in o.multipliers],
                }
                for o in orbits
            ]
            for a, orbits in ext.periodic.items()
        },
        "sign_rule": {str(a): list(v) for a, v in ext.sign_rule.items()},
        "corner_one": [asdict(comp) for comp in ext.corner_one],
        "sector_share": ext.sector_share,
        "scaling": [{"window": asdict(w), "share": share} for w, share in ext.scaling],
        "rays": {str(a): counts for a, counts in ext.rays.items()},
        "figure_rates": [asdict(rate) for rate in ext.figure_rates],
        "corners_converged": ext.corners_converged,
    }
    return write_report(format_extended_report(ext, config), out_dir, EXTENDED_REPORT_NAME, data)
