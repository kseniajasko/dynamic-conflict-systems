"""One function per figure of the paper.

Every builder returns a :class:`matplotlib.figure.Figure` created inside
:func:`~maxmap.plotting.style.figure_style`; :func:`build_and_save` saves it.  The drawing
calls (and their order) are those of the published figures, so that the output is
pixel-identical with the same Matplotlib version and fonts.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from maxmap._types import FloatArray
from maxmap.analysis import ALPHA_CRITICAL, asymmetric_attractor, exchange_cycle, x_sym
from maxmap.experiments.collapse import (
    FIGURE_TRAJECTORIES,
    collapse_deviation,
    figure_initial_points,
)
from maxmap.experiments.config import PHASE_PANELS, Window
from maxmap.experiments.diagrams import (
    ALPHAS_COORDINATE,
    BifurcationBranches,
    bifurcation_branches,
    coordinate_diagram,
)
from maxmap.experiments.phases import classify_phases, offset_grid
from maxmap.model import curve_s, orbit, phi_point, switching_curve_tp, switching_curve_tr
from maxmap.plotting.style import (
    BLUE,
    GREY,
    PURPLE,
    RED,
    SPAN_LEFT,
    SPAN_RIGHT,
    TEAL,
    TEXT_WIDTH,
    figure_style,
    new_figure,
    save_figure,
)

__all__ = [
    "FIGURES",
    "FigureOptions",
    "FigureSpec",
    "build_and_save",
    "fig_basins",
    "fig_bif_coord",
    "fig_bif_coord_sym_mirror",
    "fig_bif_main",
    "fig_bif_mech",
    "fig_collapse",
    "fig_switching_curves",
    "fig_time_series",
    "fig_trajectories",
]

logger = logging.getLogger(__name__)

PHASE_GRID_N: Final = 600
"""Grid size of Fig. 6 (``n x (n - 1)`` points per panel)."""

PHASE_STEPS: Final = 4000
"""Step limit of Fig. 6 (the phases are final after at most about 150 steps)."""

PHASE_TOL: Final = 1e-6
"""Tolerance of the phase test of Fig. 6."""

_LABEL_STYLE: Final[dict[str, Any]] = {
    "fontsize": 8.5,
    "color": "#555",
    "ha": "center",
    "va": "center",
    "weight": "bold",
}


@dataclass(frozen=True, slots=True)
class FigureOptions:
    """Resolution options; the defaults reproduce the published figures."""

    phase_grid_n: int = PHASE_GRID_N
    """Fig. 6: ``n x (n - 1)`` grid points per panel."""
    coordinate_n: int = ALPHAS_COORDINATE.size
    """Figs. 4 and 5: number of parameter values in ``[0.02, 5]``."""

    @classmethod
    def fast(cls) -> FigureOptions:
        """Coarse settings for previews (not the published figures)."""
        return cls(phase_grid_n=150, coordinate_n=150)

    @property
    def alphas(self) -> FloatArray:
        """Parameter values of the coordinate-wise diagrams."""
        lo, hi = float(ALPHAS_COORDINATE[0]), float(ALPHAS_COORDINATE[-1])
        return np.linspace(lo, hi, self.coordinate_n)


def _grid(ax: Axes, *, dotted: bool = False) -> None:
    if dotted:
        ax.grid(True, ls=":", alpha=0.5)
    else:
        ax.grid(alpha=0.15)


def _spans(ax: Axes, a_c: float, a_max: float) -> None:
    ax.axvspan(0, a_c, color=SPAN_LEFT, alpha=0.55, lw=0)
    ax.axvspan(a_c, a_max, color=SPAN_RIGHT, alpha=0.55, lw=0)
    ax.axvline(a_c, color="k", lw=1.0)


# --------------------------------------------------------------------------- Fig. 1

SHADE_DOUBLE_ACTIVE: Final = "#d5d8dc"
SHADE_MIXED: Final = "#f2f3f4"
SWITCHING_ALPHAS: Final = (1.0, 1.5, 2.0, 2.5)
"""One value in each range: alpha <= 1, 1 < alpha < 2, alpha = 2, alpha > 2."""
_Pos = tuple[float, float]
SWITCHING_LABELS: Final[dict[float, dict[str, _Pos | tuple[_Pos, _Pos]]]] = {
    1.0: {"A/P": (0.875, 0.63), "P/A": (0.63, 0.875), "P/P": (0.63, 0.555)},
    1.5: {
        "A/P": (0.87, 0.66),
        "P/A": (0.66, 0.87),
        "A/A": ((0.86, 0.865), (0.74, 0.95)),
        "P/P": ((0.605, 0.592), (0.56, 0.735)),
    },
    2.0: {"A/A": (0.77, 0.73), "A/P": (0.88, 0.62), "P/A": (0.62, 0.88)},
    2.5: {"A/A": (0.735, 0.665), "A/P": (0.905, 0.645), "P/A": (0.645, 0.905)},
}
"""Label positions; a pair (target, text) marks a thin region labelled with a pointer."""


def _switching_panel(ax: Axes, a: float) -> None:
    x = np.linspace(0.5, 1, 601)
    tp, tr = switching_curve_tp(x, a), switching_curve_tr(x, a)
    upper_p, upper_r = np.clip(tp, 0.5, 1), np.clip(tr, 0.5, 1)
    kw: dict[str, Any] = {"lw": 0, "interpolate": True, "zorder": 0}
    ax.fill_between(x, 0.5, np.minimum(upper_p, upper_r), color=SHADE_MIXED, **kw)  # A/P
    ax.fill_between(x, np.maximum(upper_p, upper_r), 1, color=SHADE_MIXED, **kw)  # P/A
    ax.fill_between(
        x, upper_r, upper_p, where=upper_p > upper_r, color=SHADE_DOUBLE_ACTIVE, **kw
    )  # A/A; P/P stays white
    ax.plot(x, tp, "-", c=BLUE, lw=2.0)  # T_p
    ax.plot(tp, x, "--", c=RED, lw=2.0)  # T_r, the mirror image of T_p
    if 1 < a < ALPHA_CRITICAL:
        ax.plot(x, curve_s(x, a), "-.", c=TEAL, lw=2.0)
    ax.plot([0.5, 1], [0.5, 1], "k:", lw=0.9)
    if 1 < a <= ALPHA_CRITICAL:  # crossing point of T_p and T_r (and of S)
        ax.plot(1 / a, 1 / a, "o", ms=5.5, c="k", zorder=6, clip_on=False)
    if a != ALPHA_CRITICAL:  # asymmetric attractor and symmetric state
        u, v = asymmetric_attractor(a)
        ax.plot([u, v], [v, u], "*", ms=9, c="k", ls="none", zorder=6)
        ax.plot(x_sym(a), x_sym(a), "D", ms=4.5, mfc="white", mec="k", mew=0.8, zorder=6)
    for text, pos in SWITCHING_LABELS[a].items():
        if isinstance(pos[0], tuple):
            ax.annotate(
                text,
                xy=pos[0],
                xytext=pos[1],
                **_LABEL_STYLE,
                arrowprops={
                    "arrowstyle": "->",
                    "lw": 0.7,
                    "color": "gray",
                    "shrinkA": 2,
                    "shrinkB": 0,
                },
            )
        else:
            ax.text(pos[0], pos[1], text, **_LABEL_STYLE)
    ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_xlim(0.5, 1)
    ax.set_ylim(0.5, 1)
    ax.set_aspect("equal")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$", labelpad=1)
    ax.set_title(rf"$\alpha={a:g}$")
    ax.grid(True, ls=":", alpha=0.35, zorder=1)
    ax.set_axisbelow(True)


def fig_switching_curves(options: FigureOptions | None = None) -> Figure:
    """Fig. 1: switching curves, regions of the branches and attractors for four alphas."""
    del options
    width = 5.0  # included at 0.77 of the text width
    fig = new_figure(width, width + 0.6)
    axs = fig.subplots(2, 2)
    for ax, a in zip(axs.ravel(), SWITCHING_ALPHAS, strict=True):
        _switching_panel(ax, a)
    handles = [
        Line2D([], [], c=BLUE, lw=2.0, ls="-", label=r"$T_p:\ \alpha x(1-y)=1-x$"),
        Line2D([], [], c=RED, lw=2.0, ls="--", label=r"$T_r:\ \alpha y(1-x)=1-y$"),
        Line2D([], [], c=TEAL, lw=2.0, ls="-.", label=r"$S:\ \alpha^2xy=1$"),
        Line2D(
            [], [], c="k", lw=0, marker="o", ms=5.5, label=r"$(\frac{1}{\alpha},\frac{1}{\alpha})$"
        ),
        Line2D([], [], c="k", lw=0, marker="*", ms=9, label="asymmetric attractor"),
        Line2D(
            [],
            [],
            c="k",
            lw=0,
            marker="D",
            ms=4.5,
            mfc="white",
            mew=0.8,
            label=r"symmetric state $x_{\mathrm{sym}}$",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
        columnspacing=1.2,
        handletextpad=0.4,
    )
    fig.tight_layout(h_pad=0.8, w_pad=1.2, rect=(0, 0.075, 1, 1))
    return fig


# --------------------------------------------------------------------------- Fig. 2


def fig_bif_main(options: FigureOptions | None = None) -> Figure:
    """Fig. 2: asymptotic values of the maxima as functions of alpha."""
    del options
    br = bifurcation_branches()
    fig = new_figure(TEXT_WIDTH, 3.9)
    ax = fig.subplots()
    _spans(ax, 2, 5)
    ax.axhline(0.5, color="gray", lw=0.7, ls=":")
    ax.plot(br.alphas_below, br.cycle[:, 0], "-", c=TEAL, lw=2.4)
    ax.plot(br.alphas_below, br.cycle[:, 1], "-", c=TEAL, lw=2.4)
    ax.plot(br.alphas_below, br.sym_below, "--", c=GREY, lw=1.5)
    ax.plot(br.alphas_above, br.fixed[:, 0], "-", c=BLUE, lw=2.6)
    ax.plot(br.alphas_above, br.fixed[:, 1], "-", c=RED, lw=2.6)
    ax.plot(br.alphas_above, 1 - 1 / br.alphas_above, ":", c=GREY, lw=1.3)
    ax.plot(2, 0.5, "o", ms=9, mfc="white", mec="k", mew=1.6, zorder=6)
    a_osc = 0.6
    x_osc, y_osc = exchange_cycle(a_osc)
    ax.annotate(
        "",
        xy=(a_osc, x_osc),
        xytext=(a_osc, y_osc),
        arrowprops={"arrowstyle": "<->", "color": TEAL, "lw": 1.5},
    )
    ax.text(
        a_osc - 0.06,
        (x_osc + y_osc) / 2,
        "oscillation\namplitude",
        fontsize=7.5,
        ha="right",
        va="center",
        color="#0e6655",
    )
    a_gap = 3.2
    x_gap, y_gap = asymmetric_attractor(a_gap)
    ax.annotate(
        "",
        xy=(a_gap, x_gap),
        xytext=(a_gap, y_gap),
        arrowprops={"arrowstyle": "<->", "color": PURPLE, "lw": 1.5},
    )
    ax.text(
        a_gap + 0.07,
        (x_gap + y_gap) / 2,
        "symmetry\ngap",
        fontsize=7.5,
        ha="left",
        va="center",
        color="#6c3483",
    )
    ax.text(
        1.0,
        1.022,
        r"OSCILLATION  $(0<\alpha<2)$",
        fontsize=9,
        ha="center",
        weight="bold",
        color="#1a5276",
    )
    ax.text(
        3.5,
        1.022,
        r"SYMMETRY BREAKING  $(\alpha>2)$",
        fontsize=9,
        ha="center",
        weight="bold",
        color="#9c640c",
    )
    ax.text(
        1.32,
        0.71,
        "4-cycle: the maxima\nexchange two values",
        fontsize=7.5,
        ha="center",
        color="#0e6655",
    )
    ax.text(
        3.7,
        0.885,
        r"winner  $x^*=1-\frac{1}{\alpha(\alpha-1)^2}$",
        fontsize=8,
        ha="center",
        color=BLUE,
    )
    ax.text(
        4.0,
        0.60,
        r"loser  $y^*=1-\frac{1}{\alpha}$" + "\n(= saddle value)",
        fontsize=8,
        ha="center",
        color=RED,
    )
    ax.annotate(
        "symmetric state (saddle)",
        xy=(0.8, x_sym(0.8)),
        xytext=(0.12, 0.525),
        fontsize=7,
        ha="left",
        color="#707b7c",
        arrowprops={"arrowstyle": "->", "color": GREY, "lw": 0.8},
    )
    ax.annotate(
        r"collapse: $x^t, y^t\to\frac{1}{2}$",
        xy=(2, 0.5),
        xytext=(2.2, 0.54),
        fontsize=8,
        weight="bold",
        arrowprops={"arrowstyle": "->", "color": "k", "lw": 0.8},
    )
    ax.set_xlabel(r"external-resource parameter $\alpha$")
    ax.set_ylabel(r"asymptotic maxima $x^t=p^t_{\max},\ y^t=r^t_{\max}$")
    ax.set_title(r"Bifurcation diagram: the two branches merge at $\alpha=2$ and split apart")
    ax.set_xlim(0, 5)
    ax.set_ylim(0.47, 1.05)
    return fig


# --------------------------------------------------------------------------- Fig. 3


def fig_bif_mech(options: FigureOptions | None = None) -> Figure:
    """Fig. 3: the symmetric state and the two-step contraction of the asymmetric attractor."""
    del options
    br: BifurcationBranches = bifurcation_branches()
    fig = new_figure(TEXT_WIDTH, 2.75)
    a1, a2 = fig.subplots(1, 2)
    a1.plot(
        br.alphas_below, br.sym_below, "-", c=TEAL, lw=2.0, label="switching branch (period 2)"
    )
    a1.plot(
        br.alphas_above, br.sym_above, "-", c=BLUE, lw=2.0, label="preserving branch (period 1)"
    )
    a1.axvline(2, color="k", lw=0.8)
    a1.axhline(0.5, color="gray", lw=0.6, ls=":")
    a1.plot(2, 0.5, "ko", ms=5)
    a1.annotate(
        "border collision\n" + r"(kink at $\alpha=2$)",
        xy=(2, 0.5),
        xytext=(2.45, 0.6),
        fontsize=7.5,
        arrowprops={"arrowstyle": "->", "color": "gray", "lw": 0.8},
    )
    a1.text(0.75, 0.78, r"$\alpha x<1$", fontsize=8, color=TEAL)
    a1.text(3.5, 0.68, r"$\alpha x>1$", fontsize=8, color=BLUE)
    a1.set_xlabel(r"$\alpha$")
    a1.set_ylabel(r"$x_{\mathrm{sym}}(\alpha)$")
    a1.set_title(r"Symmetric state: V-shape, minimum at $\alpha=2$")
    a1.set_xlim(0, 5)
    a1.set_ylim(0.47, 1.0)
    a1.legend(loc="upper center", framealpha=0.95)
    _grid(a1)

    _spans(a2, 2, 5)
    a2.plot(br.alphas_below, br.cycle_modulus, "-", c=TEAL, lw=2.0, label=r"$1-x^*y^*$ (4-cycle)")
    a2.plot(
        br.alphas_above,
        br.rho2_above,
        "-",
        c=PURPLE,
        lw=2.0,
        label=r"$\rho^2(\alpha)$ (winner-loser point)",
    )
    a2.axhline(0.75, color="k", lw=0.6, ls=":")
    a2.plot(2, 0.75, "o", ms=6, mfc="white", mec="k", mew=1.2, zorder=5)
    a2.annotate(
        r"$3/4$ from both sides",
        xy=(2, 0.75),
        xytext=(2.5, 0.83),
        fontsize=7.5,
        arrowprops={"arrowstyle": "->", "color": "gray", "lw": 0.8},
    )
    a2.set_xlabel(r"$\alpha$")
    a2.set_ylabel("contraction over two steps")
    a2.set_title("Two-step contraction of the attractor")
    a2.set_xlim(0, 5)
    a2.set_ylim(0, 0.95)
    a2.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout(w_pad=1.5)
    return fig


# --------------------------------------------------------------------------- Figs. 4, 5

ASYMMETRIC_START: Final = ((0.81, 0.19), (0.11, 0.89))
SYMMETRIC_START: Final = ((0.81, 0.19), (0.81, 0.19))
MIRROR_START: Final = ((0.81, 0.19), (0.19, 0.81))


def _coordinate_axes(ax: Axes, ylabel: str, title: str) -> None:
    ax.axvline(2.0, color="k", ls="--", lw=1.0, label=r"$\alpha=2$ (bifurcation)")
    ax.axhline(0.5, color="gray", ls=":", lw=0.6)
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _grid(ax, dotted=True)


def _pixel_scatter(ax: Axes, x: FloatArray, y: FloatArray, colour: str) -> None:
    # The pixel marker of the published figure is kept; Matplotlib draws it as a small
    # square on scatter plots and warns about that.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The pixel maker", category=UserWarning)
        ax.scatter(x, y, s=0.8, c=colour, marker=",", lw=0)


def fig_bif_coord(options: FigureOptions | None = None) -> Figure:
    """Fig. 4: coordinate-wise bifurcation diagram of the full system, asymmetric start."""
    options = options or FigureOptions()
    alphas = options.alphas
    p_first, r_first = coordinate_diagram(*ASYMMETRIC_START, alphas)
    fig = new_figure(TEXT_WIDTH, 2.7)
    a1, a2 = fig.subplots(1, 2)
    _pixel_scatter(a1, np.repeat(alphas, p_first.shape[1]), p_first.ravel(), BLUE)
    _pixel_scatter(a2, np.repeat(alphas, r_first.shape[1]), r_first.ravel(), RED)
    _coordinate_axes(a1, r"$p_1$", r"coordinate $p_1$")
    _coordinate_axes(a2, r"$r_1$", r"coordinate $r_1$")
    a1.legend(loc="upper right", framealpha=0.95)
    fig.suptitle(
        r"Asymmetric start $\mathbf{p}^0=(0.81,0.19),\ \mathbf{r}^0=(0.11,0.89)$",
        fontsize=9.5,
        y=0.97,
    )
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.97))
    return fig


def fig_bif_coord_sym_mirror(options: FigureOptions | None = None) -> Figure:
    """Fig. 5: the same diagram for the symmetric and the mirror initial states."""
    options = options or FigureOptions()
    alphas = options.alphas
    panels = (
        (
            coordinate_diagram(*SYMMETRIC_START, alphas),
            r"symmetric start $\mathbf{p}^0=\mathbf{r}^0$",
        ),
        (coordinate_diagram(*MIRROR_START, alphas), r"mirror start $p^0_1=r^0_2$"),
    )
    fig = new_figure(TEXT_WIDTH, 2.7)
    axs = fig.subplots(1, 2)
    for ax, ((p_first, r_first), title) in zip(axs, panels, strict=True):
        ax.scatter(
            np.repeat(alphas, p_first.shape[1]),
            p_first.ravel(),
            s=1.2,
            c=BLUE,
            lw=0,
            label=r"$p_1$",
        )
        ax.scatter(
            np.repeat(alphas, r_first.shape[1])[::3],
            r_first.ravel()[::3],
            s=3,
            marker="x",
            c=RED,
            lw=0.4,
            label=r"$r_1$",
        )
        _coordinate_axes(ax, r"$p_1,\ r_1$", title)
    axs[0].legend(loc="lower right", framealpha=0.95, markerscale=3)
    fig.tight_layout(w_pad=1.5)
    return fig


# --------------------------------------------------------------------------- Fig. 6

PHASE_COLOURS: Final = ("#aed6f1", "#f5b7b1", "#d5d8dc")
"""Light blue: phase of (x*, y*); light red: phase of (y*, x*); grey: not converged."""
STAR_RESOLUTION: Final = 0.1
"""Stars are drawn if the cycle points are this fraction of the window apart."""


def _phase_panel(ax: Axes, window: Window, n: int) -> None:
    a, lo, hi = window.alpha, window.lo, window.hi
    grid_x, grid_y = offset_grid(lo, hi, n)
    phases = classify_phases(grid_x, grid_y, a, max_steps=PHASE_STEPS, tol=PHASE_TOL)
    colour_index = np.where(phases < 0, 2, phases)
    dx, dy = grid_x[0, 1] - grid_x[0, 0], grid_y[1, 0] - grid_y[0, 0]
    # interpolation="none": the PDF embeds exactly the computed array
    ax.imshow(
        colour_index,
        cmap=ListedColormap(list(PHASE_COLOURS)),
        vmin=-0.5,
        vmax=2.5,
        origin="lower",
        interpolation="none",
        extent=(
            grid_x[0, 0] - dx / 2,
            grid_x[0, -1] + dx / 2,
            grid_y[0, 0] - dy / 2,
            grid_y[-1, 0] + dy / 2,
        ),
        aspect="auto",
    )
    ax.plot([lo, hi], [lo, hi], "k:", lw=0.8)  # diagonal
    xs = np.linspace(lo, hi, 2001)
    ys = curve_s(xs, a)
    inside = (ys > lo) & (ys < hi)
    ax.plot(xs[inside], ys[inside], "k--", lw=0.9)
    u, v = exchange_cycle(a)
    if lo < v and u < hi and (u - v) / (hi - lo) > STAR_RESOLUTION:
        ax.plot(u, v, "*", ms=9, c=BLUE, mec="k", mew=0.5, clip_on=False, zorder=5)
        ax.plot(v, u, "*", ms=9, c=RED, mec="k", mew=0.5, clip_on=False, zorder=5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ticks = [lo, (lo + hi) / 2, hi]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.set_xlabel(r"$x^0$")
    ax.set_ylabel(r"$y^0$", labelpad=1)
    window_text = "" if (lo, hi) == (0.5, 1.0) else rf", $[{lo:g},{hi:g}]^2$"
    ax.set_title(rf"$\alpha={a:g}$" + window_text)


def fig_basins(options: FigureOptions | None = None) -> Figure:
    """Fig. 6: phases of the exchange 2-cycle on the whole square and near the corners."""
    options = options or FigureOptions()
    fig = new_figure(TEXT_WIDTH, 4.45)
    axs = fig.subplots(2, 3)
    for ax, window in zip(axs.ravel(), PHASE_PANELS, strict=True):
        _phase_panel(ax, window, options.phase_grid_n)
    fig.tight_layout(w_pad=0.6, h_pad=0.6)
    return fig


# --------------------------------------------------------------------------- Figs. 7, 8

TRAJECTORY_OFFSETS: Final = ((0.03, 0.018), (-0.03, 0.015), (0.025, -0.012))
TRAJECTORY_STARTS: Final = (
    (0.95, 0.60),
    (0.62, 0.97),
    (0.80, 0.55),
    (0.56, 0.70),
    (0.70, 0.66),
    (0.66, 0.71),
)
TIME_SERIES_START: Final = (0.80, 0.62)


def fig_trajectories(options: FigureOptions | None = None) -> Figure:
    """Fig. 7: iterates of Phi^2 near the 4-cycle (alpha = 1.5) and of Phi (alpha = 2.5)."""
    del options
    fig = new_figure(TEXT_WIDTH, 3.25)
    axs = fig.subplots(1, 2)
    a = 1.5
    xc, yc = exchange_cycle(a)
    ax = axs[0]
    for colour, (dx, dy) in zip((TEAL, BLUE, RED), TRAJECTORY_OFFSETS, strict=True):
        x, y = xc + dx, yc + dy
        xs, ys = [x], [y]
        for _ in range(25):
            x, y = phi_point(*phi_point(x, y, a), a)
            xs.append(x)
            ys.append(y)
        ax.plot(xs, ys, "-", c=colour, lw=0.6, alpha=0.8)
        ax.plot(xs, ys, "o", ms=3, c=colour, mew=0)
        ax.plot(xs[0], ys[0], "s", ms=5, mfc="white", mec="k", mew=0.8)
    ax.plot(xc, yc, "*", ms=11, c="k")
    ax.set_xlim(xc - 0.045, xc + 0.045)
    ax.set_ylim(yc - 0.03, yc + 0.03)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"$\alpha=1.5$: iterates of $\Phi^2$ near $(x^*,y^*)$")
    _grid(ax)

    a = 2.5
    ax = axs[1]
    u, v = asymmetric_attractor(a)
    for x0, y0 in TRAJECTORY_STARTS:
        path = orbit(x0, y0, a, 30)
        colour = BLUE if x0 > y0 else RED
        ax.plot(path[:, 0], path[:, 1], "-", c=colour, lw=0.8)
        ax.plot(path[:, 0], path[:, 1], "o", ms=2.5, c=colour, mew=0)
        ax.plot(x0, y0, "s", ms=5, mfc="white", mec="k", mew=0.8)
    ax.plot([u, v], [v, u], "*", ms=11, c="k", ls="none")
    ax.plot(x_sym(a), x_sym(a), "D", ms=5, mfc="white", mec="k", mew=0.8)
    ax.plot([0.5, 1], [0.5, 1], "k:", lw=0.9)
    ax.set_xlim(0.5, 1)
    ax.set_ylim(0.5, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"$\alpha=2.5$: iterates of $\Phi$")
    _grid(ax)
    fig.tight_layout(w_pad=1.5)
    return fig


def fig_time_series(options: FigureOptions | None = None) -> Figure:
    """Fig. 8: time series of the maxima from (0.80, 0.62) for alpha = 1.5 and 2.5."""
    del options
    fig = new_figure(TEXT_WIDTH, 2.5)
    axs = fig.subplots(1, 2)
    panels = (
        (axs[0], 1.5, r"$\alpha=1.5$: exchange of values (4-cycle)"),
        (axs[1], 2.5, r"$\alpha=2.5$: winner and loser"),
    )
    for ax, a, title in panels:
        path = orbit(*TIME_SERIES_START, a, 40)
        ax.plot(path[:, 0], "-o", ms=3, lw=1.0, c=BLUE, label=r"$x^t=p^t_{\max}$")
        ax.plot(path[:, 1], "-s", ms=3, lw=1.0, c=RED, label=r"$y^t=r^t_{\max}$")
        ax.set_xlabel(r"$t$")
        ax.set_ylim(0.5, 1.0)
        ax.set_title(title)
        _grid(ax)
    axs[0].legend(loc="upper right", framealpha=0.95)
    fig.tight_layout(w_pad=1.5)
    return fig


# --------------------------------------------------------------------------- Fig. 9

COLLAPSE_STEPS: Final = 120
COLLAPSE_LINESTYLES: Final = ("-", "--", "-.", ":", "-")


def fig_collapse(options: FigureOptions | None = None) -> Figure:
    """Fig. 9: geometric collapse at alpha = 2 for five random initial points."""
    del options
    if len(COLLAPSE_LINESTYLES) != FIGURE_TRAJECTORIES:
        raise RuntimeError("one line style per trajectory of Fig. 9 is required")
    starts = figure_initial_points()
    deviation = collapse_deviation(starts[:, 0], starts[:, 1], COLLAPSE_STEPS)
    t = np.arange(COLLAPSE_STEPS)
    fig = new_figure(TEXT_WIDTH, 2.75)
    a1, a2 = fig.subplots(1, 2)
    for k, linestyle in enumerate(COLLAPSE_LINESTYLES):
        g = deviation[:, k]
        a1.semilogy(t, g, linestyle, c=PURPLE, lw=1.1, alpha=0.85)
        a2.plot(t[:-2], g[2:] / g[:-2], linestyle, c=PURPLE, lw=1.1, alpha=0.85)
    a1.semilogy(t, 0.5 * 0.75 ** (t / 2), "-", c=GREY, lw=0.9)
    a1.annotate(
        r"$\propto(3/4)^{t/2}$",
        xy=(80, 0.5 * 0.75**40),
        xytext=(86, 3e-4),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "gray", "lw": 0.8},
    )
    a1.set_xlabel(r"$t$")
    a1.set_ylabel(r"$G_t-\frac{1}{2}$")
    a1.set_title(r"$\alpha=2$: geometric collapse to $(\frac{1}{2},\frac{1}{2})$")
    _grid(a1)
    a2.axhline(0.75, color="k", lw=0.7, ls=":")
    a2.text(78, 0.77, r"$3/4$", fontsize=8.5)
    a2.set_ylim(0.3, 1.02)
    a2.set_xlabel(r"$t$")
    a2.set_ylabel(r"$(G_{t+2}-\frac{1}{2})\,/\,(G_t-\frac{1}{2})$")
    a2.set_title("Ratio over two steps")
    _grid(a2)
    fig.tight_layout(w_pad=1.5)
    return fig


# --------------------------------------------------------------------------- registry


@dataclass(frozen=True, slots=True)
class FigureSpec:
    """A figure of the paper."""

    name: str
    """Output file stem (``<name>.pdf`` and ``<name>.png``)."""
    number: int
    """Figure number in the paper."""
    description: str
    build: Callable[[FigureOptions | None], Figure]


FIGURES: Final[tuple[FigureSpec, ...]] = (
    FigureSpec(
        "fig_switching_curves", 1, "switching curves and branch regions", fig_switching_curves
    ),
    FigureSpec("fig_bif_main", 2, "bifurcation diagram of the maxima", fig_bif_main),
    FigureSpec("fig_bif_mech", 3, "symmetric branch, two-step contraction", fig_bif_mech),
    FigureSpec("fig_bif_coord", 4, "coordinate-wise diagram, asymmetric start", fig_bif_coord),
    FigureSpec(
        "fig_bif_coord_sym_mirror",
        5,
        "coordinate-wise diagram, symmetric and mirror starts",
        fig_bif_coord_sym_mirror,
    ),
    FigureSpec("fig_basins", 6, "phases of the exchange 2-cycle", fig_basins),
    FigureSpec("fig_trajectories", 7, "trajectories in the plane", fig_trajectories),
    FigureSpec("fig_time_series", 8, "time series of the maxima", fig_time_series),
    FigureSpec("fig_collapse", 9, "collapse at alpha = 2", fig_collapse),
)


def build_and_save(spec: FigureSpec, out_dir: Path, options: FigureOptions) -> list[Path]:
    """Build one figure in the paper's style and save it as PDF and PNG."""
    with figure_style():
        fig = spec.build(options)
        paths = save_figure(fig, out_dir, spec.name)
    logger.info("Fig. %d written: %s", spec.number, ", ".join(p.name for p in paths))
    return paths
