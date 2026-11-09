"""Shared figure style: colours, Matplotlib parameters, figure creation and saving.

Nothing here changes the global Matplotlib state: the style is applied with
:func:`figure_style` (a context manager), and figures are created without
:mod:`matplotlib.pyplot`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any, Final

from matplotlib import font_manager
from matplotlib import style as mpl_style
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

__all__ = [
    "BLUE",
    "GREY",
    "PNG_DPI",
    "PURPLE",
    "RED",
    "SPAN_LEFT",
    "SPAN_RIGHT",
    "TEAL",
    "TEXT_WIDTH",
    "add_panel_letters",
    "figure_style",
    "new_figure",
    "save_figure",
]

logger = logging.getLogger(__name__)

# Meaning-carrying colours.
TEAL: Final = "#117a65"
"""Oscillatory regime (both maxima visit both values in time)."""
BLUE: Final = "#1f618d"
"""Vector p / winner branch."""
RED: Final = "#c0392b"
"""Vector r / loser branch."""
GREY: Final = "#909497"
"""Symmetric state (saddle)."""
PURPLE: Final = "#8e44ad"
"""Multipliers and contraction rates."""
SPAN_LEFT: Final = "#eaf2f8"
"""Background of the range alpha < 2."""
SPAN_RIGHT: Final = "#fdf2e9"
"""Background of the range alpha > 2."""

TEXT_WIDTH: Final = 6.5
"""Full text width of the paper, in inches."""

PNG_DPI: Final = 200
"""Resolution of the PNG previews."""

PAD_INCHES: Final = 0.03
"""Padding of the tight bounding box."""

PREFERRED_FONT: Final = "Inter"
"""Font of the published figures (the default of Matplotlib 3.11 when it is installed)."""

FALLBACK_FONT: Final = "DejaVu Sans"
"""Font shipped with Matplotlib, used when :data:`PREFERRED_FONT` is not installed."""

_BASE_STYLE: Final[dict[str, Any]] = {
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "mathtext.fontset": "dejavusans",
}


@cache
def _font_family() -> tuple[str, ...]:
    """Font family list: the published font if available, else the bundled fallback."""
    try:
        font_manager.findfont(
            font_manager.FontProperties(family=PREFERRED_FONT), fallback_to_default=False
        )
    except ValueError:
        logger.warning(
            "font %r not found; using %r (figures will differ slightly from the published ones)",
            PREFERRED_FONT,
            FALLBACK_FONT,
        )
        return (FALLBACK_FONT,)
    return (PREFERRED_FONT, FALLBACK_FONT)


@contextmanager
def figure_style() -> Iterator[None]:
    """Apply the style of the paper's figures (on top of Matplotlib's default style)."""
    style = dict(_BASE_STYLE, **{"font.family": list(_font_family())})
    with mpl_style.context(["default", style]):
        yield


def new_figure(width: float, height: float) -> Figure:
    """Create a figure with an Agg canvas (no :mod:`matplotlib.pyplot` state is involved)."""
    fig = Figure(figsize=(width, height))
    FigureCanvasAgg(fig)
    return fig


def add_panel_letters(fig: Figure) -> None:
    """Prefix the titles with ``(a)``, ``(b)``, ... if more than one axes has a title."""
    titled = [ax for ax in fig.axes if ax.get_title()]
    if len(titled) > 1:
        for letter, ax in zip("abcdefghij", titled, strict=False):
            ax.set_title(f"({letter}) " + ax.get_title())


def save_figure(fig: Figure, out_dir: Path, name: str, *, png_dpi: int = PNG_DPI) -> list[Path]:
    """Save a figure as vector PDF and PNG preview with deterministic metadata.

    The PDF carries no creation date, so that repeated runs give byte-identical files.

    Returns
    -------
    list of pathlib.Path
        The written files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    add_panel_letters(fig)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    common: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": PAD_INCHES}
    fig.savefig(pdf, metadata={"CreationDate": None}, **common)
    fig.savefig(png, dpi=png_dpi, **common)
    return [pdf, png]
