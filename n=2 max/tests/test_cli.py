"""Command line interface and package hygiene."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from maxmap.cli import main


def test_import_has_no_side_effects() -> None:
    code = (
        "import sys, maxmap, maxmap.experiments.checks; "
        "assert 'matplotlib' not in sys.modules, 'matplotlib imported'; "
        "import maxmap.plotting.figures; "
        "assert 'matplotlib.pyplot' not in sys.modules, 'pyplot imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_checks_fast(tmp_path: Path) -> None:
    assert main(["-q", "checks", "--fast", "--out", str(tmp_path)]) == 0
    text = (tmp_path / "convergence_check.txt").read_text(encoding="utf-8")
    assert text.startswith("1. Random initial points: N = 5000 per alpha, seed = 2026")
    assert text.rstrip().endswith("All checks passed.")
    data = json.loads((tmp_path / "convergence_check.json").read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert len(data["random"]) == 12


def test_checks_fast_extended(tmp_path: Path) -> None:
    assert main(["-q", "checks", "--fast", "--extended", "--out", str(tmp_path)]) == 0
    text = (tmp_path / "convergence_check_extended.txt").read_text(encoding="utf-8")
    for header in ("A. Collapse", "D. Periodic orbits", "F. Corner (1,1)", "G. Scaling", "I. "):
        assert header in text
    assert "WARNING" not in text
    data = json.loads((tmp_path / "convergence_check_extended.json").read_text(encoding="utf-8"))
    assert len(data["figure_rates"]) == 5
    assert data["corners_converged"] is True


def test_checks_n_option(tmp_path: Path) -> None:
    assert main(["-q", "checks", "--fast", "--n", "300", "--out", str(tmp_path)]) == 0
    text = (tmp_path / "convergence_check.txt").read_text(encoding="utf-8")
    assert "N = 300 per alpha" in text


@pytest.mark.slow
def test_figures_fast_are_deterministic(tmp_path: Path) -> None:
    names = ["fig_time_series", "fig_basins"]
    for run in ("a", "b"):
        assert (
            main(["-q", "figures", "--fast", "--only", *names, "--out", str(tmp_path / run)]) == 0
        )
    for name in names:
        for suffix in (".pdf", ".png"):
            first = (tmp_path / "a" / f"{name}{suffix}").read_bytes()
            assert first == (tmp_path / "b" / f"{name}{suffix}").read_bytes()
        assert b"CreationDate" not in (tmp_path / "a" / f"{name}.pdf").read_bytes()


def test_unknown_figure_is_an_error(tmp_path: Path) -> None:
    assert main(["-q", "figures", "--only", "fig_nonexistent", "--out", str(tmp_path)]) == 2


def test_list_figures(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="maxmap")
    assert main(["list"]) == 0
    assert "Fig. 6" in caplog.text
    assert "fig_basins" in caplog.text
