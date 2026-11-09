"""Command line interface: ``maxmap figures``, ``maxmap checks``, ``maxmap list``.

Examples
--------
::

    maxmap figures --out out                 # Figures 1-9 of the paper (about 7 s)
    maxmap figures --only fig_basins --fast  # a coarse preview of Fig. 6
    maxmap checks --out out                  # checks of Section 5.1 (about 45 s)
    maxmap checks --out out --extended       # plus additional checks (about 1 min more)
    maxmap checks --seed 1                   # the same checks with another random sample
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from maxmap import __version__
from maxmap.experiments.checks import (
    extended_report_files,
    format_extended_report,
    format_report,
    report_files,
    run_checks,
    run_extended_checks,
)
from maxmap.experiments.config import PAPER_SEED, ChecksConfig

__all__ = ["build_parser", "main"]

logger = logging.getLogger("maxmap")

DEFAULT_OUT = Path("out")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="maxmap",
        description="Figures and numerical checks for the map of maxima.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="debug output")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="warnings only")
    sub = parser.add_subparsers(dest="command", required=True)

    figures = sub.add_parser("figures", help="produce the figures of the paper (PDF and PNG)")
    figures.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    figures.add_argument(
        "--only", nargs="+", metavar="NAME", help="figure names (see 'maxmap list'); default all"
    )
    figures.add_argument(
        "--fast", action="store_true", help="coarse grids for previews (not the paper's figures)"
    )

    checks = sub.add_parser("checks", help="run the numerical checks of Section 5.1")
    checks.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    checks.add_argument(
        "--seed", type=int, default=PAPER_SEED, help=f"random seed (default {PAPER_SEED})"
    )
    checks.add_argument(
        "--n", type=int, default=None, help="random initial points per alpha in part 1"
    )
    checks.add_argument(
        "--fast", action="store_true", help="reduced sample sizes (smoke test, a few seconds)"
    )
    checks.add_argument("--extended", action="store_true", help="also run the additional checks")

    sub.add_parser("list", help="list the figures and their numbers in the paper")
    return parser


def _configure_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)
    logger.setLevel(level)  # also effective if the root logger is configured elsewhere
    logging.getLogger("matplotlib").setLevel(max(level, logging.WARNING))


def _cmd_figures(args: argparse.Namespace) -> int:
    # Lazy import: Matplotlib is needed only for the figures.
    from maxmap.plotting.figures import FIGURES, FigureOptions, build_and_save  # noqa: PLC0415

    selected = FIGURES
    if args.only:
        known = {spec.name: spec for spec in FIGURES}
        unknown = sorted(set(args.only) - set(known))
        if unknown:
            logger.error("unknown figure(s): %s (see 'maxmap list')", ", ".join(unknown))
            return 2
        selected = tuple(spec for spec in FIGURES if spec.name in set(args.only))
    options = FigureOptions.fast() if args.fast else FigureOptions()
    start = time.perf_counter()
    for spec in selected:
        build_and_save(spec, args.out, options)
    logger.info(
        "%d figure(s) written to %s in %.1f s",
        len(selected),
        args.out,
        time.perf_counter() - start,
    )
    return 0


def _cmd_checks(args: argparse.Namespace) -> int:
    config = ChecksConfig.fast(args.seed) if args.fast else ChecksConfig(seed=args.seed)
    if args.n is not None:
        config = replace(config, random=replace(config.random, n_points=args.n))
    report = run_checks(config)
    for line in format_report(report):
        logger.info("%s", line)
    path = report_files(report, args.out)
    logger.info("written to %s (%.0f s)", path, report.seconds)
    if args.extended:
        ext = run_extended_checks(config, report)
        for line in format_extended_report(ext, config):
            logger.info("%s", line)
        logger.info("written to %s", extended_report_files(ext, config, args.out))
    return 0 if report.passed else 1


def _cmd_list() -> int:
    from maxmap.plotting.figures import FIGURES  # noqa: PLC0415

    for spec in FIGURES:
        logger.info("Fig. %d  %-26s %s", spec.number, spec.name, spec.description)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``maxmap`` command; returns the exit status."""
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose, args.quiet)
    if args.command == "figures":
        return _cmd_figures(args)
    if args.command == "checks":
        return _cmd_checks(args)
    return _cmd_list()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
