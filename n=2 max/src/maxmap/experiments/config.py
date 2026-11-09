"""Configurations of the numerical experiments of Section 5 (defaults: the values of the paper)."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "PAPER_SEED",
    "PHASE_PANELS",
    "BoundaryConfig",
    "ChecksConfig",
    "CollapseConfig",
    "CornerConfig",
    "ExtendedConfig",
    "GridConfig",
    "PeriodicSearchConfig",
    "RandomConvergenceConfig",
    "SelfSimilarityConfig",
    "UncertaintyConfig",
    "Window",
]

PAPER_SEED = 2026
"""Seed of :func:`numpy.random.default_rng` used for the published checks."""


@dataclass(frozen=True, slots=True)
class Window:
    """A square window ``[lo, hi]^2`` of initial points at a value of the parameter."""

    alpha: float
    lo: float
    hi: float

    def __post_init__(self) -> None:
        """Validate the window."""
        if not 0.5 <= self.lo < self.hi <= 1.0:
            raise ValueError(f"invalid window [{self.lo}, {self.hi}]")
        if not self.alpha > 0.0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")

    @property
    def label(self) -> str:
        """Text such as ``[0.99,1]^2`` (as in the published report)."""
        return f"[{self.lo:g},{self.hi:g}]^2"


PHASE_PANELS: tuple[Window, ...] = (
    Window(1.5, 0.5, 1.0),
    Window(1.9, 0.5, 1.0),
    Window(1.99, 0.5, 1.0),
    Window(1.99, 0.99, 1.0),
    Window(1.99, 0.5, 0.55),
    Window(1.99, 0.5, 0.505),
)
"""The six panels (a)--(f) of Fig. 6."""


@dataclass(frozen=True, slots=True)
class RandomConvergenceConfig:
    """Part 1: convergence of random initial points to the asymmetric attractor."""

    alphas: tuple[float, ...] = (0.2, 0.5, 1.0, 1.5, 1.9, 1.99, 2.01, 2.1, 2.5, 3.0, 5.0, 10.0)
    n_points: int = 100_000
    max_steps: int = 6000
    tol: float = 1e-10


@dataclass(frozen=True, slots=True)
class GridConfig:
    """Part 2: phases on the regular grids of Fig. 6."""

    panels: tuple[Window, ...] = PHASE_PANELS
    n: int = 600
    max_steps: int = 4000
    tol: float = 1e-6


@dataclass(frozen=True, slots=True)
class BoundaryConfig:
    """Part 3: colour boundaries located by bisection."""

    per_panel: int = 200
    bisections: int = 50
    colour_steps: int = 1500
    approach_steps: int = 400
    max_diagonal_distance: float = 1e-12
    max_saddle_distance: float = 1e-6


@dataclass(frozen=True, slots=True)
class UncertaintyConfig:
    """Part 4: uncertainty exponent of the colour boundaries."""

    alphas: tuple[float, ...] = (1.5, 1.9, 1.99)
    n_points: int = 1_000_000
    eps: tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5)
    max_steps: int = 1000


@dataclass(frozen=True, slots=True)
class SelfSimilarityConfig:
    """Part 5: self-similarity near the corners of the square."""

    n: int = 400
    max_steps: int = 4000
    corner_one: tuple[Window, ...] = (
        Window(1.99, 0.9, 1.0),
        Window(1.99, 0.99, 1.0),
        Window(1.99, 0.999, 1.0),
    )
    corner_half: tuple[Window, ...] = (
        Window(1.99, 0.5, 0.505),
        Window(1.999, 0.5, 0.5005),
        Window(1.9999, 0.5, 0.50005),
    )


@dataclass(frozen=True, slots=True)
class CollapseConfig:
    """Extended check: rate of collapse at ``alpha = 2`` for many initial points."""

    n_points: int = 100_000
    times: tuple[int, ...] = (20, 40, 80, 120)


@dataclass(frozen=True, slots=True)
class PeriodicSearchConfig:
    """Extended check: Newton search for periodic orbits of the map of maxima."""

    alphas: tuple[float, ...] = (0.5, 1.0, 1.5, 1.9, 1.99, 2.5, 3.0, 5.0)
    max_period: int = 8
    starts: int = 20_000
    newton_steps: int = 80


@dataclass(frozen=True, slots=True)
class CornerConfig:
    """Extended checks near the corners at the resolution of Fig. 6."""

    n: int = 600
    corner_one_reference: Window = PHASE_PANELS[3]
    corner_one_windows: tuple[Window, ...] = (Window(1.99, 0.9, 1.0), Window(1.99, 0.999, 1.0))
    sector_deg: float = 10.0
    scaling_windows: tuple[Window, ...] = (PHASE_PANELS[4], Window(1.99999, 0.5, 0.505))
    scaling_factor: float = 0.75
    ray_alphas: tuple[float, ...] = (1.99, 1.999, 1.9999)
    ray_slopes: tuple[float, ...] = (0.1, 0.25, 0.4)
    ray_u_min: float = 1e-6
    ray_u_max: float = 0.2
    ray_samples: int = 20_000


@dataclass(frozen=True, slots=True)
class ExtendedConfig:
    """Additional checks, beyond those of the report ``convergence_check.txt``."""

    collapse: CollapseConfig = field(default_factory=CollapseConfig)
    periodic: PeriodicSearchConfig = field(default_factory=PeriodicSearchConfig)
    corners: CornerConfig = field(default_factory=CornerConfig)
    control_points: int = 10_000
    sign_rule_alphas: tuple[float, ...] = (0.2, 0.5, 1.0)
    sign_rule_n: int = 300
    figure_ratio_time: int = 80
    figure_factor_time: int = 100


@dataclass(frozen=True, slots=True)
class ChecksConfig:
    """All checks of Section 5.1.

    The random streams are ``default_rng(seed)`` for part 1, ``default_rng(seed + 1)`` for
    part 3 and ``default_rng(seed + 2)`` for part 4, as for the published results.
    """

    seed: int = PAPER_SEED
    random: RandomConvergenceConfig = field(default_factory=RandomConvergenceConfig)
    grids: GridConfig = field(default_factory=GridConfig)
    boundaries: BoundaryConfig = field(default_factory=BoundaryConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    selfsimilarity: SelfSimilarityConfig = field(default_factory=SelfSimilarityConfig)
    extended: ExtendedConfig = field(default_factory=ExtendedConfig)

    @classmethod
    def fast(cls, seed: int = PAPER_SEED) -> ChecksConfig:
        """Return a reduced configuration for smoke tests (not the paper's values)."""
        return cls(
            seed=seed,
            random=RandomConvergenceConfig(n_points=5_000),
            grids=GridConfig(n=120),
            boundaries=BoundaryConfig(per_panel=20),
            uncertainty=UncertaintyConfig(n_points=20_000),
            selfsimilarity=SelfSimilarityConfig(n=100),
            extended=ExtendedConfig(
                collapse=CollapseConfig(n_points=2_000),
                periodic=PeriodicSearchConfig(alphas=(1.5, 2.5), max_period=4, starts=500),
                corners=CornerConfig(n=100, ray_samples=2_000),
                control_points=1_000,
                sign_rule_n=60,
            ),
        )
