"""Shared test configuration."""

from __future__ import annotations

from decimal import Decimal, getcontext

from hypothesis import HealthCheck, settings

settings.register_profile(
    "default", max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile("default")


def cycle_root_decimal(alpha: str, digits: int = 50) -> Decimal:
    """Root in (1/2, 1) of the cubic of Theorem 1, by bisection in decimal arithmetic."""
    getcontext().prec = digits
    a = Decimal(alpha)
    lo, hi = Decimal("0.5"), Decimal(1)

    def cubic(x: Decimal) -> Decimal:
        return a**3 * x**3 + a * x**2 + (1 - a) * x - 1

    for _ in range(4 * digits):
        mid = (lo + hi) / 2
        if cubic(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
