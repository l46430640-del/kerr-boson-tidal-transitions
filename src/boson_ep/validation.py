"""Input validation shared across numerical modules."""

from __future__ import annotations

import math


def validate_alpha_chi(alpha: float, chi: float) -> None:
    if not math.isfinite(alpha) or not 0.0 < alpha <= 0.3:
        raise ValueError("alpha must lie in (0, 0.3] for this baseline")
    if not math.isfinite(chi) or not 0.0 < chi <= 0.999:
        raise ValueError("chi must lie in (0, 0.999]")


def validate_q(q: float) -> None:
    if not math.isfinite(q) or q <= 0.0:
        raise ValueError("q must be finite and positive")

