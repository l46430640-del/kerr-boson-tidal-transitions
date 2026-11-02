"""Hydrogenic real frequencies and corrected Detweiler rates."""

from __future__ import annotations

import math

from scipy.optimize import brentq

from .models import PRIMARY_INITIAL, State
from .validation import validate_alpha_chi


def horizon_radius_M(chi: float) -> float:
    if not 0.0 <= chi < 1.0:
        raise ValueError("horizon formulas require 0 <= chi < 1")
    return 1.0 + math.sqrt(1.0 - chi * chi)


def horizon_omega_M(chi: float) -> float:
    return chi / (2.0 * horizon_radius_M(chi))


def omega_real_M(alpha: float, chi: float, state: State) -> float:
    """Return M*omega_R through the spin-dependent O(alpha^5) bracket."""
    validate_alpha_chi(alpha, chi)
    n = state.n
    l = state.l
    bracket = 1.0 - alpha**2 / (2.0 * n**2) - alpha**4 / (8.0 * n**4)
    bracket += (2 * l - 3 * n + 1) * alpha**4 / (n**4 * (l + 0.5))
    if l > 0:
        bracket += (
            2.0
            * state.m
            * chi
            * alpha**5
            / (n**3 * l * (l + 0.5) * (l + 1))
        )
    return alpha * bracket


def detweiler_coefficient(alpha: float, chi: float, state: State) -> float:
    """Corrected small-alpha coefficient in the scalar Detweiler rate."""
    validate_alpha_chi(alpha, chi)
    n = state.n
    l = state.l
    numerator = 2 ** (4 * l + 1) * math.factorial(n + l)
    denominator = n ** (2 * l + 4) * math.factorial(n - l - 1)
    factorial_factor = (
        math.factorial(l)
        / (math.factorial(2 * l) * math.factorial(2 * l + 1))
    ) ** 2
    r_plus = horizon_radius_M(chi)
    product = 1.0
    for j in range(1, l + 1):
        product *= j * j * (1.0 - chi * chi) + (
            chi * state.m - 2.0 * r_plus * alpha
        ) ** 2
    return numerator / denominator * factorial_factor * product


def gamma_detweiler_M(alpha: float, chi: float, state: State) -> float:
    """Return M*Gamma with the explicit factor 2*r_+ convention."""
    r_plus = horizon_radius_M(chi)
    horizon_term = state.m * horizon_omega_M(chi) - omega_real_M(alpha, chi, state)
    return (
        2.0
        * r_plus
        * detweiler_coefficient(alpha, chi, state)
        * horizon_term
        * alpha ** (4 * state.l + 5)
    )


def saturation_spin_approx(alpha: float, state: State = PRIMARY_INITIAL) -> float:
    if state.m <= 0:
        raise ValueError("the approximate saturation spin requires m > 0")
    value = 4.0 * state.m * alpha / (state.m**2 + 4.0 * alpha**2)
    if value > 0.999:
        raise ValueError("approximate saturation spin exceeds the supported range")
    return value


def saturation_spin_numeric(alpha: float, state: State = PRIMARY_INITIAL) -> float:
    """Solve m*Omega_H=omega_R with the same truncated real spectrum."""
    if state.m <= 0:
        raise ValueError("the numerical saturation spin requires m > 0")
    if not 0.0 < alpha <= 0.3:
        raise ValueError("alpha must lie in (0, 0.3]")

    def residual(chi: float) -> float:
        return state.m * horizon_omega_M(chi) - omega_real_M(alpha, chi, state)

    return float(brentq(residual, 1.0e-12, 0.999, xtol=1.0e-14, rtol=1.0e-14))

