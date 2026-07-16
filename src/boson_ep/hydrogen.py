"""Hydrogenic wave functions and radial integrals."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.special import eval_genlaguerre, gamma, gammainc, gammaincc


def radial_dimensionless(n: int, l: int, x: float | np.ndarray) -> float | np.ndarray:
    """Return r0^(3/2) R_nl at x=r/r0."""
    if n < 1 or not 0 <= l < n:
        raise ValueError("hydrogenic radial state requires n >= 1 and 0 <= l < n")
    x_array = np.asarray(x, dtype=float)
    normalization = math.sqrt(
        (2.0 / n) ** 3
        * math.factorial(n - l - 1)
        / (2.0 * n * math.factorial(n + l))
    )
    result = (
        normalization
        * np.exp(-x_array / n)
        * (2.0 * x_array / n) ** l
        * eval_genlaguerre(n - l - 1, 2 * l + 1, 2.0 * x_array / n)
    )
    if np.ndim(x) == 0:
        return float(result)
    return result


def radial_normalization(n: int, l: int) -> tuple[float, float]:
    value, error = quad(
        lambda x: x * x * float(radial_dimensionless(n, l, x)) ** 2,
        0.0,
        np.inf,
        epsabs=1.0e-13,
        epsrel=1.0e-13,
        limit=300,
    )
    return float(value), float(error)


def primary_radial_integrals(
    x_star: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Exact incomplete-gamma radial pieces for 211 <-> 21-1 and l*=2."""
    x_array = np.asarray(x_star, dtype=float)
    if np.any(x_array <= 0.0):
        raise ValueError("x_star must be positive")
    i_inner = gamma(7.0) * gammainc(7.0, x_array) / 24.0
    i_outer = gamma(2.0) * gammaincc(2.0, x_array) / 24.0
    if np.ndim(x_star) == 0:
        return float(i_inner), float(i_outer)
    return i_inner, i_outer


def primary_radial_integrals_quad(x_star: float) -> tuple[float, float]:
    radial_sq = lambda x: float(radial_dimensionless(2, 1, x)) ** 2
    inner, _ = quad(
        lambda x: x**4 * radial_sq(x),
        0.0,
        x_star,
        epsabs=1.0e-30,
        epsrel=1.0e-13,
        limit=300,
    )
    outer, _ = quad(
        lambda x: x ** (-1) * radial_sq(x),
        x_star,
        np.inf,
        epsabs=1.0e-30,
        epsrel=1.0e-13,
        limit=300,
    )
    return float(inner), float(outer)
