"""Quadrupolar tidal coupling for the primary hyperfine transition."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammainc, gammaincc

from .hydrogen import primary_radial_integrals
from .models import PRIMARY_FINAL, PRIMARY_INITIAL, State
from .validation import validate_alpha_chi, validate_q

PRIMARY_ANGULAR_PREFACTOR = -3.0 / 10.0
PRIMARY_DELTA_M = PRIMARY_FINAL.m - PRIMARY_INITIAL.m


def _validate_primary_transition(initial: State, final: State) -> None:
    pair = {initial, final}
    if pair != {PRIMARY_INITIAL, PRIMARY_FINAL}:
        raise NotImplementedError(
            "the baseline tidal implementation is restricted to 211 <-> 21-1"
        )


def energy_gap_M(
    alpha: float,
    chi: float,
    initial: State = PRIMARY_INITIAL,
    final: State = PRIMARY_FINAL,
) -> float:
    validate_alpha_chi(alpha, chi)
    _validate_primary_transition(initial, final)
    # Only the m-dependent O(alpha^5) bracket term survives this difference.
    return (final.m - initial.m) * chi * alpha**6 / 12.0


def resonance_frequency_M(
    alpha: float,
    chi: float,
    initial: State = PRIMARY_INITIAL,
    final: State = PRIMARY_FINAL,
) -> float:
    """Return positive M*Omega_res for the specified orientation."""
    validate_alpha_chi(alpha, chi)
    _validate_primary_transition(initial, final)
    delta_m = final.m - initial.m
    frequency = energy_gap_M(alpha, chi, initial, final) / delta_m
    if frequency <= 0.0:
        raise ValueError("the selected state ordering does not give a positive resonance")
    return float(frequency)


def orbital_radius_M(omega_M: float, q: float | np.ndarray) -> float | np.ndarray:
    if omega_M <= 0.0:
        raise ValueError("omega_M must be positive")
    q_array = np.asarray(q, dtype=float)
    if np.any(q_array <= 0.0):
        raise ValueError("q must be positive")
    result = ((1.0 + q_array) / omega_M**2) ** (1.0 / 3.0)
    if np.ndim(q) == 0:
        return float(result)
    return result


def cloud_radius_M(alpha: float, n: int = 2) -> float:
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    return n * n / alpha**2


def tidal_eta_M_array(
    alpha: float,
    chi: float,
    q: np.ndarray,
    initial: State = PRIMARY_INITIAL,
    final: State = PRIMARY_FINAL,
) -> np.ndarray:
    """Vectorized full piecewise M*eta at resonance."""
    omega = resonance_frequency_M(alpha, chi, initial, final)
    return tidal_eta_at_omega_M_array(
        alpha, chi, q, omega, initial=initial, final=final
    )


def tidal_eta_at_omega_M_array(
    alpha: float,
    chi: float,
    q: np.ndarray,
    omega_M: float,
    initial: State = PRIMARY_INITIAL,
    final: State = PRIMARY_FINAL,
) -> np.ndarray:
    """Vectorized full piecewise M*eta at an arbitrary orbital frequency."""
    validate_alpha_chi(alpha, chi)
    _validate_primary_transition(initial, final)
    if omega_M <= 0.0:
        raise ValueError("omega_M must be positive")
    q_array = np.asarray(q, dtype=float)
    if np.any(q_array <= 0.0):
        raise ValueError("q must be positive")
    omega = float(omega_M)
    radius = orbital_radius_M(omega, q_array)
    x_star = alpha**2 * radius
    i_inner, i_outer = primary_radial_integrals(x_star)
    inner = (
        q_array
        / (1.0 + q_array)
        * omega
        / alpha**3
        * PRIMARY_ANGULAR_PREFACTOR
        * i_inner
    )
    outer = (
        q_array
        * (1.0 + q_array) ** (2.0 / 3.0)
        * alpha**7
        / omega ** (7.0 / 3.0)
        * PRIMARY_ANGULAR_PREFACTOR
        * i_outer
    )
    return omega * np.abs(inner + outer)


def tidal_eta_M(
    alpha: float,
    chi: float,
    q: float,
    initial: State = PRIMARY_INITIAL,
    final: State = PRIMARY_FINAL,
) -> float:
    validate_q(q)
    return float(
        tidal_eta_M_array(
            alpha, chi, np.asarray([q]), initial=initial, final=final
        )[0]
    )


def tidal_eta_at_omega_M(
    alpha: float,
    chi: float,
    q: float,
    omega_M: float,
    initial: State = PRIMARY_INITIAL,
    final: State = PRIMARY_FINAL,
) -> float:
    validate_alpha_chi(alpha, chi)
    validate_q(q)
    _validate_primary_transition(initial, final)
    if omega_M <= 0.0:
        raise ValueError("omega_M must be positive")
    return _tidal_eta_at_omega_M_unchecked(alpha, chi, q, omega_M)


def _tidal_eta_at_omega_M_unchecked(
    alpha: float,
    chi: float,
    q: float,
    omega_M: float,
) -> float:
    """Fast scalar kernel for ODE callbacks after one-time validation."""
    radius = ((1.0 + q) / omega_M**2) ** (1.0 / 3.0)
    x_star = alpha * alpha * radius
    if x_star > 50.0:
        i_inner = 30.0
        i_outer = 0.0
    else:
        i_inner = 30.0 * float(gammainc(7.0, x_star))
        i_outer = float(gammaincc(2.0, x_star)) / 24.0
    inner = (
        q
        / (1.0 + q)
        * omega_M
        / alpha**3
        * PRIMARY_ANGULAR_PREFACTOR
        * i_inner
    )
    outer = (
        q
        * (1.0 + q) ** (2.0 / 3.0)
        * alpha**7
        / omega_M ** (7.0 / 3.0)
        * PRIMARY_ANGULAR_PREFACTOR
        * i_outer
    )
    return omega_M * math.fabs(inner + outer)


def tidal_eta_far_M(alpha: float, chi: float, q: float) -> float:
    """Analytic R >> r0 limit for regression checks."""
    validate_alpha_chi(alpha, chi)
    validate_q(q)
    return chi**2 * alpha**9 / 16.0 * q / (1.0 + q)
