"""Orbital, cloud, and resonance timescale hierarchy."""

from __future__ import annotations

import math

from .constants import M_SECONDS_PER_SOLAR_MASS
from .models import PRIMARY_FINAL, PRIMARY_INITIAL, TimescaleResult
from .spectrum import gamma_detweiler_M
from .tides import (
    PRIMARY_DELTA_M,
    cloud_radius_M,
    energy_gap_M,
    orbital_radius_M,
    resonance_frequency_M,
    tidal_eta_M,
)
from .validation import validate_alpha_chi, validate_q


def gw_frequency_sweep_M2(
    omega_M: float,
    q: float,
    order: str = "newtonian",
) -> float:
    validate_q(q)
    if omega_M <= 0.0:
        raise ValueError("omega_M must be positive")
    leading = (
        96.0
        / 5.0
        * q
        / (1.0 + q) ** (1.0 / 3.0)
        * omega_M ** (11.0 / 3.0)
    )
    if order == "newtonian":
        return leading
    if order == "1pn":
        symmetric_mass_ratio = q / (1.0 + q) ** 2
        x_pn = ((1.0 + q) * omega_M) ** (2.0 / 3.0)
        correction = 1.0 - (
            743.0 / 336.0 + 11.0 * symmetric_mass_ratio / 4.0
        ) * x_pn
        if correction <= 0.0:
            raise ValueError("1PN chirp correction is non-positive")
        return leading * correction
    raise ValueError("order must be 'newtonian' or '1pn'")


def compute_timescales(
    alpha: float,
    chi: float,
    q: float,
    mass_msun: float,
) -> TimescaleResult:
    validate_alpha_chi(alpha, chi)
    validate_q(q)
    if not math.isfinite(mass_msun) or mass_msun <= 0.0:
        raise ValueError("mass_msun must be finite and positive")

    omega = resonance_frequency_M(alpha, chi)
    delta_e = energy_gap_M(alpha, chi)
    gamma_grow = gamma_detweiler_M(alpha, chi, PRIMARY_INITIAL)
    gamma_absorb = gamma_detweiler_M(alpha, chi, PRIMARY_FINAL)
    eta = tidal_eta_M(alpha, chi, q)
    domega = gw_frequency_sweep_M2(omega, q)
    delta_dot = abs(PRIMARY_DELTA_M) * domega
    radius = orbital_radius_M(omega, q)
    radius_ratio = radius / cloud_radius_M(alpha)
    times_M = {
        "T_orb": 2.0 * math.pi / omega,
        "T_split": 1.0 / abs(delta_e),
        "T_grow": 1.0 / gamma_grow if gamma_grow > 0.0 else math.inf,
        "T_abs": 1.0 / abs(gamma_absorb),
        "T_eta": 1.0 / eta,
        "T_GW": omega / domega,
        "T_sweep": 1.0 / math.sqrt(delta_dot),
        "T_width": 2.0 * eta / delta_dot,
    }
    mass_seconds = mass_msun * M_SECONDS_PER_SOLAR_MASS
    times_seconds = {name: value * mass_seconds for name, value in times_M.items()}
    hierarchy = tuple(
        name for name, _ in sorted(times_M.items(), key=lambda item: item[1])
    )
    return TimescaleResult(
        alpha=alpha,
        chi=chi,
        q=q,
        mass_msun=mass_msun,
        omega_res_M=omega,
        delta_e_M=delta_e,
        gamma_grow_M=gamma_grow,
        gamma_absorb_M=gamma_absorb,
        eta_M=eta,
        domega_M2=domega,
        landau_zener_z=eta * eta / delta_dot,
        radius_M=radius,
        radius_over_cloud=radius_ratio,
        times_M=times_M,
        times_seconds=times_seconds,
        hierarchy=hierarchy,
    )
