"""Pre-resonance cloud-formation history and saturation veto."""

from __future__ import annotations

import math

from scipy.optimize import brentq

from .constants import SOLAR_MASS_IN_PLANCK_MASS
from .models import FormationConfig, FormationResult, PRIMARY_INITIAL
from .spectrum import gamma_detweiler_M, saturation_spin_numeric
from .tides import resonance_frequency_M
from .validation import validate_alpha_chi, validate_q


def _growth_efolds_between(
    omega_low: float,
    omega_high: float,
    gamma_M: float,
    q: float,
) -> float:
    if not 0.0 < omega_low < omega_high:
        raise ValueError("frequency bounds must be positive and increasing")
    coefficient = 96.0 / 5.0 * q / (1.0 + q) ** (1.0 / 3.0)
    return float(
        3.0
        * gamma_M
        / (4.0 * coefficient)
        * (omega_low ** (-8.0 / 3.0) - omega_high ** (-8.0 / 3.0))
    )


def assess_formation_history(config: FormationConfig) -> FormationResult:
    """Apply the hard veto when a high-spin cloud saturates before resonance."""
    validate_alpha_chi(config.alpha, config.chi)
    validate_q(config.q)
    if config.mass_msun <= 0.0 or not math.isfinite(config.mass_msun):
        raise ValueError("mass_msun must be finite and positive")
    if not 0.0 < config.birth_frequency_fraction < 1.0:
        raise ValueError("birth_frequency_fraction must lie in (0, 1)")
    if config.seed_occupancy <= 0.0:
        raise ValueError("seed_occupancy must be positive")

    omega_res = resonance_frequency_M(config.alpha, config.chi)
    gamma = gamma_detweiler_M(config.alpha, config.chi, PRIMARY_INITIAL)
    chi_sat = saturation_spin_numeric(config.alpha, PRIMARY_INITIAL)
    spin_excess = max(0.0, config.chi - chi_sat)
    cloud_fraction = config.alpha * spin_excess / PRIMARY_INITIAL.m
    mass_planck = config.mass_msun * SOLAR_MASS_IN_PLANCK_MASS
    occupation = (
        cloud_fraction * mass_planck * mass_planck / config.alpha
        if cloud_fraction > 0.0
        else 0.0
    )
    log_required = (
        math.log(occupation / config.seed_occupancy)
        if occupation > config.seed_occupancy
        else 0.0
    )

    if gamma <= 0.0 or log_required <= 0.0:
        return FormationResult(
            alpha=config.alpha,
            chi=config.chi,
            q=config.q,
            mass_msun=config.mass_msun,
            status="formation_veto",
            saturation_spin=chi_sat,
            cloud_mass_fraction=cloud_fraction,
            log_required_occupancy=log_required,
            pre_resonance_efolds=0.0,
            birth_frequency_fraction=config.birth_frequency_fraction,
            latest_birth_frequency_fraction=None,
        )

    pre_efolds = _growth_efolds_between(
        config.birth_frequency_fraction * omega_res,
        omega_res,
        gamma,
        config.q,
    )

    def residual(fraction: float) -> float:
        return _growth_efolds_between(
            fraction * omega_res, omega_res, gamma, config.q
        ) - log_required

    lower = 1.0e-6
    upper = 1.0 - 1.0e-12
    latest_fraction = float(
        brentq(residual, lower, upper, xtol=1.0e-13, rtol=1.0e-13)
    )
    status = "formation_veto" if pre_efolds >= log_required else "physical_effect"
    return FormationResult(
        alpha=config.alpha,
        chi=config.chi,
        q=config.q,
        mass_msun=config.mass_msun,
        status=status,
        saturation_spin=chi_sat,
        cloud_mass_fraction=cloud_fraction,
        log_required_occupancy=log_required,
        pre_resonance_efolds=pre_efolds,
        birth_frequency_fraction=config.birth_frequency_fraction,
        latest_birth_frequency_fraction=latest_fraction,
    )
