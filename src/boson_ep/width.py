"""Near-EP screening, systematic-error accounting, and effect widths."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
from scipy.optimize import brentq

from .dynamics import evolve_transition
from .ep import find_ep_roots
from .formation import assess_formation_history
from .models import (
    EffectWidthResult,
    EvolutionConfig,
    EvolutionResult,
    FormationConfig,
    PRIMARY_FINAL,
    PRIMARY_INITIAL,
    State,
    UncertaintyBudget,
    WidthScanConfig,
)
from .relativity import solve_quasibound_cf
from .spectrum import omega_real_M
from .tides import PRIMARY_DELTA_M, resonance_frequency_M, tidal_eta_M_array
from .timescales import gw_frequency_sweep_M2


def _two_level_truncation_bound(alpha: float, chi: float, eta: float) -> float:
    """Conservative n<=4 leakage/energy-shift bound using eta as an upper coupling."""
    omega_initial = omega_real_M(alpha, chi, PRIMARY_INITIAL)
    omega_res = resonance_frequency_M(alpha, chi)
    inverse_gaps: list[float] = []
    for n_value in range(2, 5):
        for l_value in range(n_value):
            if (l_value + PRIMARY_INITIAL.l + 2) % 2 != 0:
                continue
            if abs(l_value - PRIMARY_INITIAL.l) > 2:
                continue
            for m_value in range(-l_value, l_value + 1):
                delta_m = m_value - PRIMARY_INITIAL.m
                if abs(delta_m) > 2:
                    continue
                state = State(n_value, l_value, m_value)
                if state in {PRIMARY_INITIAL, PRIMARY_FINAL}:
                    continue
                detuning = (
                    omega_real_M(alpha, chi, state)
                    - omega_initial
                    - delta_m * omega_res
                )
                if abs(detuning) > 0.0:
                    inverse_gaps.append(1.0 / abs(detuning))
    if not inverse_gaps:
        return 0.0
    leakage = sum((eta * inverse_gap) ** 2 for inverse_gap in inverse_gaps)
    energy_shift = eta * eta * sum(inverse_gaps)
    reference = max(eta, alpha**6 * max(chi, 1.0e-12), 1.0e-300)
    return float(min(1.0, leakage + energy_shift / reference))


def _effect_for(config: EvolutionConfig) -> EvolutionResult:
    return evolve_transition(
        replace(config, store_trajectory=False, crosscheck=False)
    )


def compute_uncertainty_budget(
    config: EvolutionConfig,
    central: EvolutionResult | None = None,
) -> UncertaintyBudget:
    """Return the operational RSS theory-error budget for D=|S_NH-S_fac|."""
    if central is None:
        central = _effect_for(config)
    center = central.effect_abs

    if central.status == "integration_not_converged":
        numerical = 1.0
    else:
        numerical = max(
            central.solver_error or 0.0,
            central.probability_balance_residual,
        )

    cf_settings = (100, 200, 400)
    cf_pairs: list[tuple[complex, complex]] = []
    if config.alpha >= 0.12:
        for truncation in cf_settings:
            from .models import CFSettings

            settings = CFSettings(truncation=truncation)
            initial = solve_quasibound_cf(
                config.alpha, config.chi, PRIMARY_INITIAL, settings
            )
            final = solve_quasibound_cf(
                config.alpha, config.chi, PRIMARY_FINAL, settings
            )
            if initial.converged and final.converged:
                cf_pairs.append((initial.frequency_M, final.frequency_M))
    spectrum = 1.0
    if len(cf_pairs) == len(cf_settings):
        gamma_scale = max(
            abs(cf_pairs[-1][0].imag),
            abs(cf_pairs[-1][1].imag),
            1.0e-300,
        )
        truncation_error = max(
            abs(cf_pairs[-1][0].imag - cf_pairs[-2][0].imag),
            abs(cf_pairs[-1][1].imag - cf_pairs[-2][1].imag),
        ) / gamma_scale
        if truncation_error < 1.0e-3:
            try:
                calibrated = _effect_for(
                    replace(
                        config,
                        spectrum_model="continued_fraction",
                        omega_initial_M=cf_pairs[-1][0],
                        omega_final_M=cf_pairs[-1][1],
                    )
                )
                spectrum = abs(calibrated.effect_abs - center)
            except (RuntimeError, ValueError):
                spectrum = 1.0

    eta_fraction = min(0.25, 2.5 * config.alpha * config.alpha)
    if central.landau_zener_z > 100.0 and center < 1.0e-10:
        # Even the downward eta envelope remains exponentially saturated in
        # the physical EP scan.  Use a conservative absolute floor rather
        # than repeating several long integrations whose output underflows.
        tidal = 1.0e-10
        chirp = 1.0e-10
    else:
        tidal_values = [
            _effect_for(
                replace(config, coupling_scale=1.0 + sign * eta_fraction)
            ).effect_abs
            for sign in (-1.0, 1.0)
        ]
        tidal = max(abs(value - center) for value in tidal_values)
        chirp_value = _effect_for(replace(config, chirp_order="1pn")).effect_abs
        chirp = abs(chirp_value - center)
    two_level = _two_level_truncation_bound(
        config.alpha, config.chi, central.eta_res_M
    )
    components = {
        "numerical": numerical,
        "spectrum": spectrum,
        "tidal_matrix": tidal,
        "chirp": chirp,
        "two_level": two_level,
    }
    total = math.sqrt(sum(value * value for value in components.values()))
    worst = max(components, key=components.get)
    return UncertaintyBudget(
        numerical=numerical,
        spectrum=spectrum,
        tidal_matrix=tidal,
        chirp=chirp,
        two_level=two_level,
        total=total,
        worst_component=worst,
    )


def _asymptotic_screen(
    alpha: float,
    chi: float,
    q_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constant-coefficient infinite-window LZ screen.

    In this limit a constant loss on the destination level does not alter the
    asymptotic diabatic survival.  Nonzero D must therefore come from the full
    chirp/coupling evolution and cannot be manufactured by the screen.
    """
    eta = tidal_eta_M_array(alpha, chi, q_grid)
    omega = resonance_frequency_M(alpha, chi)
    sweep = np.asarray(
        [abs(PRIMARY_DELTA_M) * gw_frequency_sweep_M2(omega, float(q)) for q in q_grid]
    )
    survival = np.exp(-2.0 * math.pi * eta * eta / sweep)
    return survival.copy(), survival.copy(), np.zeros_like(survival)


def compute_near_ep_width(config: WidthScanConfig) -> EffectWidthResult:
    """Screen and refine the qualified q interval surrounding the physical EP."""
    roots = find_ep_roots(config.alpha, config.chi, (config.q_min, config.q_max))
    physical = [root for root in roots if root.status == "physical_root"]
    q_grid = np.geomspace(config.q_min, config.q_max, config.q_points)
    survival_nh, survival_fac, effect_grid = _asymptotic_screen(
        config.alpha, config.chi, q_grid
    )
    sigma_grid = np.zeros_like(q_grid)
    if not physical:
        return EffectWidthResult(
            alpha=config.alpha,
            chi=config.chi,
            mass_msun=config.mass_msun,
            status="no_ep",
            q_ep=None,
            q_low=None,
            q_high=None,
            formal_relative_width=0.0,
            physical_relative_width=0.0,
            effect_at_ep=None,
            sigma_at_ep=None,
            uncertainty=None,
            formation_status=None,
            q_grid=q_grid,
            effect_grid=effect_grid,
            sigma_grid=sigma_grid,
            survival_nh_grid=survival_nh,
            survival_factorized_grid=survival_fac,
        )

    q_ep = float(physical[0].q)
    evolution_config = EvolutionConfig(
        alpha=config.alpha,
        chi=config.chi,
        q=q_ep,
        edge_factor=config.edge_factor,
        rtol=config.rtol,
        atol=config.atol,
        store_trajectory=False,
        crosscheck=False,
    )
    central = evolve_transition(evolution_config)
    uncertainty = (
        compute_uncertainty_budget(evolution_config, central)
        if config.include_systematics
        else UncertaintyBudget(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "none")
    )
    sigma_grid.fill(uncertainty.total)
    qualified = (
        effect_grid >= config.effect_threshold
    ) & (effect_grid >= config.sigma_multiplier * sigma_grid)

    q_low: float | None = None
    q_high: float | None = None
    formal_width = 0.0
    if np.any(qualified):
        ep_index = int(np.argmin(np.abs(np.log(q_grid / q_ep))))
        if qualified[ep_index]:
            low = ep_index
            high = ep_index
            while low > 0 and qualified[low - 1]:
                low -= 1
            while high + 1 < q_grid.size and qualified[high + 1]:
                high += 1
            score = np.minimum(
                effect_grid - config.effect_threshold,
                effect_grid - config.sigma_multiplier * sigma_grid,
            )

            def interpolated_score(q_value: float) -> float:
                return float(
                    np.interp(np.log(q_value), np.log(q_grid), score)
                )

            q_low = float(q_grid[low])
            if low > 0 and score[low - 1] * score[low] <= 0.0:
                q_low = float(
                    brentq(
                        interpolated_score,
                        float(q_grid[low - 1]),
                        float(q_grid[low]),
                        rtol=1.0e-12,
                    )
                )
            q_high = float(q_grid[high])
            if high + 1 < q_grid.size and score[high] * score[high + 1] <= 0.0:
                q_high = float(
                    brentq(
                        interpolated_score,
                        float(q_grid[high]),
                        float(q_grid[high + 1]),
                        rtol=1.0e-12,
                    )
                )
            formal_width = (q_high - q_low) / q_ep

    formation = assess_formation_history(
        FormationConfig(config.alpha, config.chi, q_ep, config.mass_msun)
    )
    physical_width = formal_width if formation.status != "formation_veto" else 0.0
    if central.status == "integration_not_converged":
        status = "integration_not_converged"
    elif central.effect_abs < config.effect_threshold:
        status = "effect_below_threshold"
    elif central.effect_abs < config.sigma_multiplier * uncertainty.total:
        status = "systematics_dominated"
    elif formation.status == "formation_veto":
        status = "formation_veto"
    else:
        status = "physical_effect"
    return EffectWidthResult(
        alpha=config.alpha,
        chi=config.chi,
        mass_msun=config.mass_msun,
        status=status,
        q_ep=q_ep,
        q_low=q_low,
        q_high=q_high,
        formal_relative_width=formal_width,
        physical_relative_width=physical_width,
        effect_at_ep=central.effect_abs,
        sigma_at_ep=uncertainty.total,
        uncertainty=uncertainty,
        formation_status=formation.status,
        q_grid=q_grid,
        effect_grid=effect_grid,
        sigma_grid=sigma_grid,
        survival_nh_grid=survival_nh,
        survival_factorized_grid=survival_fac,
    )
