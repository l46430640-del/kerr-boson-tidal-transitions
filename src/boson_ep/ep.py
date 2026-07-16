"""Exceptional-point conditions and robust q root finding."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from .models import EPResult, PRIMARY_FINAL, PRIMARY_INITIAL
from .spectrum import gamma_detweiler_M
from .tides import (
    cloud_radius_M,
    orbital_radius_M,
    resonance_frequency_M,
    tidal_eta_at_omega_M,
    tidal_eta_at_omega_M_array,
    tidal_eta_M,
    tidal_eta_M_array,
)
from .validation import validate_alpha_chi

VALID_STATUSES = {
    "physical_root",
    "root_q_gt_1",
    "no_root",
    "not_superradiant",
    "tidal_expansion_invalid",
    "calibration_failed",
}


def delta_gamma_M(alpha: float, chi: float) -> float:
    return gamma_detweiler_M(alpha, chi, PRIMARY_INITIAL) - gamma_detweiler_M(
        alpha, chi, PRIMARY_FINAL
    )


def analytic_far_tide_q_ep(alpha: float, chi: float) -> float | None:
    """Return the positive far-tide root, allowing q > 1 diagnostically."""
    validate_alpha_chi(alpha, chi)
    a_value = 8.0 * abs(delta_gamma_M(alpha, chi)) / (chi**2 * alpha**9)
    if not 0.0 < a_value < 1.0:
        return None
    return a_value / (1.0 - a_value)


def _empty_result(
    alpha: float,
    chi: float,
    status: str,
    *,
    omega_res_M: float | None = None,
    delta_gamma_value: float | None = None,
    analytic_q: float | None | object = ...,
) -> EPResult:
    if analytic_q is ...:
        analytic_q = analytic_far_tide_q_ep(alpha, chi)
    return EPResult(
        alpha=alpha,
        chi=chi,
        q=None,
        status=status,
        residual=None,
        discriminant_normalized=None,
        omega_res_M=(
            resonance_frequency_M(alpha, chi)
            if omega_res_M is None
            else omega_res_M
        ),
        radius_M=None,
        radius_over_cloud=None,
        eta_M=None,
        delta_gamma_M=(
            delta_gamma_M(alpha, chi)
            if delta_gamma_value is None
            else delta_gamma_value
        ),
        analytic_q=analytic_q,
    )


def _model_inputs(
    alpha: float,
    chi: float,
    spectrum_model: str,
) -> tuple[float, float, float] | None:
    if spectrum_model == "hydrogenic_detweiler":
        return (
            resonance_frequency_M(alpha, chi),
            gamma_detweiler_M(alpha, chi, PRIMARY_INITIAL),
            gamma_detweiler_M(alpha, chi, PRIMARY_FINAL),
        )
    if spectrum_model == "continued_fraction":
        from .relativity import solve_quasibound_cf

        initial = solve_quasibound_cf(alpha, chi, PRIMARY_INITIAL)
        final = solve_quasibound_cf(alpha, chi, PRIMARY_FINAL)
        if not initial.converged or not final.converged:
            return None
        delta_m = PRIMARY_FINAL.m - PRIMARY_INITIAL.m
        omega = (final.frequency_M.real - initial.frequency_M.real) / delta_m
        if omega <= 0.0:
            return None
        return omega, initial.frequency_M.imag, final.frequency_M.imag
    raise ValueError("unsupported spectrum_model")


def find_ep_roots(
    alpha: float,
    chi: float,
    q_bounds: tuple[float, float] = (1.0e-6, 1.0e3),
    sample_count: int = 512,
    *,
    spectrum_model: str = "hydrogenic_detweiler",
) -> list[EPResult]:
    validate_alpha_chi(alpha, chi)
    q_min, q_max = q_bounds
    if not 0.0 < q_min < q_max:
        raise ValueError("q_bounds must be positive and increasing")
    if sample_count < 16:
        raise ValueError("sample_count must be at least 16")

    model_inputs = _model_inputs(alpha, chi, spectrum_model)
    if model_inputs is None:
        result = _empty_result(alpha, chi, "calibration_failed")
        return [
            EPResult(
                **{
                    **result.to_dict(),
                    "omega_res_M": math.nan,
                    "delta_gamma_M": math.nan,
                    "analytic_q": None,
                }
            )
        ]
    omega, gamma_initial, gamma_final = model_inputs
    marginal_tolerance = 1.0e-12 * max(abs(gamma_final), alpha**9)
    if gamma_initial <= marginal_tolerance:
        return [
            _empty_result(
                alpha,
                chi,
                "not_superradiant",
                omega_res_M=omega,
                delta_gamma_value=gamma_initial - gamma_final,
                analytic_q=(
                    analytic_far_tide_q_ep(alpha, chi)
                    if spectrum_model == "hydrogenic_detweiler"
                    else None
                ),
            )
        ]

    signed_delta_gamma = gamma_initial - gamma_final
    delta_gamma = abs(signed_delta_gamma)
    q_grid = np.geomspace(q_min, q_max, sample_count)
    if spectrum_model == "hydrogenic_detweiler":
        eta_grid = tidal_eta_M_array(alpha, chi, q_grid)
    else:
        eta_grid = tidal_eta_at_omega_M_array(alpha, chi, q_grid, omega)
    values = 2.0 * eta_grid - delta_gamma
    brackets: list[tuple[float, float]] = []
    exact_grid_roots: list[float] = []
    for index in range(sample_count - 1):
        left_value = float(values[index])
        right_value = float(values[index + 1])
        if left_value == 0.0:
            exact_grid_roots.append(float(q_grid[index]))
        if left_value * right_value < 0.0:
            brackets.append((float(q_grid[index]), float(q_grid[index + 1])))
    if float(values[-1]) == 0.0:
        exact_grid_roots.append(float(q_grid[-1]))

    roots = exact_grid_roots[:]
    for left, right in brackets:
        root = brentq(
            lambda q: 2.0
            * (
                tidal_eta_M(alpha, chi, q)
                if spectrum_model == "hydrogenic_detweiler"
                else tidal_eta_at_omega_M(alpha, chi, q, omega)
            )
            - delta_gamma,
            left,
            right,
            xtol=1.0e-14,
            rtol=1.0e-14,
            maxiter=200,
        )
        roots.append(float(root))
    roots.sort()
    unique_roots: list[float] = []
    for root in roots:
        if not unique_roots or not math.isclose(
            root, unique_roots[-1], rel_tol=1.0e-10, abs_tol=1.0e-13
        ):
            unique_roots.append(root)

    if not unique_roots:
        return [
            _empty_result(
                alpha,
                chi,
                "no_root",
                omega_res_M=omega,
                delta_gamma_value=signed_delta_gamma,
                analytic_q=(
                    analytic_far_tide_q_ep(alpha, chi)
                    if spectrum_model == "hydrogenic_detweiler"
                    else None
                ),
            )
        ]

    analytic_q = (
        analytic_far_tide_q_ep(alpha, chi)
        if spectrum_model == "hydrogenic_detweiler"
        else None
    )
    results: list[EPResult] = []
    for root in unique_roots:
        eta = (
            tidal_eta_M(alpha, chi, root)
            if spectrum_model == "hydrogenic_detweiler"
            else tidal_eta_at_omega_M(alpha, chi, root, omega)
        )
        radius = orbital_radius_M(omega, root)
        radius_ratio = radius / cloud_radius_M(alpha)
        residual = abs(2.0 * eta - delta_gamma) / delta_gamma
        discriminant = abs(4.0 * eta * eta - delta_gamma * delta_gamma)
        discriminant_scale = 4.0 * eta * eta + delta_gamma * delta_gamma
        status = "physical_root" if root <= 1.0 else "root_q_gt_1"
        if radius_ratio < 10.0:
            status = "tidal_expansion_invalid"
        results.append(
            EPResult(
                alpha=alpha,
                chi=chi,
                q=root,
                status=status,
                residual=residual,
                discriminant_normalized=discriminant / discriminant_scale,
                omega_res_M=omega,
                radius_M=radius,
                radius_over_cloud=radius_ratio,
                eta_M=eta,
                delta_gamma_M=signed_delta_gamma,
                analytic_q=analytic_q,
            )
        )
    return results
