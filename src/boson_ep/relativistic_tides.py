"""Relativistic Kerr QBS tidal kernels and transition-atlas orchestration.

This module deliberately separates a Newtonian operator, Kerr wavefunctions,
and the covariant ``delta Box_g`` operator.  The latter uses the adiabatic IRG
metric in :mod:`boson_ep.tidal_metric`; it is not a global binary metric.
"""

from __future__ import annotations

import cmath
from dataclasses import replace
from functools import lru_cache
import math
from typing import Callable, Mapping, Sequence
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import eval_genlaguerre, gammaln, sph_harm_y

from .models import (
    AtlasConfig,
    GaugeAuditResult,
    GaugeVectorSpec,
    KernelErrorBudgetResult,
    KerrModeResult,
    PhenomenologyResult,
    RelativisticTideSettings,
    State,
    TransitionConfig,
    TransitionKernelResult,
    TransitionResult,
)
from .relativity import (
    angular_mode_second_derivative,
    angular_mode_values,
    build_radial_ode_evaluator,
    radial_mode_values,
    saturation_spin_cf,
    solve_kerr_mode,
)
from .tidal_metric import (
    boyer_lindquist_christoffel,
    irg_algebraic_residuals,
    irg_metric_coefficient,
    irg_metric_coefficient_bl,
    kerr_metric_advanced,
    kerr_metric_boyer_lindquist,
    schwarzschild_rw_metric_coefficient,
)


TRANSITION_CHANNELS = (
    (State(2, 1, 1), State(2, 1, -1)),
    (State(3, 2, 2), State(3, 2, 0)),
    (State(3, 2, 2), State(3, 0, 0)),
    (State(4, 3, 3), State(4, 3, 1)),
    (State(4, 3, 3), State(4, 1, 1)),
)


def _complex_fsum(values: Sequence[complex]) -> complex:
    """Accumulate oscillatory quadrature contributions without pair loss."""
    materialized = [complex(value) for value in values]
    return complex(
        math.fsum(value.real for value in materialized),
        math.fsum(value.imag for value in materialized),
    )


MetricProvider = Callable[[float, float], np.ndarray]


_MODE_FIELD_PROVIDER_CACHE: dict[tuple[object, ...], Callable] = {}
_MODE_FIELD_PROVIDER_CACHE_LIMIT = 256


def _qbs_decay_rate(alpha: float, frequency: complex) -> float:
    """Positive real part of the complex QBS decay wave number."""
    value = cmath.sqrt(alpha * alpha - frequency * frequency)
    if value.real < 0.0:
        value = -value
    return max(float(value.real), 1.0e-5)


def _hydrogenic_radial(alpha: float, state: State, radius: np.ndarray) -> np.ndarray:
    bohr = 1.0 / alpha**2
    rho = 2.0 * radius / (state.n * bohr)
    log_norm = (
        math.log(2.0)
        - 2.0 * math.log(state.n)
        - 1.5 * math.log(bohr)
        + 0.5
        * (
            gammaln(state.n - state.l)
            - gammaln(state.n + state.l + 1)
        )
    )
    return (
        math.exp(log_norm)
        * rho**state.l
        * np.exp(-rho / 2.0)
        * eval_genlaguerre(state.n - state.l - 1, 2 * state.l + 1, rho)
    )


def hydrogenic_newtonian_kernel_M(
    alpha: float,
    initial: State,
    final: State,
    radial_nodes: int = 180,
    angular_nodes: int = 120,
) -> complex:
    """Newtonian quadrupole kernel per epsilon for the m*=-2 harmonic."""
    if final.m != initial.m - 2:
        return 0.0j
    x_nodes, x_weights = np.polynomial.legendre.leggauss(radial_nodes)
    scale = max(initial.n, final.n) ** 2 / alpha**2
    outer = 35.0 * scale
    radius = 0.5 * outer * (x_nodes + 1.0)
    radial_weights = 0.5 * outer * x_weights
    radial = np.sum(
        radial_weights
        * radius**4
        * _hydrogenic_radial(alpha, initial, radius)
        * _hydrogenic_radial(alpha, final, radius)
    )
    u_nodes, u_weights = np.polynomial.legendre.leggauss(angular_nodes)
    theta = np.arccos(u_nodes)
    angular = np.sum(
        u_weights
        * (1.0 - u_nodes**2)
        * sph_harm_y(final.l, final.m, theta, 0.0)
        * sph_harm_y(initial.l, initial.m, theta, 0.0)
    )
    # Phi_N/epsilon has m=-2 coefficient -3 r^2 sin^2(theta)/8.
    return complex(-3.0 * alpha * math.pi / 4.0 * radial * angular)


@lru_cache(maxsize=128)
def _cached_mode(
    alpha: float, chi: float, state: State, settings
) -> KerrModeResult:
    return solve_kerr_mode(alpha, chi, state, settings)


def _mode_fields(
    mode: KerrModeResult,
    radius: float,
    theta: float,
) -> tuple[complex, np.ndarray]:
    radial, radial_first, radial_second = radial_mode_values(
        radius,
        mode.alpha,
        mode.chi,
        mode.state,
        mode.frequency_M,
        mode.radial_coefficients,
        advanced=True,
    )
    angular, angular_first, _ = angular_mode_values(
        theta,
        mode.state,
        mode.separation_constant,
        mode.angular_l_values,
        mode.angular_coefficients,
    )
    angular_second = angular_mode_second_derivative(
        theta,
        mode.state,
        mode.alpha,
        mode.chi,
        mode.frequency_M,
        mode.separation_constant,
        angular,
        angular_first,
    )
    value = complex(radial * angular)
    derivatives = np.asarray(
        [
            -1j * mode.frequency_M * value,
            radial_first * angular,
            radial * angular_first,
            1j * mode.state.m * value,
        ],
        dtype=complex,
    )
    second = np.asarray([radial_second * angular, radial * angular_second])
    return value, np.concatenate((derivatives, second))


def _mode_fields_bl(
    mode: KerrModeResult,
    radius: float,
    theta: float,
) -> tuple[complex, np.ndarray]:
    """Mode and derivatives on a Boyer-Lindquist t=constant slice."""
    radial, radial_first, radial_second = radial_mode_values(
        radius,
        mode.alpha,
        mode.chi,
        mode.state,
        mode.frequency_M,
        mode.radial_coefficients,
        advanced=False,
    )
    angular, angular_first, _ = angular_mode_values(
        theta,
        mode.state,
        mode.separation_constant,
        mode.angular_l_values,
        mode.angular_coefficients,
    )
    angular_second = angular_mode_second_derivative(
        theta,
        mode.state,
        mode.alpha,
        mode.chi,
        mode.frequency_M,
        mode.separation_constant,
        angular,
        angular_first,
    )
    value = complex(radial * angular)
    derivatives = np.asarray(
        [
            -1j * mode.frequency_M * value,
            radial_first * angular,
            radial * angular_first,
            1j * mode.state.m * value,
        ],
        dtype=complex,
    )
    return value, np.concatenate(
        (derivatives, np.asarray([radial_second * angular, radial * angular_second]))
    )


def _mode_field_provider(
    mode: KerrModeResult,
    outer_decay_lengths: float = 20.0,
) -> Callable[[float, float], tuple[complex, np.ndarray]]:
    """Build a separably cached BL mode evaluator for a quadrature run."""
    key = (
        mode.alpha,
        mode.chi,
        mode.state,
        mode.frequency_M,
        mode.separation_constant,
        mode.selected_truncation,
        float(outer_decay_lengths),
        hash(mode.radial_coefficients.tobytes()),
        hash(mode.angular_coefficients.tobytes()),
    )
    cached = _MODE_FIELD_PROVIDER_CACHE.get(key)
    if cached is not None:
        return cached
    radial_evaluator = build_radial_ode_evaluator(
        mode.alpha,
        mode.chi,
        mode.state,
        mode.frequency_M,
        mode.separation_constant,
        mode.radial_coefficients,
        outer_decay_lengths=outer_decay_lengths,
    )

    @lru_cache(maxsize=None)
    def radial_part(radius: float) -> tuple[complex, complex, complex]:
        values = radial_evaluator(radius)
        return tuple(complex(value) for value in values)

    @lru_cache(maxsize=None)
    def angular_part(theta: float) -> tuple[complex, complex, complex]:
        angular, angular_first, _ = angular_mode_values(
            theta,
            mode.state,
            mode.separation_constant,
            mode.angular_l_values,
            mode.angular_coefficients,
        )
        angular_second = angular_mode_second_derivative(
            theta,
            mode.state,
            mode.alpha,
            mode.chi,
            mode.frequency_M,
            mode.separation_constant,
            angular,
            angular_first,
        )
        return complex(angular), complex(angular_first), complex(angular_second)

    def fields(radius: float, theta: float) -> tuple[complex, np.ndarray]:
        radial, radial_first, radial_second = radial_part(float(radius))
        angular, angular_first, angular_second = angular_part(float(theta))
        value = radial * angular
        derivatives = np.asarray(
            [
                -1j * mode.frequency_M * value,
                radial_first * angular,
                radial * angular_first,
                1j * mode.state.m * value,
                radial_second * angular,
                radial * angular_second,
            ],
            dtype=complex,
        )
        return complex(value), derivatives

    if len(_MODE_FIELD_PROVIDER_CACHE) >= _MODE_FIELD_PROVIDER_CACHE_LIMIT:
        _MODE_FIELD_PROVIDER_CACHE.pop(next(iter(_MODE_FIELD_PROVIDER_CACHE)))
    _MODE_FIELD_PROVIDER_CACHE[key] = fields
    return fields


def _density_tensor(
    radius: float,
    theta: float,
    chi: float,
    omega_orb_M: float,
) -> tuple[np.ndarray, float]:
    metric = kerr_metric_advanced(radius, theta, chi)
    inverse = np.linalg.inv(metric)
    perturbation = irg_metric_coefficient(radius, theta, chi, omega_orb_M)
    raised = inverse @ perturbation @ inverse
    sqrt_determinant = (radius * radius + chi * chi * math.cos(theta) ** 2) * math.sin(theta)
    trace = np.einsum("ab,ab->", inverse, perturbation)
    density = sqrt_determinant * (0.5 * trace * inverse - raised)
    return density, sqrt_determinant


def _density_tensor_bl(
    radius: float,
    theta: float,
    chi: float,
    omega_orb_M: float,
    gauge: str | MetricProvider,
) -> tuple[np.ndarray, float, complex]:
    metric = kerr_metric_boyer_lindquist(radius, theta, chi)
    inverse = np.linalg.inv(metric)
    if callable(gauge):
        perturbation = np.asarray(gauge(radius, theta), dtype=complex)
    elif gauge == "irg":
        perturbation = irg_metric_coefficient_bl(
            radius, theta, chi, omega_orb_M
        )
    elif gauge == "rw":
        if abs(chi) > 1.0e-14:
            raise ValueError("Regge-Wheeler gauge is implemented only for chi=0")
        if abs(omega_orb_M) > 1.0e-14:
            raise ValueError(
                "the Regge-Wheeler provider is a static Omega->0 benchmark; "
                "finite-frequency calls are not defined"
            )
        perturbation = schwarzschild_rw_metric_coefficient(
            radius, theta, omega_orb_M
        )
    else:
        raise ValueError("gauge must be 'irg' or 'rw'")
    raised = inverse @ perturbation @ inverse
    sqrt_determinant = (
        radius * radius + chi * chi * math.cos(theta) ** 2
    ) * math.sin(theta)
    trace = np.einsum("ab,ab->", inverse, perturbation)
    density = sqrt_determinant * (0.5 * trace * inverse - raised)
    return density, sqrt_determinant, complex(trace)


def _metric_perturbation_bl(
    radius: float,
    theta: float,
    chi: float,
    omega_orb_M: float,
    gauge: str | MetricProvider,
) -> np.ndarray:
    """Return the covariant BL perturbation used by both operator forms."""
    if callable(gauge):
        return np.asarray(gauge(radius, theta), dtype=complex)
    if gauge == "irg":
        return irg_metric_coefficient_bl(radius, theta, chi, omega_orb_M)
    if gauge == "rw":
        if abs(chi) > 1.0e-14 or abs(omega_orb_M) > 1.0e-14:
            raise ValueError("Regge-Wheeler is restricted to static Schwarzschild")
        return schwarzschild_rw_metric_coefficient(radius, theta, 0.0)
    raise ValueError("gauge must be 'irg', 'rw', or a metric provider")


def _raised_perturbation_bl(
    radius: float,
    theta: float,
    chi: float,
    omega_orb_M: float,
    gauge: str | MetricProvider,
) -> tuple[np.ndarray, complex]:
    metric = kerr_metric_boyer_lindquist(radius, theta, chi)
    inverse = np.linalg.inv(metric)
    perturbation = _metric_perturbation_bl(
        radius, theta, chi, omega_orb_M, gauge
    )
    return inverse @ perturbation @ inverse, complex(
        np.einsum("ab,ab->", inverse, perturbation)
    )


def _background_christoffel_bl(
    radius: float, theta: float, chi: float, difference_scale: float = 1.0
) -> np.ndarray:
    """Analytic BL Christoffels, independent of the tidal operator code."""
    del difference_scale
    return boyer_lindquist_christoffel(radius, theta, chi)


def _delta_box_connection_bl(
    mode: KerrModeResult,
    radius: float,
    theta: float,
    omega_orb_M: float,
    gauge: str | MetricProvider,
    difference_scale: float = 1.0,
    field_provider: Callable[[float, float], tuple[complex, np.ndarray]] | None = None,
) -> complex:
    """Connection-form ``delta Box`` on the production BL slice."""
    fields_provider = field_provider or (lambda r, t: _mode_fields_bl(mode, r, t))
    value, fields = fields_provider(radius, theta)
    first = fields[:4]
    second = np.zeros((4, 4), dtype=complex)
    second[0, 0] = (-1j * mode.frequency_M) ** 2 * value
    second[3, 3] = (1j * mode.state.m) ** 2 * value
    second[0, 3] = second[3, 0] = (
        -1j * mode.frequency_M * 1j * mode.state.m * value
    )
    second[1, 1] = fields[4]
    second[2, 2] = fields[5]
    second[0, 1] = second[1, 0] = -1j * mode.frequency_M * first[1]
    second[0, 2] = second[2, 0] = -1j * mode.frequency_M * first[2]
    second[1, 3] = second[3, 1] = 1j * mode.state.m * first[1]
    second[2, 3] = second[3, 2] = 1j * mode.state.m * first[2]
    angular, angular_first, _ = angular_mode_values(
        theta, mode.state, mode.separation_constant, mode.angular_l_values,
        mode.angular_coefficients,
    )
    if abs(angular) > 1.0e-30:
        radial_first = first[1] / angular
        cross_derivative = radial_first * angular_first
    else:
        cross_derivative = 0.0j
    second[1, 2] = second[2, 1] = cross_derivative

    connection = _background_christoffel_bl(
        radius, theta, mode.chi, difference_scale
    )
    h_raised, trace = _raised_perturbation_bl(
        radius, theta, mode.chi, omega_orb_M, gauge
    )
    hessian = second - np.einsum("cab,c->ab", connection, first)
    r_plus = 1.0 + math.sqrt(1.0 - mode.chi**2)
    radial_step = difference_scale * min(
        max(2.0e-4 * radius, 2.0e-5), 0.2 * (radius - r_plus)
    )
    theta_step = difference_scale * 2.0e-4
    derivative_h = np.zeros((4, 4, 4), dtype=complex)
    derivative_trace = np.zeros(4, dtype=complex)
    derivative_h[0] = 2j * omega_orb_M * h_raised
    derivative_h[3] = -2j * h_raised
    derivative_trace[0] = 2j * omega_orb_M * trace
    derivative_trace[3] = -2j * trace

    def raised_at(r_value: float, theta_value: float):
        return _raised_perturbation_bl(
            r_value, theta_value, mode.chi, omega_orb_M, gauge
        )

    r_m2_h, r_m2_trace = raised_at(radius - 2.0 * radial_step, theta)
    r_m1_h, r_m1_trace = raised_at(radius - radial_step, theta)
    r_p1_h, r_p1_trace = raised_at(radius + radial_step, theta)
    r_p2_h, r_p2_trace = raised_at(radius + 2.0 * radial_step, theta)
    derivative_h[1] = (
        r_m2_h - 8.0 * r_m1_h + 8.0 * r_p1_h - r_p2_h
    ) / (12.0 * radial_step)
    derivative_trace[1] = (
        r_m2_trace - 8.0 * r_m1_trace
        + 8.0 * r_p1_trace - r_p2_trace
    ) / (12.0 * radial_step)
    t_m2_h, t_m2_trace = raised_at(radius, theta - 2.0 * theta_step)
    t_m1_h, t_m1_trace = raised_at(radius, theta - theta_step)
    t_p1_h, t_p1_trace = raised_at(radius, theta + theta_step)
    t_p2_h, t_p2_trace = raised_at(radius, theta + 2.0 * theta_step)
    derivative_h[2] = (
        t_m2_h - 8.0 * t_m1_h + 8.0 * t_p1_h - t_p2_h
    ) / (12.0 * theta_step)
    derivative_trace[2] = (
        t_m2_trace - 8.0 * t_m1_trace
        + 8.0 * t_p1_trace - t_p2_trace
    ) / (12.0 * theta_step)

    divergence = np.zeros(4, dtype=complex)
    for index in range(4):
        divergence[index] = sum(derivative_h[a, a, index] for a in range(4))
        divergence[index] += sum(
            connection[a, a, c] * h_raised[c, index]
            + connection[index, a, c] * h_raised[a, c]
            for a in range(4) for c in range(4)
        )
    inverse = np.linalg.inv(
        kerr_metric_boyer_lindquist(radius, theta, mode.chi)
    )
    raised_trace_gradient = inverse @ derivative_trace
    return complex(
        -np.einsum("ab,ab->", h_raised, hessian)
        - np.dot(divergence - 0.5 * raised_trace_gradient, first)
    )


def _delta_box_divergence_bl(
    mode: KerrModeResult,
    radius: float,
    theta: float,
    omega_orb_M: float,
    gauge: str | MetricProvider,
    difference_scale: float = 1.0,
    field_provider: Callable[[float, float], tuple[complex, np.ndarray]] | None = None,
) -> complex:
    """Divergence-form delta Box on a BL t=constant slice."""
    fields = field_provider or (lambda r, t: _mode_fields_bl(mode, r, t))
    value, derivatives = fields(radius, theta)
    first = derivatives[:4]
    density, sqrt_determinant, trace = _density_tensor_bl(
        radius, theta, mode.chi, omega_orb_M, gauge
    )
    r_plus = 1.0 + math.sqrt(1.0 - mode.chi**2)
    radial_step = difference_scale * min(
        max(2.0e-4 * radius, 2.0e-5),
        0.2 * (radius - r_plus),
    )
    theta_step = difference_scale * 2.0e-4

    def flux_r(r_value: float) -> complex:
        tensor, _, _ = _density_tensor_bl(
            r_value, theta, mode.chi, omega_orb_M, gauge
        )
        local_first = fields(r_value, theta)[1][:4]
        return complex(tensor[1] @ local_first)

    def flux_theta(theta_value: float) -> complex:
        tensor, _, _ = _density_tensor_bl(
            radius, theta_value, mode.chi, omega_orb_M, gauge
        )
        local_first = fields(radius, theta_value)[1][:4]
        return complex(tensor[2] @ local_first)

    radial_derivative = (
        flux_r(radius - 2.0 * radial_step)
        - 8.0 * flux_r(radius - radial_step)
        + 8.0 * flux_r(radius + radial_step)
        - flux_r(radius + 2.0 * radial_step)
    ) / (12.0 * radial_step)
    theta_derivative = (
        flux_theta(theta - 2.0 * theta_step)
        - 8.0 * flux_theta(theta - theta_step)
        + 8.0 * flux_theta(theta + theta_step)
        - flux_theta(theta + 2.0 * theta_step)
    ) / (12.0 * theta_step)
    total_frequency = -mode.frequency_M + 2.0 * omega_orb_M
    total_m = mode.state.m - 2
    time_derivative = 1j * total_frequency * complex(density[0] @ first)
    phi_derivative = 1j * total_m * complex(density[3] @ first)
    divergence = (
        time_derivative + radial_derivative + theta_derivative + phi_derivative
    ) / sqrt_determinant
    # Varying the leading 1/sqrt(-g) in Box contributes
    # -h Box(Phi)/2 = -h mu^2 Phi/2 on an unperturbed KG solution.
    return divergence - 0.5 * trace * mode.alpha**2 * value


def _raised_perturbation(
    radius: float,
    theta: float,
    chi: float,
    omega_orb_M: float,
) -> tuple[np.ndarray, complex]:
    metric = kerr_metric_advanced(radius, theta, chi)
    inverse = np.linalg.inv(metric)
    perturbation = irg_metric_coefficient(radius, theta, chi, omega_orb_M)
    return inverse @ perturbation @ inverse, complex(
        np.einsum("ab,ab->", inverse, perturbation)
    )


def _background_christoffel(radius: float, theta: float, chi: float) -> np.ndarray:
    metric = kerr_metric_advanced(radius, theta, chi)
    inverse = np.linalg.inv(metric)
    steps = (max(2.0e-5 * radius, 2.0e-6), 2.0e-5)
    derivatives = np.zeros((4, 4, 4), dtype=float)
    derivatives[1] = (
        kerr_metric_advanced(radius + steps[0], theta, chi)
        - kerr_metric_advanced(radius - steps[0], theta, chi)
    ) / (2.0 * steps[0])
    derivatives[2] = (
        kerr_metric_advanced(radius, theta + steps[1], chi)
        - kerr_metric_advanced(radius, theta - steps[1], chi)
    ) / (2.0 * steps[1])
    connection = np.zeros((4, 4, 4), dtype=float)
    for upper in range(4):
        for first in range(4):
            for second in range(4):
                connection[upper, first, second] = 0.5 * sum(
                    inverse[upper, lower]
                    * (
                        derivatives[first, lower, second]
                        + derivatives[second, lower, first]
                        - derivatives[lower, first, second]
                    )
                    for lower in range(4)
                )
    return connection


def _delta_box_connection(
    mode: KerrModeResult,
    radius: float,
    theta: float,
    omega_orb_M: float,
    difference_scale: float = 1.0,
) -> complex:
    """Independent connection-form evaluation of the linearized operator."""
    value, fields = _mode_fields(mode, radius, theta)
    first = fields[:4]
    second = np.zeros((4, 4), dtype=complex)
    second[0, 0] = (-1j * mode.frequency_M) ** 2 * value
    second[3, 3] = (1j * mode.state.m) ** 2 * value
    second[0, 3] = second[3, 0] = (
        -1j * mode.frequency_M * 1j * mode.state.m * value
    )
    second[1, 1] = fields[4]
    second[2, 2] = fields[5]
    second[0, 1] = second[1, 0] = -1j * mode.frequency_M * first[1]
    second[0, 2] = second[2, 0] = -1j * mode.frequency_M * first[2]
    second[1, 3] = second[3, 1] = 1j * mode.state.m * first[1]
    second[2, 3] = second[3, 2] = 1j * mode.state.m * first[2]
    radial, radial_first, _ = radial_mode_values(
        radius, mode.alpha, mode.chi, mode.state, mode.frequency_M,
        mode.radial_coefficients, advanced=True
    )
    angular, angular_first, _ = angular_mode_values(
        theta, mode.state, mode.separation_constant, mode.angular_l_values,
        mode.angular_coefficients
    )
    second[1, 2] = second[2, 1] = radial_first * angular_first
    connection = _background_christoffel(radius, theta, mode.chi)
    h_raised, trace = _raised_perturbation(
        radius, theta, mode.chi, omega_orb_M
    )
    hessian = second - np.einsum("cab,c->ab", connection, first)

    radial_step = difference_scale * max(2.0e-5 * radius, 2.0e-6)
    theta_step = difference_scale * 2.0e-5
    derivative_h = np.zeros((4, 4, 4), dtype=complex)
    derivative_trace = np.zeros(4, dtype=complex)
    derivative_h[0] = 2j * omega_orb_M * h_raised
    derivative_h[3] = -2j * h_raised
    derivative_trace[0] = 2j * omega_orb_M * trace
    derivative_trace[3] = -2j * trace
    plus_h, plus_trace = _raised_perturbation(
        radius + radial_step, theta, mode.chi, omega_orb_M
    )
    minus_h, minus_trace = _raised_perturbation(
        radius - radial_step, theta, mode.chi, omega_orb_M
    )
    derivative_h[1] = (plus_h - minus_h) / (2.0 * radial_step)
    derivative_trace[1] = (plus_trace - minus_trace) / (2.0 * radial_step)
    plus_h, plus_trace = _raised_perturbation(
        radius, theta + theta_step, mode.chi, omega_orb_M
    )
    minus_h, minus_trace = _raised_perturbation(
        radius, theta - theta_step, mode.chi, omega_orb_M
    )
    derivative_h[2] = (plus_h - minus_h) / (2.0 * theta_step)
    derivative_trace[2] = (plus_trace - minus_trace) / (2.0 * theta_step)
    divergence = np.zeros(4, dtype=complex)
    for index in range(4):
        divergence[index] = sum(derivative_h[a, a, index] for a in range(4))
        divergence[index] += sum(
            connection[a, a, c] * h_raised[c, index]
            + connection[index, a, c] * h_raised[a, c]
            for a in range(4)
            for c in range(4)
        )
    inverse = np.linalg.inv(kerr_metric_advanced(radius, theta, mode.chi))
    raised_trace_gradient = inverse @ derivative_trace
    return complex(
        -np.einsum("ab,ab->", h_raised, hessian)
        - np.dot(divergence - 0.5 * raised_trace_gradient, first)
    )


def _delta_box_divergence(
    mode: KerrModeResult,
    radius: float,
    theta: float,
    omega_orb_M: float,
    difference_scale: float = 1.0,
) -> complex:
    """Evaluate delta Box using its conservative divergence form."""
    _, derivatives = _mode_fields(mode, radius, theta)
    first = derivatives[:4]
    density, sqrt_determinant = _density_tensor(
        radius, theta, mode.chi, omega_orb_M
    )
    radial_step = difference_scale * max(2.0e-5 * radius, 2.0e-6)
    theta_step = difference_scale * 2.0e-5

    def flux_r(r_value: float) -> complex:
        tensor, _ = _density_tensor(r_value, theta, mode.chi, omega_orb_M)
        local_first = _mode_fields(mode, r_value, theta)[1][:4]
        return complex(tensor[1] @ local_first)

    def flux_theta(theta_value: float) -> complex:
        tensor, _ = _density_tensor(radius, theta_value, mode.chi, omega_orb_M)
        local_first = _mode_fields(mode, radius, theta_value)[1][:4]
        return complex(tensor[2] @ local_first)

    radial_derivative = (
        flux_r(radius + radial_step) - flux_r(radius - radial_step)
    ) / (2.0 * radial_step)
    theta_derivative = (
        flux_theta(theta + theta_step) - flux_theta(theta - theta_step)
    ) / (2.0 * theta_step)
    total_frequency = -mode.frequency_M + 2.0 * omega_orb_M
    total_m = mode.state.m - 2
    time_derivative = 1j * total_frequency * complex(density[0] @ first)
    phi_derivative = 1j * total_m * complex(density[3] @ first)
    return (time_derivative + radial_derivative + theta_derivative + phi_derivative) / sqrt_determinant


def _kerr_newtonian_kernel(
    initial: KerrModeResult,
    final: KerrModeResult,
    settings: RelativisticTideSettings | None = None,
) -> complex:
    if final.state.m != initial.state.m - 2:
        return 0.0j
    r_plus = 1.0 + math.sqrt(1.0 - initial.chi**2)
    decay_i = _qbs_decay_rate(initial.alpha, initial.frequency_M)
    decay_f = _qbs_decay_rate(final.alpha, final.frequency_M)
    mode_settings = settings.mode if settings is not None else None
    outer_lengths = 28.0 if mode_settings is None else mode_settings.outer_decay_lengths
    radial_nodes = 100 if settings is None else settings.radial_nodes
    angular_nodes = 80 if settings is None else settings.angular_nodes
    cutoff = 1.0e-4 if settings is None else settings.horizon_cutoffs[0]
    outer = r_plus + outer_lengths / min(decay_i, decay_f)
    nodes, weights = np.polynomial.legendre.leggauss(radial_nodes)
    radius = r_plus + cutoff + 0.5 * (outer - r_plus - cutoff) * (nodes + 1.0)
    radial_weights = 0.5 * (outer - r_plus - cutoff) * weights
    # A tiny domain margin prevents a roundoff-level mismatch between the
    # common physical cutoff and each mode's decay-length parametrization.
    common_span = (outer - r_plus) * 1.001
    radial_i = build_radial_ode_evaluator(
        initial.alpha, initial.chi, initial.state, initial.frequency_M,
        initial.separation_constant, initial.radial_coefficients,
        outer_decay_lengths=common_span * decay_i,
    )(radius)[0]
    radial_f = build_radial_ode_evaluator(
        final.alpha, final.chi, final.state, final.frequency_M,
        final.separation_constant, final.radial_coefficients,
        outer_decay_lengths=common_span * decay_f,
    )(radius)[0]
    radial = np.sum(radial_weights * radius**4 * radial_i * radial_f)
    u_nodes, u_weights = np.polynomial.legendre.leggauss(angular_nodes)
    theta = np.arccos(u_nodes)
    angular_i = angular_mode_values(
        theta, initial.state, initial.separation_constant,
        initial.angular_l_values, initial.angular_coefficients
    )[0]
    angular_f = angular_mode_values(
        theta, final.state, final.separation_constant,
        final.angular_l_values, final.angular_coefficients
    )[0]
    angular = np.sum(u_weights * (1.0 - u_nodes**2) * angular_i * angular_f)
    # QBS modes have unit Klein-Gordon bilinear norm.  Multiplication by
    # sqrt(2 omega_i) sqrt(2 omega_f) converts the weak-field radial factors
    # to unit Schrodinger normalization before inserting mu Phi_N.
    kg_to_schrodinger = 2.0 * cmath.sqrt(
        initial.frequency_M * final.frequency_M
    )
    return complex(
        -3.0
        * initial.alpha
        * math.pi
        / 4.0
        * radial
        * angular
        * kg_to_schrodinger
    )


def covariant_tidal_kernel_M(
    alpha: float,
    chi: float,
    initial: State,
    final: State,
    omega_orb_M: float,
    settings: RelativisticTideSettings = RelativisticTideSettings(),
) -> complex:
    """Return the complex covariant QBS-QBS kernel per unit epsilon."""
    if final.m != initial.m - 2:
        return 0.0j
    initial_mode = _cached_mode(alpha, chi, initial, settings.mode)
    final_mode = _cached_mode(alpha, chi, final, settings.mode)
    if not initial_mode.converged or not final_mode.converged:
        raise RuntimeError("a Kerr mode failed convergence; no covariant kernel returned")
    return _bl_tidal_kernel(
        initial_mode, final_mode, omega_orb_M, settings, gauge="irg"
    )


def _bl_tidal_kernel(
    initial_mode: KerrModeResult,
    final_mode: KerrModeResult,
    omega_orb_M: float,
    settings: RelativisticTideSettings,
    gauge: str | MetricProvider,
    *,
    radial_strategy: str = "gauss",
    radial_breaks: tuple[float, ...] = (),
    radial_domain: tuple[float, float] | None = None,
    operator_form: str = "divergence",
    initial_field_provider: Callable | None = None,
    final_field_provider: Callable | None = None,
) -> complex:
    """Project delta Box on a BL slice using the QBS bilinear left mode."""
    chi = initial_mode.chi
    alpha = initial_mode.alpha
    r_plus = 1.0 + math.sqrt(1.0 - chi**2)
    decay_i = _qbs_decay_rate(alpha, initial_mode.frequency_M)
    decay_f = _qbs_decay_rate(alpha, final_mode.frequency_M)
    outer_lengths = settings.mode.outer_decay_lengths
    outer = r_plus + outer_lengths / min(decay_i, decay_f)
    lower = r_plus + settings.horizon_cutoffs[0]
    if radial_domain is not None:
        lower = max(lower, radial_domain[0])
        outer = min(outer, radial_domain[1])
        if outer <= lower:
            return 0.0j
    u_nodes, u_weights = np.polynomial.legendre.leggauss(settings.angular_nodes)
    theta_values = np.arccos(u_nodes)
    common_span = (outer - r_plus) * 1.001
    initial_fields = (
        _mode_field_provider(initial_mode, common_span * decay_i)
        if initial_field_provider is None else initial_field_provider
    )
    final_fields = (
        _mode_field_provider(final_mode, common_span * decay_f)
        if final_field_provider is None else final_field_provider
    )

    def radial_integrand(radius: float) -> complex:
        angular_terms: list[complex] = []
        for theta, angular_weight in zip(theta_values, u_weights, strict=True):
            final_value = final_fields(float(radius), float(theta))[0]
            if operator_form == "divergence":
                source = _delta_box_divergence_bl(
                    initial_mode,
                    float(radius),
                    float(theta),
                    omega_orb_M,
                    gauge,
                    field_provider=initial_fields,
                )
            elif operator_form == "connection":
                source = _delta_box_connection_bl(
                    initial_mode,
                    float(radius),
                    float(theta),
                    omega_orb_M,
                    gauge,
                    field_provider=initial_fields,
                )
            else:
                raise ValueError("operator_form must be 'divergence' or 'connection'")
            sigma = radius * radius + chi * chi * math.cos(theta) ** 2
            angular_terms.append(
                complex(angular_weight * sigma * final_value * source)
            )
        return _complex_fsum(angular_terms)

    explicit_breaks = bool(radial_breaks)
    if explicit_breaks:
        breaks = sorted(value for value in radial_breaks if lower < value < outer)
        intervals = list(zip([lower, *breaks], [*breaks, outer], strict=True))
        nodes_per_interval = max(
            4, math.ceil(settings.radial_nodes / len(intervals))
        )

        def coordinate_integrand(value: float) -> complex:
            return radial_integrand(value)
    else:
        # Integrate in log(r-r_+) and split the full hierarchy into three
        # panels.  This is the production segmented Gauss-Legendre strategy;
        # ``radial_nodes`` remains the approximate total node count.
        log_edges = np.linspace(
            math.log(lower - r_plus), math.log(outer - r_plus), 4
        )
        intervals = list(zip(log_edges[:-1], log_edges[1:], strict=True))
        nodes_per_interval = max(4, math.ceil(settings.radial_nodes / 3))

        def coordinate_integrand(value: float) -> complex:
            offset = math.exp(value)
            return offset * radial_integrand(r_plus + offset)
    total = 0.0j
    if radial_strategy == "gauss":
        nodes, weights = np.polynomial.legendre.leggauss(nodes_per_interval)
        for left, right in intervals:
            radius_values = left + 0.5 * (right - left) * (nodes + 1.0)
            local_weights = 0.5 * (right - left) * weights
            total += _complex_fsum(
                weight * radial_integrand(float(radius))
                if explicit_breaks
                else weight * coordinate_integrand(float(radius))
                for radius, weight in zip(radius_values, local_weights, strict=True)
            )
    elif radial_strategy == "quad":
        for left, right in intervals:
            options = {
                "epsabs": settings.radial_atol,
                "epsrel": settings.radial_rtol,
                "limit": settings.max_subdivisions,
            }
            real = quad(lambda value: coordinate_integrand(value).real, left, right, **options)[0]
            imag = quad(lambda value: coordinate_integrand(value).imag, left, right, **options)[0]
            total += complex(real, imag)
    else:
        raise ValueError("radial_strategy must be 'gauss' or 'quad'")
    # J reflection of the final mode supplies exp(-i m_f phi), so the
    # on-resonance azimuthal integral is exactly 2 pi.
    # The second-order KG equation gives i dot(c_f)=-deltaO_fi c_i after
    # projection.  This minus sign aligns the covariant operator with the
    # Newtonian Hamiltonian convention used by the transition API.
    return complex(-2.0 * math.pi * total)


def tidal_kernel_from_modes_M(
    initial_mode: KerrModeResult,
    final_mode: KerrModeResult,
    omega_orb_M: float,
    settings: RelativisticTideSettings,
    metric: str | MetricProvider = "irg",
    *,
    radial_strategy: str = "gauss",
    radial_breaks: tuple[float, ...] = (),
    radial_domain: tuple[float, float] | None = None,
    operator_form: str = "divergence",
) -> complex:
    """Certification-level projection for precomputed modes and a metric."""
    return _bl_tidal_kernel(
        initial_mode,
        final_mode,
        omega_orb_M,
        settings,
        metric,
        radial_strategy=radial_strategy,
        radial_breaks=radial_breaks,
        radial_domain=radial_domain,
        operator_form=operator_form,
    )


def schwarzschild_rw_tidal_kernel_M(
    alpha: float,
    initial: State,
    final: State,
    settings: RelativisticTideSettings = RelativisticTideSettings(),
) -> complex:
    """Independent static Schwarzschild Regge-Wheeler-gauge benchmark.

    The local RW provider is intentionally restricted to ``Omega=0``.  The
    helper therefore accepts hyperfine-degenerate channels only; using it as
    a finite-frequency Kerr-atlas audit would silently drop time derivatives.
    """
    if final.m != initial.m - 2:
        return 0.0j
    initial_mode = _cached_mode(alpha, 0.0, initial, settings.mode)
    final_mode = _cached_mode(alpha, 0.0, final, settings.mode)
    if not initial_mode.converged or not final_mode.converged:
        raise RuntimeError("a Schwarzschild QBS failed convergence")
    omega_orb = (
        initial_mode.frequency_M.real - final_mode.frequency_M.real
    ) / (initial.m - final.m)
    if abs(omega_orb) > 1.0e-14:
        raise ValueError(
            "Regge-Wheeler comparison is available only in the Omega->0 limit"
        )
    return _bl_tidal_kernel(
        initial_mode, final_mode, omega_orb, settings, gauge="rw"
    )


def schwarzschild_irg_tidal_kernel_M(
    alpha: float,
    initial: State,
    final: State,
    settings: RelativisticTideSettings = RelativisticTideSettings(),
) -> complex:
    """Schwarzschild limit of the IRG kernel on the same BL slice."""
    if final.m != initial.m - 2:
        return 0.0j
    initial_mode = _cached_mode(alpha, 0.0, initial, settings.mode)
    final_mode = _cached_mode(alpha, 0.0, final, settings.mode)
    if not initial_mode.converged or not final_mode.converged:
        raise RuntimeError("a Schwarzschild QBS failed convergence")
    omega_orb = (
        initial_mode.frequency_M.real - final_mode.frequency_M.real
    ) / (initial.m - final.m)
    return _bl_tidal_kernel(
        initial_mode, final_mode, omega_orb, settings, gauge="irg"
    )


def schwarzschild_newtonian_tidal_kernel_M(
    alpha: float,
    initial: State,
    final: State,
    settings: RelativisticTideSettings = RelativisticTideSettings(),
) -> complex:
    """Newtonian quadrupole projected between normalized Schwarzschild QBSs."""
    if final.m != initial.m - 2:
        return 0.0j
    initial_mode = _cached_mode(alpha, 0.0, initial, settings.mode)
    final_mode = _cached_mode(alpha, 0.0, final, settings.mode)
    if not initial_mode.converged or not final_mode.converged:
        raise RuntimeError("a Schwarzschild QBS failed convergence")
    return _kerr_newtonian_kernel(initial_mode, final_mode, settings)


def relativistic_tidal_eta_M(
    alpha: float,
    chi: float,
    q: float,
    initial: State,
    final: State,
    settings: RelativisticTideSettings = RelativisticTideSettings(),
) -> complex:
    initial_mode = _cached_mode(alpha, chi, initial, settings.mode)
    final_mode = _cached_mode(alpha, chi, final, settings.mode)
    omega_res = (initial_mode.frequency_M.real - final_mode.frequency_M.real) / (
        initial.m - final.m
    )
    kernel = covariant_tidal_kernel_M(
        alpha, chi, initial, final, omega_res, settings
    )
    epsilon = q / (1.0 + q) * omega_res**2
    return epsilon * kernel


def _resolved_mode_pair(
    resolved_modes: Sequence[KerrModeResult] | Mapping[object, KerrModeResult],
    initial: State,
    final: State,
) -> tuple[KerrModeResult, KerrModeResult]:
    """Normalize the two supported resolved-mode containers."""
    if isinstance(resolved_modes, Mapping):
        def find(state: State) -> KerrModeResult:
            for key in (state, state.label):
                if key in resolved_modes:
                    return resolved_modes[key]
            raise KeyError(f"missing resolved mode {state.label}")
        return find(initial), find(final)
    if len(resolved_modes) != 2:
        raise ValueError("resolved_modes must contain the initial and final modes")
    first, second = resolved_modes
    if first.state == final and second.state == initial:
        first, second = second, first
    if first.state != initial or second.state != final:
        raise ValueError("resolved_modes do not match the transition states")
    return first, second


def _transition_point_id(alpha: float, initial: State, final: State) -> str:
    return f"a{alpha:.6f}_{initial.label}_{final.label}"


def compute_transition_kernel(
    config: TransitionConfig,
    resolved_modes: Sequence[KerrModeResult] | Mapping[object, KerrModeResult],
) -> TransitionKernelResult:
    """Compute the q-independent part of one atlas point.

    In particular, this function never assigns a tidal-validity status.  The
    companion separation is q-dependent and is evaluated only by
    :func:`evaluate_transition_at_q`.
    """
    point_id = _transition_point_id(config.alpha, config.initial, config.final)
    chi = float(config.chi) if config.chi is not None else math.nan
    try:
        initial_mode, final_mode = _resolved_mode_pair(
            resolved_modes, config.initial, config.final
        )
        chi = float(initial_mode.chi)
        if not initial_mode.converged or not final_mode.converged:
            raise RuntimeError("one or both resolved Kerr modes did not converge")
        if abs(initial_mode.chi - final_mode.chi) > 1.0e-12:
            raise RuntimeError("resolved Kerr modes use different background spins")
        delta_m = config.initial.m - config.final.m
        if delta_m == 0:
            return TransitionKernelResult(
                point_id, config.alpha, chi, config.initial, config.final,
                "no_prograde_resonance", "resonance", None, None,
                None, None, None, "Delta m is zero",
            )
        omega_res = (
            initial_mode.frequency_M.real - final_mode.frequency_M.real
        ) / delta_m
        if not math.isfinite(omega_res) or omega_res <= 0.0:
            return TransitionKernelResult(
                point_id, config.alpha, chi, config.initial, config.final,
                "no_prograde_resonance", "resonance", omega_res, None,
                None, None, None, "the prograde resonance frequency is non-positive",
            )
        hydrogenic = hydrogenic_newtonian_kernel_M(
            config.alpha, config.initial, config.final
        )
        semirelativistic = _kerr_newtonian_kernel(
            initial_mode, final_mode, config.settings
        )
        covariant = tidal_kernel_from_modes_M(
            initial_mode, final_mode, omega_res, config.settings, "irg"
        )
        values = (hydrogenic, semirelativistic, covariant)
        if not all(np.isfinite(value.real) and np.isfinite(value.imag) for value in values):
            raise RuntimeError("a transition kernel is non-finite")
        return TransitionKernelResult(
            point_id=point_id,
            alpha=config.alpha,
            chi=chi,
            initial=config.initial,
            final=config.final,
            kernel_status="ok",
            failure_stage=None,
            omega_res_M=float(omega_res),
            r_99_M=float(max(initial_mode.r_99_M, final_mode.r_99_M)),
            hydrogenic_kernel_M=hydrogenic,
            semirelativistic_kernel_M=semirelativistic,
            covariant_kernel_M=covariant,
            message="q-independent covariant adiabatic Kerr kernel",
        )
    except (KeyError, RuntimeError, ValueError, FloatingPointError) as error:
        return TransitionKernelResult(
            point_id, config.alpha, chi, config.initial, config.final,
            "mode_not_converged", "kernel", None, None, None, None, None,
            str(error),
        )


def _lz_quantities(kernel: complex, epsilon: float, chirp: float, delta_m: int):
    eta = epsilon * kernel
    z_value = abs(eta) ** 2 / (abs(delta_m) * chirp)
    depletion = -math.expm1(-2.0 * math.pi * z_value)
    return eta, float(z_value), float(depletion)


def evaluate_transition_at_q(
    kernel: TransitionKernelResult,
    q: float,
    error_budget: KernelErrorBudgetResult | None = None,
    *,
    tidal_radius_limit: float = 0.10,
    adiabatic_frequency_limit: float = 1.0e-2,
) -> PhenomenologyResult:
    """Restore the binary mass ratio and evaluate all three kernel layers."""
    if not math.isfinite(q) or q <= 0.0:
        raise ValueError("q must be positive and finite")
    empty = dict(
        point_id=kernel.point_id, alpha=kernel.alpha, chi=kernel.chi,
        initial=kernel.initial, final=kernel.final, q=float(q),
        separation_M=None, r_99_over_b=None, tidal_valid=False,
        adiabatic_valid=False, publication_valid=False,
        eta_hydrogenic_M=None, eta_semirelativistic_M=None,
        eta_covariant_M=None, z_hydrogenic=None, z_semirelativistic=None,
        z_covariant=None, depletion_hydrogenic=None,
        depletion_semirelativistic=None, depletion_covariant=None,
        power_ratio_covariant_to_hydrogenic=None,
        error_lower=None, error_upper=None,
        eta_covariant_abs_lower=None, eta_covariant_abs_upper=None,
        z_covariant_lower=None, z_covariant_upper=None,
        power_ratio_lower=None, power_ratio_upper=None,
    )
    if kernel.kernel_status != "ok" or kernel.omega_res_M is None:
        return PhenomenologyResult(status=kernel.kernel_status, **empty)
    omega = kernel.omega_res_M
    separation = (1.0 + q) ** (1.0 / 3.0) * omega ** (-2.0 / 3.0)
    radius_ratio = float(kernel.r_99_M / separation)
    tidal_valid = radius_ratio <= tidal_radius_limit
    adiabatic_valid = omega <= adiabatic_frequency_limit
    status = "ok" if tidal_valid and adiabatic_valid else (
        "tidal_expansion_invalid" if not tidal_valid else "adiabatic_tide_invalid"
    )
    epsilon = q / (1.0 + q) * omega**2
    chirp = (
        96.0 / 5.0 * q / (1.0 + q) ** (1.0 / 3.0)
        * omega ** (11.0 / 3.0)
    )
    delta_m = kernel.initial.m - kernel.final.m
    h_eta, h_z, h_dep = _lz_quantities(
        complex(kernel.hydrogenic_kernel_M), epsilon, chirp, delta_m
    )
    s_eta, s_z, s_dep = _lz_quantities(
        complex(kernel.semirelativistic_kernel_M), epsilon, chirp, delta_m
    )
    c_eta, c_z, c_dep = _lz_quantities(
        complex(kernel.covariant_kernel_M), epsilon, chirp, delta_m
    )
    power_ratio = abs(kernel.covariant_kernel_M / kernel.hydrogenic_kernel_M) ** 2
    lower = upper = None
    eta_lower = eta_upper = z_lower = z_upper = None
    power_lower = power_upper = None
    if error_budget is not None and error_budget.complete:
        error_abs = float(error_budget.systematic_error_abs or 0.0)
        central_abs = abs(kernel.covariant_kernel_M)
        lower_kernel = max(0.0, central_abs - error_abs)
        upper_kernel = central_abs + error_abs
        low_eta, z_lower, lower = _lz_quantities(
            lower_kernel, epsilon, chirp, delta_m
        )
        high_eta, z_upper, upper = _lz_quantities(
            upper_kernel, epsilon, chirp, delta_m
        )
        eta_lower, eta_upper = abs(low_eta), abs(high_eta)
        hydrogenic_abs = abs(complex(kernel.hydrogenic_kernel_M))
        if hydrogenic_abs > 0.0:
            power_lower = (lower_kernel / hydrogenic_abs) ** 2
            power_upper = (upper_kernel / hydrogenic_abs) ** 2
    return PhenomenologyResult(
        point_id=kernel.point_id, alpha=kernel.alpha, chi=kernel.chi,
        initial=kernel.initial, final=kernel.final, q=float(q), status=status,
        separation_M=float(separation), r_99_over_b=radius_ratio,
        tidal_valid=tidal_valid, adiabatic_valid=adiabatic_valid,
        publication_valid=(tidal_valid and adiabatic_valid),
        eta_hydrogenic_M=h_eta, eta_semirelativistic_M=s_eta,
        eta_covariant_M=c_eta, z_hydrogenic=h_z,
        z_semirelativistic=s_z, z_covariant=c_z,
        depletion_hydrogenic=h_dep, depletion_semirelativistic=s_dep,
        depletion_covariant=c_dep,
        power_ratio_covariant_to_hydrogenic=float(power_ratio),
        error_lower=lower, error_upper=upper,
        eta_covariant_abs_lower=eta_lower,
        eta_covariant_abs_upper=eta_upper,
        z_covariant_lower=z_lower,
        z_covariant_upper=z_upper,
        power_ratio_lower=power_lower,
        power_ratio_upper=power_upper,
    )


def _next_grid_value(value: float | int, grid: Sequence[float | int]):
    for candidate in grid:
        if candidate > value:
            return candidate
    return grid[-1]


def compute_kernel_error_budget(
    config: TransitionConfig,
    resolved_modes: Sequence[KerrModeResult] | Mapping[object, KerrModeResult],
) -> KernelErrorBudgetResult:
    """Measure a paired one-at-a-time numerical error envelope.

    Each row is a real rerun or an independently measured operator-form
    residual.  Missing or non-finite variants make the budget incomplete;
    there is no analytic ``alpha`` proxy or silent zero fallback.
    """
    pair = _resolved_mode_pair(resolved_modes, config.initial, config.final)
    central_result = compute_transition_kernel(config, pair)
    if central_result.kernel_status != "ok":
        return KernelErrorBudgetResult(
            central_result.point_id, "error_budget_incomplete", (), None, None,
            None, None, False,
        )
    central = complex(central_result.covariant_kernel_M)
    scale = max(abs(central), 1.0e-300)
    settings = config.settings
    source_rows: list[dict[str, object]] = []

    def add_source(name: str, value: complex, detail: str) -> None:
        shift = value - central
        source_rows.append({
            "source": name,
            "status": "ok" if np.isfinite(value.real) and np.isfinite(value.imag) else "failed",
            "variant_kernel_real": value.real,
            "variant_kernel_imag": value.imag,
            "signed_shift_real": shift.real,
            "signed_shift_imag": shift.imag,
            "absolute_shift": abs(shift),
            "relative_shift": abs(shift) / scale,
            "detail": detail,
        })

    def kernel_for(tide: RelativisticTideSettings, *, resolve: bool = False,
                   radial_strategy: str = "gauss",
                   operator_form: str = "divergence") -> complex:
        modes = pair
        if resolve:
            modes = tuple(
                _cached_mode(
                    config.alpha, central_result.chi, state, tide.mode
                )
                for state in (config.initial, config.final)
            )
            if not all(mode.converged for mode in modes):
                raise RuntimeError("variant mode solve did not converge")
        return tidal_kernel_from_modes_M(
            modes[0], modes[1], float(central_result.omega_res_M), tide, "irg",
            radial_strategy=radial_strategy, operator_form=operator_form,
        )

    variants: list[tuple[str, RelativisticTideSettings, bool, str]] = []
    current_n = max(
        settings.mode.truncation,
        *(int(mode.selected_truncation or 0) for mode in pair),
    )
    next_n = int(_next_grid_value(current_n, (400, 600, 800, 1200, 1600)))
    variants.append(("mode_N", replace(settings, mode=replace(
        settings.mode, truncation=next_n, series_terms=next_n
    )), True, f"N={next_n}"))
    next_lmax = int(_next_grid_value(settings.mode.angular_lmax, (14, 18)))
    variants.append(("angular_lmax", replace(settings, mode=replace(
        settings.mode, angular_lmax=next_lmax
    )), True, f"lmax={next_lmax}"))
    next_radial = int(_next_grid_value(settings.radial_nodes, (56, 80, 112)))
    variants.append(("radial_nodes", replace(settings, radial_nodes=next_radial), False,
                     f"radial_nodes={next_radial}"))
    next_angular = int(_next_grid_value(settings.angular_nodes, (72, 96, 128)))
    variants.append(("angular_nodes", replace(settings, angular_nodes=next_angular), False,
                     f"angular_nodes={next_angular}"))
    current_cutoff = settings.horizon_cutoffs[0]
    cutoff_grid = (1.0e-4, 3.0e-5, 1.0e-5)
    next_cutoff = next((item for item in cutoff_grid if item < current_cutoff), cutoff_grid[-1])
    variants.append(("horizon_cutoff", replace(
        settings,
        mode=replace(settings.mode, horizon_cutoff=next_cutoff),
        horizon_cutoffs=(next_cutoff,),
    ), True, f"cutoff={next_cutoff:.1e}"))
    next_counterterm = int(_next_grid_value(settings.mode.counterterm_order, (3, 4, 5)))
    variants.append(("counterterm", replace(settings, mode=replace(
        settings.mode, counterterm_order=next_counterterm
    )), True, f"counterterm={next_counterterm}"))
    next_outer = float(_next_grid_value(settings.mode.outer_decay_lengths, (28.0, 40.0)))
    variants.append(("outer_range", replace(settings, mode=replace(
        settings.mode, outer_decay_lengths=next_outer
    )), True, f"outer_decay_lengths={next_outer:g}"))

    for name, variant, resolve, detail in variants:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", IntegrationWarning)
                value = kernel_for(variant, resolve=resolve)
            integration = [str(item.message) for item in caught
                           if issubclass(item.category, IntegrationWarning)]
            if integration:
                raise RuntimeError("; ".join(integration))
            add_source(name, value, detail)
        except (RuntimeError, ValueError, FloatingPointError) as error:
            source_rows.append({
                "source": name, "status": "failed", "variant_kernel_real": None,
                "variant_kernel_imag": None, "signed_shift_real": None,
                "signed_shift_imag": None, "absolute_shift": None,
                "relative_shift": None, "detail": str(error),
            })
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IntegrationWarning)
            value = kernel_for(settings, radial_strategy="quad")
        integration = [str(item.message) for item in caught
                       if issubclass(item.category, IntegrationWarning)]
        if integration:
            raise RuntimeError("; ".join(integration))
        add_source("radial_strategy", value,
                   "segmented Gauss-Legendre versus scipy.integrate.quad")
    except (RuntimeError, ValueError, FloatingPointError) as error:
        source_rows.append({
            "source": "radial_strategy", "status": "failed",
            "variant_kernel_real": None, "variant_kernel_imag": None,
            "signed_shift_real": None, "signed_shift_imag": None,
            "absolute_shift": None, "relative_shift": None, "detail": str(error),
        })

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IntegrationWarning)
            value = kernel_for(settings, operator_form="connection")
        integration = [str(item.message) for item in caught
                       if issubclass(item.category, IntegrationWarning)]
        if integration:
            raise RuntimeError("; ".join(integration))
        add_source(
            "delta_box_form",
            value,
            "full-kernel connection form versus production divergence form",
        )
    except (RuntimeError, ValueError, FloatingPointError) as error:
        source_rows.append({
            "source": "delta_box_form", "status": "failed",
            "variant_kernel_real": None, "variant_kernel_imag": None,
            "signed_shift_real": None, "signed_shift_imag": None,
            "absolute_shift": None, "relative_shift": None, "detail": str(error),
        })

    complete = len(source_rows) == 9 and all(row["status"] == "ok" for row in source_rows)
    if not complete:
        return KernelErrorBudgetResult(
            central_result.point_id, "error_budget_incomplete", tuple(source_rows),
            None, None, None, None, False,
        )
    relatives = [float(row["relative_shift"]) for row in source_rows]
    worst_index = int(np.argmax(relatives))
    worst_relative = relatives[worst_index]
    return KernelErrorBudgetResult(
        point_id=central_result.point_id,
        status="ok",
        sources=tuple(source_rows),
        rss_relative=float(math.sqrt(sum(value * value for value in relatives))),
        worst_relative=worst_relative,
        worst_source=str(source_rows[worst_index]["source"]),
        systematic_error_abs=float(worst_relative * scale),
        complete=True,
    )


def direct_ward_audit(
    config: TransitionConfig,
    resolved_modes: Sequence[KerrModeResult] | Mapping[object, KerrModeResult],
    specs: Sequence[GaugeVectorSpec],
    *,
    include_off_resonance: bool = True,
    physical_kernel: complex | None = None,
):
    """Public production Ward audit, imported lazily to avoid a module cycle."""
    from .certification import direct_ward_audit as run_direct_ward_audit

    return run_direct_ward_audit(
        config, resolved_modes, specs,
        include_off_resonance=include_off_resonance,
        physical_kernel=physical_kernel,
    )


def compute_relativistic_transition(config: TransitionConfig) -> TransitionResult:
    settings = config.settings
    try:
        chi = config.chi
        if chi is None:
            chi = saturation_spin_cf(config.alpha, config.initial, settings.mode)
        initial_mode = _cached_mode(config.alpha, chi, config.initial, settings.mode)
        final_mode = _cached_mode(config.alpha, chi, config.final, settings.mode)
        if not initial_mode.converged or not final_mode.converged:
            raise RuntimeError("mode convergence failed")
        delta_m = config.initial.m - config.final.m
        if delta_m == 0:
            raise RuntimeError("transition has Delta m=0 and no rotating-tide resonance")
        omega_res = (
            initial_mode.frequency_M.real - final_mode.frequency_M.real
        ) / delta_m
        if omega_res <= 0.0:
            return TransitionResult(
                config.alpha, chi, config.q, config.initial, config.final,
                "no_prograde_resonance", omega_res, math.nan, math.nan,
                math.nan, complex(np.nan), complex(np.nan), complex(np.nan),
                complex(np.nan), math.nan, math.nan, math.inf, math.inf,
                "Re(omega_i-omega_f)/Delta m is non-positive"
            )
        status = "ok"
        separation = (1.0 + config.q) ** (1.0 / 3.0) * omega_res ** (-2.0 / 3.0)
        r_99_over_b = max(initial_mode.r_99_M, final_mode.r_99_M) / separation
        if r_99_over_b > settings.tidal_radius_limit:
            status = "tidal_expansion_invalid"
        if omega_res > settings.adiabatic_frequency_limit:
            status = "adiabatic_tide_invalid"
        hydrogenic = hydrogenic_newtonian_kernel_M(
            config.alpha, config.initial, config.final
        )
        semirelativistic = _kerr_newtonian_kernel(initial_mode, final_mode)
        covariant = covariant_tidal_kernel_M(
            config.alpha, chi, config.initial, config.final, omega_res, settings
        )
        epsilon = config.q / (1.0 + config.q) * omega_res**2
        eta = epsilon * covariant
        chirp = (
            96.0
            / 5.0
            * config.q
            / (1.0 + config.q) ** (1.0 / 3.0)
            * omega_res ** (11.0 / 3.0)
        )
        z_value = abs(eta) ** 2 / (abs(delta_m) * chirp)
        depletion = -math.expm1(-2.0 * math.pi * z_value)
        numerical = abs(covariant) * max(
            initial_mode.cf_residual,
            final_mode.cf_residual,
            initial_mode.radial_residual,
            final_mode.radial_residual,
        )
        measured_budget = compute_kernel_error_budget(
            replace(config, chi=chi), (initial_mode, final_mode)
        )
        systematic = (
            float(measured_budget.systematic_error_abs)
            if measured_budget.complete
            and measured_budget.systematic_error_abs is not None
            else math.inf
        )
        if not measured_budget.complete:
            status = "error_budget_incomplete"
        return TransitionResult(
            config.alpha, chi, config.q, config.initial, config.final, status,
            omega_res, separation, epsilon, r_99_over_b, hydrogenic,
            semirelativistic, covariant, eta, z_value, depletion, numerical,
            systematic, "covariant adiabatic IRG tide; global binary effects excluded"
        )
    except RuntimeError as error:
        chi_value = float(config.chi) if config.chi is not None else math.nan
        return TransitionResult(
            config.alpha, chi_value, config.q, config.initial, config.final,
            "mode_not_converged", math.nan, math.nan, math.nan, math.nan,
            complex(np.nan), complex(np.nan), complex(np.nan), complex(np.nan),
            math.nan, math.nan, math.inf, math.inf, str(error)
        )


def scan_relativistic_atlas(config: AtlasConfig) -> list[TransitionResult]:
    results: list[TransitionResult] = []
    for alpha in config.alphas:
        for initial, final in TRANSITION_CHANNELS:
            results.append(
                compute_relativistic_transition(
                    TransitionConfig(
                        alpha=float(alpha), initial=initial, final=final,
                        q=float(config.q_values[0]), settings=config.settings
                    )
                )
            )
    return results


def gauge_audit_transition(config: TransitionConfig) -> GaugeAuditResult:
    """Backward-compatible summary of the production direct Ward audit."""
    chi = config.chi
    if chi is None:
        chi = saturation_spin_cf(config.alpha, config.initial, config.settings.mode)
    initial_mode = _cached_mode(config.alpha, chi, config.initial, config.settings.mode)
    final_mode = _cached_mode(config.alpha, chi, config.final, config.settings.mode)
    delta_m = config.initial.m - config.final.m
    if delta_m == 0:
        raise ValueError("gauge audit requires a nonzero Delta m")
    omega_res = (
        initial_mode.frequency_M.real - final_mode.frequency_M.real
    ) / delta_m
    r_plus = 1.0 + math.sqrt(1.0 - chi**2)
    sample_radii = (r_plus + 0.1, r_plus + 1.0, r_plus + 5.0)
    residuals = [
        irg_algebraic_residuals(radius, 1.1, chi, omega_res)
        for radius in sample_radii
    ]
    trace = max(value[0] for value in residuals)
    tetrad = max(value[1] for value in residuals)
    cloud_inner = max(r_plus + 0.1, 0.2 * max(initial_mode.r_99_M, final_mode.r_99_M))
    cloud_outer = max(cloud_inner + 1.0, 0.7 * max(initial_mode.r_99_M, final_mode.r_99_M))
    supports = (
        ("near_horizon", r_plus + 2.0, r_plus + 10.0),
        ("cloud_core", cloud_inner, cloud_outer),
    )
    specs = tuple(
        GaugeVectorSpec(kind, left, right, amplitude, support_name=name)
        for kind in ("temporal", "radial", "polar", "axial")
        for name, left, right in supports
        for amplitude in (0.1, 1.0)
    )
    ward_result = direct_ward_audit(
        replace(config, chi=chi), (initial_mode, final_mode), specs,
        include_off_resonance=False,
    )
    from .certification import operator_form_audit

    operator_residual, _ = operator_form_audit(
        replace(config, chi=chi), resolved_modes=(initial_mode, final_mode)
    )
    ward = max(
        ward_result.maximum_invariance_residual,
        ward_result.maximum_pure_gauge_residual,
        ward_result.maximum_linearity_residual,
    )
    status = "ok" if (
        ward_result.status == "ok"
        and operator_residual < 1.0e-10
        and trace < 1.0e-10
        and tetrad < 1.0e-10
    ) else "gauge_audit_failed"
    # The finite-frequency RW implementation is not a valid Kerr atlas
    # control.  Schwarzschild IRG/RW remains exclusively an Omega->0
    # weak-field certification observable.
    schwarzschild = None
    return GaugeAuditResult(
        config.alpha, chi, config.initial, config.final, status, trace, tetrad,
        float(operator_residual), ward, schwarzschild
    )
