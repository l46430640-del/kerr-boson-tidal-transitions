"""Submission-gate checks for high-order tides and independent Kerr kernels.

The multipole path keeps the exact inside/outside point-companion expansion.
It is deliberately Newtonian: it measures the spatial multipole truncation of
the covariant quadrupole calculation rather than pretending to be a global
binary metric.
"""

from __future__ import annotations

from dataclasses import replace
import cmath
import math
from typing import Mapping, Sequence
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad, solve_bvp, solve_ivp
from scipy.optimize import minimize_scalar
from scipy.special import sph_harm_y

from .models import (
    ContourKernelSettings,
    ContourModeResult,
    HighOrderTideBudgetResult,
    IndependentKernelResult,
    KerrModeResult,
    MultipoleEtaResult,
    MultipoleKernelResult,
    MultipoleTideSettings,
    State,
    TransitionConfig,
)
from .relativity import angular_mode_values, build_radial_ode_evaluator
from .relativistic_tides import (
    _bl_tidal_kernel,
    _complex_fsum,
    _kerr_newtonian_kernel,
    _qbs_decay_rate,
    compute_transition_kernel,
    hydrogenic_newtonian_kernel_M,
)


_INDEPENDENT_MODE_PROVIDERS: dict[tuple[object, ...], object] = {}
_MULTIPOLE_CONTEXT_CACHE: dict[tuple[object, ...], tuple[object, ...]] = {}
_MULTIPOLE_FULL_KERNEL_CACHE: dict[tuple[object, ...], complex] = {}


class _RegularAngularSolution:
    def __init__(self, solution, power: int):
        self._solution = solution
        self.power = power
        self.rms_residuals = solution.rms_residuals

    def sol(self, theta):
        theta_array = np.asarray(theta, dtype=float)
        regular = self._solution.sol(theta_array)
        sine = np.sin(theta_array)
        cosine = np.cos(theta_array)
        factor = sine**self.power
        value = factor * regular[0]
        first = factor * (
            regular[1]
            + self.power * cosine / np.maximum(sine, 1.0e-300) * regular[0]
        )
        return np.vstack((value, first))


def _resolved_pair(
    resolved_modes: Sequence[KerrModeResult] | Mapping[object, KerrModeResult],
    initial: State,
    final: State,
) -> tuple[KerrModeResult, KerrModeResult]:
    if isinstance(resolved_modes, Mapping):
        def pick(state: State) -> KerrModeResult:
            for key in (state, state.label):
                if key in resolved_modes:
                    return resolved_modes[key]
            raise KeyError(f"missing resolved mode {state.label}")
        return pick(initial), pick(final)
    if len(resolved_modes) != 2:
        raise ValueError("resolved_modes must contain initial and final modes")
    pair = (resolved_modes[0], resolved_modes[1])
    if pair[0].state != initial or pair[1].state != final:
        raise ValueError("resolved modes do not match transition states")
    return pair


def _point_id(config: TransitionConfig) -> str:
    return f"a{config.alpha:.6f}_{config.initial.label}_{config.final.label}"


def _resonance_frequency(pair: tuple[KerrModeResult, KerrModeResult]) -> float:
    delta_m = pair[0].state.m - pair[1].state.m
    if delta_m == 0:
        raise ValueError("transition has zero Delta m")
    return float((pair[0].frequency_M.real - pair[1].frequency_M.real) / delta_m)


def _separation(omega: float, q: float, orbital_model: str) -> float:
    base = (1.0 + q) ** (1.0 / 3.0) * omega ** (-2.0 / 3.0)
    if orbital_model == "newtonian":
        return float(base)
    if orbital_model != "1pn":
        raise ValueError("orbital_model must be 'newtonian' or '1pn'")
    symmetric_ratio = q / (1.0 + q) ** 2
    x_value = ((1.0 + q) * omega) ** (2.0 / 3.0)
    return float(base * (1.0 + (-3.0 + symmetric_ratio) * x_value / 3.0))


def _chirp(omega: float, q: float, orbital_model: str) -> float:
    value = 96.0 / 5.0 * q / (1.0 + q) ** (1.0 / 3.0) * omega ** (11.0 / 3.0)
    if orbital_model == "newtonian":
        return float(value)
    symmetric_ratio = q / (1.0 + q) ** 2
    x_value = ((1.0 + q) * omega) ** (2.0 / 3.0)
    correction = 1.0 - (743.0 / 336.0 + 11.0 * symmetric_ratio / 4.0) * x_value
    return float(value * correction)


def _depletion(eta: complex, chirp: float, delta_m: int) -> float:
    z_value = abs(eta) ** 2 / (abs(delta_m) * chirp)
    return float(-math.expm1(-2.0 * math.pi * z_value))


def _complex_quad(function, lower: float, upper: float, rtol: float) -> complex:
    if upper <= lower:
        return 0.0j
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        real = quad(lambda x: float(np.real(function(x))), lower, upper,
                    epsabs=1.0e-11, epsrel=rtol, limit=400)[0]
        imag = quad(lambda x: float(np.imag(function(x))), lower, upper,
                    epsabs=1.0e-11, epsrel=rtol, limit=400)[0]
    messages = [str(item.message) for item in caught
                if issubclass(item.category, IntegrationWarning)]
    if messages:
        raise RuntimeError("multipole quadrature warning: " + "; ".join(messages))
    return complex(real, imag)


def _gauss_integral(function, lower: float, upper: float, nodes: int) -> complex:
    if upper <= lower:
        return 0.0j
    abscissa, weights = np.polynomial.legendre.leggauss(nodes)
    radius = lower + 0.5 * (upper - lower) * (abscissa + 1.0)
    values = np.asarray(function(radius), dtype=complex)
    return complex(0.5 * (upper - lower) * np.sum(weights * values))


def _multipole_kernel(
    config: TransitionConfig,
    pair: tuple[KerrModeResult, KerrModeResult],
    ell: int,
    harmonic_m: int,
    separation_M: float,
    settings: MultipoleTideSettings,
    strategy: str,
) -> tuple[complex, complex]:
    if ell < 2 or abs(harmonic_m) > ell:
        raise ValueError("invalid tidal multipole")
    if harmonic_m != pair[0].state.m - pair[1].state.m:
        return 0.0j, 0.0j
    source_coefficient = complex(
        -4.0 * math.pi / (2 * ell + 1)
        * np.conjugate(sph_harm_y(ell, harmonic_m, math.pi / 2.0, 0.0))
    )
    context_key = (
        id(pair[0]), id(pair[1]), float(config.settings.mode.outer_decay_lengths),
        float(config.settings.horizon_cutoffs[0]), int(settings.angular_nodes),
        int(ell), int(harmonic_m),
    )
    context = _MULTIPOLE_CONTEXT_CACHE.get(context_key)
    if context is None:
        r_plus = 1.0 + math.sqrt(1.0 - pair[0].chi**2)
        cutoff = config.settings.horizon_cutoffs[0]
        decay_i = _qbs_decay_rate(config.alpha, pair[0].frequency_M)
        decay_f = _qbs_decay_rate(config.alpha, pair[1].frequency_M)
        outer = (
            r_plus + config.settings.mode.outer_decay_lengths
            / min(decay_i, decay_f)
        )
        common_span = 1.001 * (outer - r_plus)
        radial_i = build_radial_ode_evaluator(
            config.alpha, pair[0].chi, pair[0].state, pair[0].frequency_M,
            pair[0].separation_constant, pair[0].radial_coefficients,
            outer_decay_lengths=common_span * decay_i,
        )
        radial_f = build_radial_ode_evaluator(
            config.alpha, pair[1].chi, pair[1].state, pair[1].frequency_M,
            pair[1].separation_constant, pair[1].radial_coefficients,
            outer_decay_lengths=common_span * decay_f,
        )

        def product(radius):
            return radial_i(radius)[0] * radial_f(radius)[0]

        u_nodes, u_weights = np.polynomial.legendre.leggauss(
            settings.angular_nodes
        )
        theta = np.arccos(u_nodes)
        angular_i = angular_mode_values(
            theta, pair[0].state, pair[0].separation_constant,
            pair[0].angular_l_values, pair[0].angular_coefficients,
        )[0]
        angular_f = angular_mode_values(
            theta, pair[1].state, pair[1].separation_constant,
            pair[1].angular_l_values, pair[1].angular_coefficients,
        )[0]
        tidal_harmonic = sph_harm_y(ell, harmonic_m, theta, 0.0)
        angular = _complex_fsum(
            [complex(w * sf * si * y) for w, sf, si, y in
             zip(u_weights, angular_f, angular_i, tidal_harmonic, strict=True)]
        )
        kg_to_schrodinger = 2.0 * cmath.sqrt(
            pair[0].frequency_M * pair[1].frequency_M
        )
        context = (
            r_plus + cutoff, outer, product, angular, kg_to_schrodinger
        )
        _MULTIPOLE_CONTEXT_CACHE[context_key] = context
    lower, outer, product, angular, kg_to_schrodinger = context
    split = min(max(separation_M, lower), outer)
    if strategy == "gauss":
        integrate = lambda f, a, b: _gauss_integral(f, a, b, settings.radial_nodes)
    elif strategy == "quad":
        integrate = lambda f, a, b: _complex_quad(f, a, b, settings.radial_rtol)
    else:
        raise ValueError("strategy must be gauss or quad")
    full_key = context_key + (strategy, int(settings.radial_nodes), float(settings.radial_rtol))
    radial = None
    if separation_M >= outer:
        radial = _MULTIPOLE_FULL_KERNEL_CACHE.get(full_key)
    if radial is None:
        inside = integrate(lambda r: r ** (ell + 2) * product(r), lower, split)
        outside = integrate(lambda r: r ** (1 - ell) * product(r), split, outer)
        radial = inside + separation_M ** (2 * ell + 1) * outside
        if separation_M >= outer:
            _MULTIPOLE_FULL_KERNEL_CACHE[full_key] = complex(radial)
    kernel = (
        config.alpha * 2.0 * math.pi * source_coefficient
        * radial * angular * kg_to_schrodinger
    )
    return complex(kernel), source_coefficient


def newtonian_multipole_kernel_M(
    config: TransitionConfig,
    resolved_modes: Sequence[KerrModeResult] | Mapping[object, KerrModeResult],
    ell: int,
    harmonic_m: int,
) -> complex:
    """Return the exact inside/outside multipole kernel at ``config.q``."""
    pair = _resolved_pair(resolved_modes, config.initial, config.final)
    omega = _resonance_frequency(pair)
    separation = _separation(omega, config.q, "newtonian")
    settings = MultipoleTideSettings(
        radial_nodes=max(112, config.settings.radial_nodes),
        angular_nodes=max(128, config.settings.angular_nodes),
    )
    return _multipole_kernel(
        config, pair, ell, harmonic_m, separation, settings, "gauss"
    )[0]


def _eta_row(
    config: TransitionConfig,
    pair: tuple[KerrModeResult, KerrModeResult],
    q: float,
    orbital_model: str,
    settings: MultipoleTideSettings,
    covariant_kernel: complex,
    hydrogenic_kernel: complex,
) -> MultipoleEtaResult:
    omega = _resonance_frequency(pair)
    separation = _separation(omega, q, orbital_model)
    partial: dict[int, complex] = {}
    running = 0.0j
    quadrupole = 0.0j
    for ell in range(2, settings.ell_max + 1):
        kernel, _ = _multipole_kernel(
            replace(config, q=q), pair, ell,
            pair[0].state.m - pair[1].state.m,
            separation, settings, "gauss",
        )
        contribution = q / separation ** (ell + 1) * kernel
        running += contribution
        partial[ell] = running
        if ell == 2:
            quadrupole = contribution
    eta_l8 = partial[min(8, settings.ell_max)]
    eta_l10 = partial[settings.ell_max]
    shift = abs(eta_l10 - quadrupole)
    tail = abs(eta_l10 - eta_l8)
    sigma_mult = shift + 2.0 * tail
    sigma_covariant = 2.0 * shift
    epsilon_2 = q / separation**3
    corrected = epsilon_2 * covariant_kernel + (eta_l10 - quadrupole)
    chirp = _chirp(omega, q, orbital_model)
    delta_m = pair[0].state.m - pair[1].state.m
    h_eta = epsilon_2 * hydrogenic_kernel
    depletion_h = _depletion(h_eta, chirp, delta_m)
    depletion_exact = _depletion(eta_l10, chirp, delta_m)
    depletion_corrected = _depletion(corrected, chirp, delta_m)
    radius_ratio = max(pair[0].r_99_M, pair[1].r_99_M) / separation
    status = "ok" if radius_ratio <= 0.10 and omega <= 1.0e-2 else (
        "tidal_expansion_invalid" if radius_ratio > 0.10
        else "adiabatic_tide_invalid"
    )
    return MultipoleEtaResult(
        _point_id(config), float(q), orbital_model, separation, epsilon_2,
        quadrupole, eta_l10, corrected, eta_l8, eta_l10, shift, tail,
        sigma_mult, sigma_covariant, float(radius_ratio),
        bool(radius_ratio <= 0.05), bool(radius_ratio <= 0.07),
        bool(radius_ratio <= 0.10), chirp,
        depletion_h, depletion_exact, depletion_corrected,
        abs(depletion_corrected - depletion_h), status,
    )


def newtonian_exact_companion_eta_M(
    config: TransitionConfig,
    resolved_modes: Sequence[KerrModeResult] | Mapping[object, KerrModeResult],
    q: float,
    l_max: int = 10,
) -> MultipoleEtaResult:
    pair = _resolved_pair(resolved_modes, config.initial, config.final)
    settings = MultipoleTideSettings(ell_max=l_max)
    kernel = compute_transition_kernel(config, pair)
    if kernel.kernel_status != "ok":
        raise RuntimeError(kernel.message or kernel.kernel_status)
    return _eta_row(
        config, pair, q, "newtonian", settings,
        complex(kernel.covariant_kernel_M),
        complex(kernel.hydrogenic_kernel_M),
    )


def compute_high_order_tide_budget(
    config: TransitionConfig,
    resolved_modes: Sequence[KerrModeResult] | Mapping[object, KerrModeResult],
    q_values: Sequence[float],
    settings: MultipoleTideSettings = MultipoleTideSettings(),
) -> HighOrderTideBudgetResult:
    pair = _resolved_pair(resolved_modes, config.initial, config.final)
    kernel = compute_transition_kernel(config, pair)
    if kernel.kernel_status != "ok":
        return HighOrderTideBudgetResult(
            _point_id(config), kernel.kernel_status, (), math.inf, math.inf,
            None, None, kernel.message,
        )
    rows: list[MultipoleEtaResult] = []
    for q in q_values:
        for orbital_model in ("newtonian", "1pn"):
            rows.append(_eta_row(
                config, pair, float(q), orbital_model, settings,
                complex(kernel.covariant_kernel_M),
                complex(kernel.hydrogenic_kernel_M),
            ))
    physical = [row for row in rows if row.status == "ok"]
    relative = [
        row.multipole_shift_abs / max(abs(row.eta_quadrupole_M), 1.0e-300)
        for row in physical
    ]
    tail = [
        row.multipole_tail_abs / max(abs(row.eta_l10_M), 1.0e-300)
        for row in physical
    ]
    robust = [
        row.q for row in physical
        if row.r_99_over_b <= 0.07 and row.depletion_change >= 0.10
    ]
    maximum_shift = max(relative, default=math.inf)
    maximum_tail = max(tail, default=math.inf)
    status = "ok" if maximum_tail < 1.0e-4 else "multipole_not_converged"
    return HighOrderTideBudgetResult(
        _point_id(config), status, tuple(rows), maximum_shift, maximum_tail,
        min(robust) if robust else None, max(robust) if robust else None,
        "exact Newtonian point-companion multipoles; covariant higher multipoles are enveloped",
    )


def multipole_kernel_result(
    config: TransitionConfig,
    resolved_modes: Sequence[KerrModeResult] | Mapping[object, KerrModeResult],
    ell: int,
    settings: MultipoleTideSettings = MultipoleTideSettings(),
) -> MultipoleKernelResult:
    pair = _resolved_pair(resolved_modes, config.initial, config.final)
    omega = _resonance_frequency(pair)
    separation = _separation(omega, config.q, "newtonian")
    harmonic_m = pair[0].state.m - pair[1].state.m
    gauss, coefficient = _multipole_kernel(
        config, pair, ell, harmonic_m, separation, settings, "gauss"
    )
    quad_value, _ = _multipole_kernel(
        config, pair, ell, harmonic_m, separation, settings, "quad"
    )
    difference = abs(gauss - quad_value) / max(abs(quad_value), 1.0e-300)
    status = "ok" if difference < 1.0e-4 else "multipole_not_converged"
    return MultipoleKernelResult(
        _point_id(config), ell, harmonic_m, coefficient, gauss, difference,
        status, "full r<b/r>b radial branches",
    )


# The independent contour path is implemented below the multipole machinery so
# it can share only immutable result types, never the production mode provider.


def _angular_bvp(
    alpha: float,
    chi: float,
    state: State,
    omega: complex,
    settings: ContourKernelSettings,
) -> tuple[complex, object, float]:
    eps = settings.angular_epsilon
    theta = np.linspace(eps, math.pi - eps, settings.angular_nodes)
    power = abs(state.m)
    spherical, spherical_derivative = sph_harm_y(
        state.l, state.m, theta, 0.0, diff_n=1
    )
    factor = np.sin(theta)**power
    guess = spherical / factor
    derivative = (
        spherical_derivative[..., 0]
        - power * np.cos(theta) / np.sin(theta) * spherical
    ) / factor
    guess_scale = guess[0]
    guess /= guess_scale
    derivative /= guess_scale
    y_guess = np.vstack((guess.astype(complex), derivative.astype(complex)))
    eigen_guess = complex(state.l * (state.l + 1), 0.0)
    c_squared = chi * chi * (omega * omega - alpha * alpha)

    def equation(x, y, parameter):
        sine = np.sin(x)
        angular = parameter[0]
        second = (
            -(2 * power + 1) * np.cos(x) / sine * y[1]
            - (angular - power * (power + 1)
               + c_squared * np.cos(x)**2) * y[0]
        )
        return np.vstack((y[1], second))

    def boundary(left, right, parameter):
        endpoint_coefficient = (
            parameter[0] - power * (power + 1) + c_squared
        ) / (2.0 * (power + 1))
        return np.asarray([
            left[0] - 1.0,
            left[1] + endpoint_coefficient * eps * left[0],
            right[1] - endpoint_coefficient * eps * right[0],
        ], dtype=complex)

    solution = solve_bvp(
        equation, boundary, theta, y_guess, p=np.asarray([eigen_guess]),
        tol=1.0e-9, max_nodes=12000, verbose=0,
    )
    if not solution.success:
        raise RuntimeError(f"angular BVP failed: {solution.message}")
    wrapped = _RegularAngularSolution(solution, power)
    residual = float(np.max(solution.rms_residuals))
    return complex(solution.p[0]), wrapped, float(residual)


def _radial_equation(alpha, chi, state, omega, angular):
    def equation(radius, vector):
        delta = radius * radius - 2.0 * radius + chi * chi
        potential = (
            (omega**2 * (radius * radius + chi * chi)**2
             - 4.0 * radius * chi * state.m * omega + state.m**2 * chi**2) / delta
            - omega**2 * chi**2 - alpha**2 * radius**2 - angular
        )
        return np.asarray([
            vector[1],
            -(2.0 * (radius - 1.0) * vector[1] + potential * vector[0]) / delta,
        ], dtype=complex)
    return equation


def _horizon_frobenius_coefficients(
    alpha: float, chi: float, state: State, omega: complex,
    angular: complex, order: int,
) -> tuple[complex, np.ndarray]:
    r_plus = 1.0 + math.sqrt(1.0 - chi * chi)
    gap = 2.0 * math.sqrt(1.0 - chi * chi)
    sigma = 2.0 * r_plus * (omega - state.m * chi / (2.0 * r_plus)) / gap
    exponent = -1j * sigma
    radius = np.asarray([r_plus, 1.0], dtype=complex)
    delta = np.asarray([0.0, gap, 1.0], dtype=complex)
    delta_prime = np.asarray([gap, 2.0], dtype=complex)
    radius_squared = np.polynomial.polynomial.polymul(radius, radius)
    r2a2 = radius_squared.copy(); r2a2[0] += chi * chi
    p_poly = omega**2 * np.polynomial.polynomial.polymul(r2a2, r2a2)
    p_poly = np.polynomial.polynomial.polyadd(
        p_poly, -4.0 * chi * state.m * omega * radius
    )
    p_poly[0] += state.m**2 * chi**2
    q_poly = -alpha**2 * radius_squared
    q_poly[0] += -omega**2 * chi**2 - angular
    a_poly = np.polynomial.polynomial.polymul(delta, delta)
    b_poly = np.polynomial.polynomial.polymul(delta, delta_prime)
    c_poly = np.polynomial.polynomial.polyadd(
        p_poly, np.polynomial.polynomial.polymul(q_poly, delta)
    )

    def coefficient(values: np.ndarray, index: int) -> complex:
        total = 0.0j
        for k, value in enumerate(a_poly):
            j = index - k + 2
            if 0 <= j < len(values):
                total += value * (exponent + j) * (exponent + j - 1.0) * values[j]
        for k, value in enumerate(b_poly):
            j = index - k + 1
            if 0 <= j < len(values):
                total += value * (exponent + j) * values[j]
        for k, value in enumerate(c_poly):
            j = index - k
            if 0 <= j < len(values):
                total += value * values[j]
        return complex(total)

    result = np.zeros(order + 1, dtype=complex)
    result[0] = 1.0
    for index in range(1, order + 1):
        result[index] = 0.0
        base = coefficient(result, index)
        result[index] = 1.0
        slope = coefficient(result, index) - base
        if abs(slope) < 1.0e-300:
            raise RuntimeError("singular horizon Frobenius recurrence")
        result[index] = -base / slope
    return exponent, result


def _series_inverse(values: np.ndarray, size: int) -> np.ndarray:
    result = np.zeros(size, dtype=complex)
    result[0] = 1.0 / values[0]
    for index in range(1, size):
        result[index] = -sum(
            values[k] * result[index - k]
            for k in range(1, min(index + 1, len(values)))
        ) / values[0]
    return result


def _outer_log_derivative(
    alpha: float, chi: float, state: State, omega: complex,
    angular: complex, radius: float, order: int,
) -> complex:
    size = order + 4
    d_poly = np.zeros(size, dtype=complex)
    d_poly[:3] = (1.0, -2.0, chi * chi)
    inverse_d = _series_inverse(d_poly, size)
    inverse_d2 = np.convolve(inverse_d, inverse_d)[:size]
    numerator_a = np.zeros(size, dtype=complex)
    numerator_a[1:3] = (2.0, -2.0)
    a_series = np.convolve(numerator_a, inverse_d)[:size]
    p_scaled = np.zeros(size, dtype=complex)
    p_scaled[0] = omega**2
    p_scaled[2] += 2.0 * omega**2 * chi**2
    p_scaled[3] += -4.0 * chi * state.m * omega
    p_scaled[4] += omega**2 * chi**4 + state.m**2 * chi**2
    q_scaled = np.zeros(size, dtype=complex)
    q_scaled[0] = -alpha**2
    q_scaled[2] = -omega**2 * chi**2 - angular
    b_series = (
        np.convolve(p_scaled, inverse_d2)[:size]
        + np.convolve(q_scaled, inverse_d)[:size]
    )
    decay = cmath.sqrt(alpha * alpha - omega * omega)
    if decay.real < 0.0:
        decay = -decay
    kappa = -decay
    coulomb = (alpha * alpha - 2.0 * omega * omega) / kappa - 1.0
    log_series = np.zeros(size, dtype=complex)
    log_series[0] = kappa
    log_series[1] = coulomb

    def residual(values: np.ndarray) -> np.ndarray:
        derivative = np.zeros(size, dtype=complex)
        for j in range(1, size - 1):
            derivative[j + 1] = -j * values[j]
        return (
            derivative + np.convolve(values, values)[:size]
            + np.convolve(a_series, values)[:size] + b_series
        )

    for index in range(2, order + 2):
        log_series[index] = 0.0
        base = residual(log_series)[index]
        slope = 2.0 * kappa
        if abs(slope) < 1.0e-300:
            raise RuntimeError("singular outer logarithmic recurrence")
        log_series[index] = -base / slope
    inverse_radius = 1.0 / radius
    return complex(np.polynomial.polynomial.polyval(inverse_radius, log_series))


def _frobenius_initial(exponent: complex, coefficients: np.ndarray, offset: float):
    powers = np.arange(len(coefficients), dtype=float)
    polynomial = np.sum(coefficients * offset**powers)
    derivative_poly = np.sum(
        coefficients[1:] * powers[1:] * offset ** (powers[1:] - 1.0)
    )
    value = offset**exponent * polynomial
    derivative = offset**exponent * (
        derivative_poly + exponent / offset * polynomial
    )
    return np.asarray([value, derivative], dtype=complex)


def _independent_radial_shooting(
    alpha: float, chi: float, state: State, omega: complex, angular: complex,
    settings: ContourKernelSettings,
):
    r_plus = 1.0 + math.sqrt(1.0 - chi * chi)
    r_minus = 2.0 - r_plus
    gap = r_plus - r_minus
    decay_complex = cmath.sqrt(alpha * alpha - omega * omega)
    if decay_complex.real < 0.0:
        decay_complex = -decay_complex
    kappa = -decay_complex
    coulomb = (alpha * alpha - 2.0 * omega * omega) / decay_complex - 1.0
    equation = _radial_equation(alpha, chi, state, omega, angular)
    exponent, frobenius = _horizon_frobenius_coefficients(
        alpha, chi, state, omega, angular, settings.frobenius_order
    )
    offset = min(settings.horizon_offsets)
    inner = r_plus + offset
    initial = _frobenius_initial(exponent, frobenius, offset)
    decay = max(decay_complex.real, 1.0e-5)
    outer = r_plus + 120.0 / decay
    del kappa, coulomb
    outer_log_derivative = _outer_log_derivative(
        alpha, chi, state, omega, angular, outer, settings.asymptotic_order
    )
    asymptotic = np.asarray([1.0 + 0.0j, outer_log_derivative], dtype=complex)
    matches = tuple(
        r_plus + length / decay for length in settings.match_decay_lengths
    )
    outward = solve_ivp(
        equation, (inner, max(matches)), initial, method="DOP853",
        rtol=min(settings.rtol, 1.0e-12),
        atol=min(settings.atol, 1.0e-14), dense_output=True,
    )
    inward = solve_ivp(
        equation, (outer, min(matches)), asymptotic, method="DOP853",
        rtol=min(settings.rtol, 1.0e-12),
        atol=min(settings.atol, 1.0e-14), dense_output=True,
    )
    if not outward.success or not inward.success:
        raise RuntimeError("independent radial shooting failed")
    residuals = []
    scales = []
    for match_value in matches:
        local_left = outward.sol(match_value)
        local_right = inward.sol(match_value)
        local_scale = local_left[0] / local_right[0]
        scales.append(local_scale)
        local_right = local_scale * local_right
        residuals.append(
            abs(local_left[0] * local_right[1]
                - local_left[1] * local_right[0])
            / (abs(local_left[0] * local_right[1])
               + abs(local_left[1] * local_right[0]) + 1.0e-300)
        )
    # The exponentially growing complementary solution makes late matches
    # ill-conditioned even at a certified frequency.  Evaluate all requested
    # match radii and use the one with the smallest Wronskian mismatch.
    best = int(np.argmin(residuals))
    match = matches[best]
    scale = scales[best]
    wronskian = residuals[best]

    def evaluate(radius):
        array = np.asarray(radius, dtype=float)
        flat = array.ravel()
        values = np.empty_like(flat, dtype=complex)
        first = np.empty_like(flat, dtype=complex)
        low = flat <= match
        if np.any(low):
            result = outward.sol(flat[low])
            values[low], first[low] = result
        if np.any(~low):
            result = scale * inward.sol(flat[~low])
            values[~low], first[~low] = result
        shape = array.shape
        if array.ndim == 0:
            return complex(values[0]), complex(first[0])
        return values.reshape(shape), first.reshape(shape)
    return evaluate, float(wronskian), r_plus, outer


def _independent_finite_part_norm(
    radial,
    alpha: float,
    chi: float,
    state: State,
    omega: complex,
    angular_gamma: complex,
    r_plus: float,
    outer: float,
    offsets: Sequence[float],
) -> tuple[complex, float, tuple[complex, ...]]:
    def integrand(radius):
        radial_value = radial(radius)[0]
        delta = radius * radius - 2.0 * radius + chi * chi
        kernel = (
            2.0 * omega
            * ((radius * radius + chi * chi) ** 2 - angular_gamma * chi * chi * delta)
            - 4.0 * radius * chi * state.m
        )
        return kernel / delta * radial_value * radial_value

    integrals: list[complex] = []
    for cutoff in offsets:
        lower = math.log(cutoff)
        upper = math.log(outer - r_plus)
        edges = np.linspace(lower, upper, 7)
        pieces = []
        for left, right in zip(edges[:-1], edges[1:], strict=True):
            pieces.append(_gauss_integral(
                lambda log_x: integrand(r_plus + np.exp(log_x)) * np.exp(log_x),
                float(left), float(right), 80,
            ))
        integrals.append(_complex_fsum(pieces))
    gap = 2.0 * math.sqrt(1.0 - chi * chi)
    sigma = 2.0 * r_plus * (omega - state.m * chi / (2.0 * r_plus)) / gap
    exponent = -2j * sigma
    cutoffs = np.asarray(offsets, dtype=float)

    def fit(indices, powers):
        selected = cutoffs[np.asarray(indices)]
        continuation_columns = []
        for power in powers:
            local_exponent = exponent + power
            # At saturation the ingoing exponent tends to zero.  Fitting
            # c**exponent beside a constant is then rank deficient and gives
            # an erroneous factor of two in the finite part.  The analytic
            # continuation of c**p/p is log(c) at p=0, so use that basis
            # explicitly rather than relying on a nearly singular limit.
            if abs(local_exponent) < 1.0e-7:
                continuation_columns.append(np.log(selected))
            else:
                continuation_columns.append(
                    selected**local_exponent / local_exponent
                )
        matrix = np.column_stack([
            np.ones(len(selected), dtype=complex),
            *continuation_columns,
        ])
        values = np.asarray(integrals, dtype=complex)[np.asarray(indices)]
        return complex(np.linalg.lstsq(matrix, values, rcond=None)[0][0])

    count = len(cutoffs)
    estimates = (
        fit(range(count), (0, 1)),
        fit(range(count), (0, 1, 2)),
        fit(range(1, count), (0, 1)),
    )
    mean = sum(estimates) / len(estimates)
    spread = max(abs(value - mean) for value in estimates) / max(abs(mean), 1.0e-300)
    return 1j * mean, float(spread), estimates


def _independent_contour_norm(
    radial,
    alpha: float,
    chi: float,
    state: State,
    omega: complex,
    angular: complex,
    angular_gamma: complex,
    r_plus: float,
    outer: float,
    settings: ContourKernelSettings,
) -> tuple[complex, float, tuple[complex, ...]]:
    """Evaluate the QBS bilinear norm on three deformed Hankel contours.

    The contour begins on a complex ray from the outer horizon and returns to
    the real axis inside the Frobenius disk.  Its ODE is parameterized by a
    real variable; the missing ray to the branch point is restored by the
    analytic primitive of the independently generated Frobenius series.  No
    local Hadamard counterterm from the production solver is used.
    """
    exponent, frobenius = _horizon_frobenius_coefficients(
        alpha, chi, state, omega, angular, settings.frobenius_order
    )
    gap = 2.0 * math.sqrt(1.0 - chi * chi)
    equation = _radial_equation(alpha, chi, state, omega, angular)
    offset = min(settings.horizon_offsets)
    join_offset = min(0.10 * gap, 0.02)
    join_offset = max(join_offset, 50.0 * offset)
    join = r_plus + join_offset

    # Smooth coefficient H(x) in integrand=x**(2 exponent-1) H(x).
    radius_poly = np.asarray([r_plus, 1.0], dtype=complex)
    radius_squared = np.polynomial.polynomial.polymul(radius_poly, radius_poly)
    r2a2 = radius_squared.copy(); r2a2[0] += chi * chi
    numerator = 2.0 * omega * np.polynomial.polynomial.polymul(r2a2, r2a2)
    delta_poly = np.asarray([0.0, gap, 1.0], dtype=complex)
    numerator = np.polynomial.polynomial.polyadd(
        numerator, -2.0 * omega * angular_gamma * chi * chi * delta_poly
    )
    numerator = np.polynomial.polynomial.polyadd(
        numerator, -4.0 * chi * state.m * radius_poly
    )
    series_size = 2 * settings.frobenius_order + 5
    inverse_gap = _series_inverse(
        np.asarray([gap, 1.0], dtype=complex), series_size
    )
    smooth = np.convolve(
        np.convolve(numerator, inverse_gap)[:series_size],
        np.convolve(frobenius, frobenius)[:series_size],
    )[:series_size]
    power = 2.0 * exponent

    def primitive(z_value: complex) -> complex:
        total = 0.0j
        for index, coefficient in enumerate(smooth):
            local_power = power + index
            if abs(local_power) < 1.0e-9:
                total += coefficient * cmath.log(z_value)
            else:
                total += coefficient * z_value**local_power / local_power
        return complex(total)

    def norm_integrand(radius_value: complex, radial_value: complex) -> complex:
        delta = radius_value * radius_value - 2.0 * radius_value + chi * chi
        kernel = (
            2.0 * omega
            * ((radius_value * radius_value + chi * chi) ** 2
               - angular_gamma * chi * chi * delta)
            - 4.0 * radius_value * chi * state.m
        )
        return kernel / delta * radial_value * radial_value

    real_edges = np.linspace(math.log(join_offset), math.log(outer - r_plus), 9)
    real_tail = _complex_fsum(
        _gauss_integral(
            lambda log_x: (
                norm_integrand(
                    r_plus + np.exp(log_x),
                    radial(r_plus + np.exp(log_x))[0],
                ) * np.exp(log_x)
            ),
            float(left), float(right), 96,
        )
        for left, right in zip(real_edges[:-1], real_edges[1:], strict=True)
    )

    estimates: list[complex] = []
    for angle in settings.contour_angles:
        z0 = offset * cmath.exp(1j * angle)
        initial = _frobenius_initial(exponent, frobenius, z0)
        direction = join_offset - z0

        def contour_equation(parameter: float, vector: np.ndarray) -> np.ndarray:
            radius_value = r_plus + z0 + direction * float(parameter)
            local = equation(radius_value, vector[:2])
            return np.asarray([
                direction * local[0],
                direction * local[1],
                direction * norm_integrand(radius_value, vector[0]),
            ], dtype=complex)

        solution = solve_ivp(
            contour_equation, (0.0, 1.0),
            np.asarray([initial[0], initial[1], 0.0j], dtype=complex),
            method="DOP853", rtol=min(settings.rtol, 1.0e-12),
            atol=min(settings.atol, 1.0e-14),
        )
        if not solution.success:
            raise RuntimeError("complex contour ODE failed")
        contour_endpoint = solution.y[:2, -1]
        real_endpoint = np.asarray(radial(join), dtype=complex)
        scale = complex(
            np.vdot(contour_endpoint, real_endpoint)
            / np.vdot(contour_endpoint, contour_endpoint)
        )
        mismatch = np.linalg.norm(scale * contour_endpoint - real_endpoint) / max(
            np.linalg.norm(real_endpoint), 1.0e-300
        )
        if mismatch > 1.0e-8:
            raise RuntimeError(
                f"complex contour matching residual {mismatch:.3e} exceeds tolerance"
            )
        finite_part = scale * scale * (
            complex(solution.y[2, -1]) + primitive(z0)
        ) + real_tail
        estimates.append(1j * finite_part)
    mean = sum(estimates) / len(estimates)
    spread = max(abs(value - mean) for value in estimates) / max(
        abs(mean), 1.0e-300
    )
    return complex(mean), float(spread), tuple(estimates)


def _independent_field_provider(
    mode: ContourModeResult,
):
    key = (mode.alpha, mode.chi, mode.state, mode.frequency_M)
    provider = _INDEPENDENT_MODE_PROVIDERS.get(key)
    if provider is None:
        raise KeyError("independent mode provider is unavailable")
    return provider


def solve_kerr_mode_contour(
    alpha: float,
    chi: float,
    state: State,
    certified_frequency: complex,
    settings: ContourKernelSettings = ContourKernelSettings(),
) -> ContourModeResult:
    try:
        angular, angular_solution, angular_residual = _angular_bvp(
            alpha, chi, state, certified_frequency, settings
        )
        radial, wronskian, r_plus, outer = _independent_radial_shooting(
            alpha, chi, state, certified_frequency, angular, settings
        )
        theta_grid = np.linspace(
            settings.angular_epsilon,
            math.pi - settings.angular_epsilon,
            max(800, settings.angular_nodes * 4),
        )
        angular_grid = angular_solution.sol(theta_grid)[0]
        angular_norm = 2.0 * math.pi * np.trapz(
            np.sin(theta_grid) * angular_grid * angular_grid, theta_grid
        )
        angular_scale = 1.0 / cmath.sqrt(complex(angular_norm))
        angular_gamma = 2.0 * math.pi * np.trapz(
            np.sin(theta_grid) ** 3
            * (angular_scale * angular_grid) ** 2,
            theta_grid,
        )
        raw_norm, spread, _ = _independent_contour_norm(
            radial, alpha, chi, state, certified_frequency, angular,
            complex(angular_gamma), r_plus, outer, settings,
        )
        radial_scale = 1.0 / cmath.sqrt(raw_norm)
        radius = np.geomspace(r_plus + 1.0e-3, outer, 600)
        values = radial_scale * radial(radius)[0]
        density = np.abs(values) ** 2 * radius**2
        r_peak = float(radius[int(np.argmax(density))])

        def angular_part(theta_value: float):
            local = angular_solution.sol(float(theta_value))
            value = angular_scale * complex(local[0])
            first = angular_scale * complex(local[1])
            sine = math.sin(theta_value)
            second = (
                -math.cos(theta_value) / sine * first
                + (state.m**2 / sine**2
                   - chi**2 * (certified_frequency**2 - alpha**2)
                   * math.cos(theta_value)**2 - angular) * value
            )
            return value, first, complex(second)

        radial_equation = _radial_equation(
            alpha, chi, state, certified_frequency, angular
        )

        def fields(radius_value: float, theta_value: float):
            raw_value, raw_first = radial(float(radius_value))
            radial_value = radial_scale * raw_value
            radial_first = radial_scale * raw_first
            radial_second = radial_scale * radial_equation(
                float(radius_value), np.asarray([raw_value, raw_first])
            )[1]
            angular_value, angular_first, angular_second = angular_part(theta_value)
            value = radial_value * angular_value
            derivatives = np.asarray([
                -1j * certified_frequency * value,
                radial_first * angular_value,
                radial_value * angular_first,
                1j * state.m * value,
                radial_second * angular_value,
                radial_value * angular_second,
            ], dtype=complex)
            return complex(value), derivatives

        _INDEPENDENT_MODE_PROVIDERS[
            (alpha, chi, state, certified_frequency)
        ] = fields
        normalization = radial_scale * angular_scale
        converged = bool(
            wronskian < 1.0e-8 and spread < 2.0e-3
            and angular_residual < 1.0e-8
        )
        status = "ok" if converged else (
            "shooting_mismatch" if wronskian >= 1.0e-8 else "contour_not_converged"
        )
        return ContourModeResult(
            alpha, chi, state, certified_frequency, angular, angular_residual,
            wronskian, spread, normalization, r_peak, converged, status,
            "fixed-frequency angular BVP, two-sided shooting, and complex Hankel-contour norm",
        )
    except (RuntimeError, ValueError, FloatingPointError) as error:
        return ContourModeResult(
            alpha, chi, state, certified_frequency, complex(math.nan, math.nan),
            math.inf, math.inf, math.inf, complex(math.nan, math.nan), math.nan,
            False, "contour_not_converged", str(error),
        )


def independent_kerr_kernel_M(
    config: TransitionConfig,
    contour_modes: Sequence[ContourModeResult],
    settings: ContourKernelSettings = ContourKernelSettings(),
) -> IndependentKernelResult:
    if len(contour_modes) != 2:
        raise ValueError("contour_modes must contain initial and final modes")
    initial, final = contour_modes
    if initial.state != config.initial or final.state != config.final:
        raise ValueError("contour modes do not match transition states")
    if not all(mode.converged for mode in contour_modes):
        return IndependentKernelResult(
            _point_id(config), None, 0.0j, None, None,
            max(mode.contour_norm_spread for mode in contour_modes),
            max(mode.shooting_wronskian_residual for mode in contour_modes),
            math.inf, "contour_not_converged", "one or both contour modes failed",
        )

    def surrogate(mode: ContourModeResult) -> KerrModeResult:
        return KerrModeResult(
            alpha=mode.alpha, chi=mode.chi, state=mode.state,
            frequency_M=mode.frequency_M,
            separation_constant=mode.separation_constant,
            angular_l_values=np.asarray([mode.state.l], dtype=int),
            angular_coefficients=np.asarray([1.0 + 0.0j]),
            radial_coefficients=np.asarray([1.0 + 0.0j]),
            bilinear_norm=1.0 + 0.0j, r_99_M=mode.r_peak_M,
            cf_residual=0.0, radial_residual=mode.shooting_wronskian_residual,
            angular_residual=mode.angular_residual, converged=True,
            message="independent contour surrogate",
        )

    delta_m = initial.state.m - final.state.m
    if delta_m == 0:
        raise ValueError("transition has zero Delta m")
    omega = float((initial.frequency_M.real - final.frequency_M.real) / delta_m)
    independent = _bl_tidal_kernel(
        surrogate(initial), surrogate(final), omega, config.settings, "irg",
        radial_strategy="quad",
        initial_field_provider=_independent_field_provider(initial),
        final_field_provider=_independent_field_provider(final),
    )
    return IndependentKernelResult(
        _point_id(config), None, independent, None, None,
        max(mode.contour_norm_spread for mode in contour_modes),
        max(mode.shooting_wronskian_residual for mode in contour_modes),
        0.0, "ok", "standalone independent projection; no certified comparison attached",
    )


def validate_kerr_kernel(
    config: TransitionConfig,
    certified_modes: Sequence[KerrModeResult] | Mapping[object, KerrModeResult],
    settings: ContourKernelSettings = ContourKernelSettings(),
) -> IndependentKernelResult:
    pair = _resolved_pair(certified_modes, config.initial, config.final)
    central = compute_transition_kernel(config, pair)
    if central.kernel_status != "ok":
        return IndependentKernelResult(
            _point_id(config), 0.0j, 0.0j, math.inf, math.inf, math.inf,
            math.inf, math.inf, "kernel_crosscheck_failed", central.message,
        )
    contour = tuple(
        solve_kerr_mode_contour(
            mode.alpha, mode.chi, mode.state, mode.frequency_M, settings
        ) for mode in pair
    )
    if not all(mode.converged for mode in contour):
        return IndependentKernelResult(
            _point_id(config), complex(central.covariant_kernel_M), 0.0j,
            math.inf, math.inf,
            max(mode.contour_norm_spread for mode in contour),
            max(mode.shooting_wronskian_residual for mode in contour),
            math.inf, "contour_not_converged",
            "; ".join(mode.message for mode in contour if not mode.converged),
        )
    eigen_difference = max(
        abs(contour_mode.separation_constant - certified.separation_constant)
        / max(abs(certified.separation_constant), 1.0)
        for contour_mode, certified in zip(contour, pair, strict=True)
    )
    # Reproject the production operator with independently generated and
    # independently normalized field providers.
    omega = float(central.omega_res_M)
    independent = _bl_tidal_kernel(
        pair[0], pair[1], omega, config.settings, "irg",
        radial_strategy="quad",
        initial_field_provider=_independent_field_provider(contour[0]),
        final_field_provider=_independent_field_provider(contour[1]),
    )
    certified = complex(central.covariant_kernel_M)
    amplitude = abs(abs(independent) - abs(certified)) / max(abs(certified), 1.0e-300)
    phase = abs(cmath.phase(independent / certified)) if certified else math.inf
    contour_spread = max(mode.contour_norm_spread for mode in contour)
    wronskian = max(mode.shooting_wronskian_residual for mode in contour)
    passed = bool(
        amplitude < 5.0e-3 and phase < 5.0e-3 and contour_spread < 2.0e-3
        and wronskian < 1.0e-8 and eigen_difference < 1.0e-8
    )
    return IndependentKernelResult(
        _point_id(config), certified, independent, amplitude, phase,
        contour_spread, wronskian, eigen_difference,
        "ok" if passed else "kernel_crosscheck_failed",
        "fixed CF frequency; independent BVP/shooting/complex-contour normalization and quad reprojection",
    )


KernelCrosscheckResult = IndependentKernelResult
