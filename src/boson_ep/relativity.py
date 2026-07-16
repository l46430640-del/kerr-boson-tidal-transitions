"""Kerr massive-scalar quasibound frequencies from Dolan's continued fraction."""

from __future__ import annotations

import cmath
import math
from dataclasses import replace
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, cumulative_trapezoid, quad, solve_ivp
from scipy.linalg import eig
from scipy.optimize import brentq, least_squares, root
from scipy.special import sph_harm_y

from .models import (
    CFResult,
    CFSettings,
    KerrModeResult,
    KerrModeSettings,
    SaturationResult,
    State,
)
from .spectrum import gamma_detweiler_M, horizon_omega_M, omega_real_M


def _cosine_coefficient(l_value: int, m_value: int) -> float:
    if l_value <= 0 or abs(m_value) > l_value:
        return 0.0
    return math.sqrt(
        (l_value * l_value - m_value * m_value)
        / ((2 * l_value - 1) * (2 * l_value + 1))
    )


def _spheroidal_matrix(
    c_squared: complex,
    state: State,
    lmax: int,
) -> tuple[np.ndarray, np.ndarray]:
    if lmax < state.l + 4:
        raise ValueError("lmax must include at least two same-parity couplings")
    start = abs(state.m)
    if (start - state.l) % 2:
        start += 1
    l_values = np.arange(start, lmax + 1, 2, dtype=int)
    matrix = np.zeros((len(l_values), len(l_values)), dtype=complex)
    for column, l_value in enumerate(l_values):
        a_l = _cosine_coefficient(int(l_value), state.m)
        a_lp1 = _cosine_coefficient(int(l_value) + 1, state.m)
        matrix[column, column] = (
            l_value * (l_value + 1)
            - c_squared * (a_l * a_l + a_lp1 * a_lp1)
        )
        if column + 1 < len(l_values):
            coupling = a_lp1 * _cosine_coefficient(int(l_value) + 2, state.m)
            matrix[column + 1, column] = -c_squared * coupling
            matrix[column, column + 1] = -c_squared * coupling
    return l_values, matrix


def spheroidal_eigenvalue(
    c_squared: complex,
    state: State,
    lmax: int = 12,
) -> complex:
    """Return the overlap-tracked ``l`` branch from the spectral matrix."""
    value, _, _ = spheroidal_eigensystem(
        c_squared,
        state,
        lmax,
        reference_l_values=np.asarray([state.l], dtype=int),
        reference_coefficients=np.asarray([1.0 + 0.0j]),
    )
    return value


def spheroidal_eigensystem(
    c_squared: complex,
    state: State,
    lmax: int = 14,
    reference_l_values: np.ndarray | None = None,
    reference_coefficients: np.ndarray | None = None,
) -> tuple[complex, np.ndarray, np.ndarray]:
    """Return the continuously labelled scalar spheroidal angular branch.

    The angular matrix is complex symmetric, so the coefficients are
    normalized with the bilinear convention ``c.T @ c = 1`` rather than the
    Hermitian convention.  This is the angular part of the QBS bilinear
    normalization used by Cannizzaro et al.
    """
    l_values, matrix = _spheroidal_matrix(c_squared, state, lmax)
    values, vectors = eig(matrix)
    if reference_coefficients is None or reference_l_values is None:
        target_index = int(np.argmin(np.abs(values - state.l * (state.l + 1))))
    else:
        reference = np.zeros(len(l_values), dtype=complex)
        for l_value, coefficient in zip(
            reference_l_values, reference_coefficients, strict=True
        ):
            matches = np.flatnonzero(l_values == int(l_value))
            if matches.size:
                reference[int(matches[0])] = coefficient
        reference_norm = np.linalg.norm(reference)
        if reference_norm == 0.0:
            raise ValueError("reference angular branch has no overlapping basis")
        overlaps = np.abs(reference @ vectors) / (
            reference_norm * np.linalg.norm(vectors, axis=0)
        )
        target_index = int(np.argmax(overlaps))
    coefficients = vectors[:, target_index]
    bilinear = np.dot(coefficients, coefficients)
    coefficients = coefficients / cmath.sqrt(bilinear)
    target_position = int(np.where(l_values == state.l)[0][0])
    if coefficients[target_position].real < 0.0:
        coefficients = -coefficients
    return complex(values[target_index]), l_values, coefficients


def _bound_kappa(alpha: float, omega: complex) -> complex:
    value = cmath.sqrt(alpha * alpha - omega * omega)
    if value.real < 0.0:
        value = -value
    return -value


def _leaver_recurrence(
    alpha: float,
    chi: float,
    state: State,
    omega: complex,
    angular: complex,
):
    """Return Dolan's three-term recurrence and radial exponents."""
    kappa = _bound_kappa(alpha, omega)
    b_value = math.sqrt(1.0 - chi * chi)
    shifted = omega - 1j * kappa
    horizon_shift = omega - chi * state.m / 2.0
    c0 = 1.0 - 2j * omega - 2j * horizon_shift / b_value
    c1 = (
        -4.0
        + 4j * (omega - 1j * kappa * (1.0 + b_value))
        + 4j * horizon_shift / b_value
        - 2.0 * (omega * omega + kappa * kappa) / kappa
    )
    c2 = (
        3.0
        - 2j * omega
        - 2.0 * (kappa * kappa - omega * omega) / kappa
        - 2j * horizon_shift / b_value
    )
    c3 = (
        2j * shifted**3 / kappa
        + 2.0 * shifted**2 * b_value
        + kappa * kappa * chi * chi
        + 2j * kappa * chi * state.m
        - angular
        - 1.0
        - shifted**2 / kappa
        + 2.0 * kappa * b_value
        + 2j / b_value * (shifted**2 / kappa + 1.0) * horizon_shift
    )
    c4 = (
        shifted**4 / (kappa * kappa)
        + 2j * omega * shifted**2 / kappa
        - 2j / b_value * shifted**2 / kappa * horizon_shift
    )

    def alpha_n(index: int) -> complex:
        return index * index + (c0 + 1.0) * index + c0

    def beta_n(index: int) -> complex:
        return -2.0 * index * index + (c1 + 2.0) * index + c3

    def gamma_n(index: int) -> complex:
        return index * index + (c2 - 3.0) * index + c4

    return alpha_n, beta_n, gamma_n, kappa


def radial_continued_fraction_residual(
    alpha: float,
    chi: float,
    state: State,
    omega: complex,
    settings: CFSettings = CFSettings(),
) -> complex:
    """Evaluate Dolan Eqs. (35)-(48) with a finite backward fraction."""
    if not 0.0 < alpha <= 0.5:
        raise ValueError("continued-fraction calibration supports 0 < alpha <= 0.5")
    if not 0.0 <= chi < 1.0:
        raise ValueError("continued-fraction calibration requires 0 <= chi < 1")
    if settings.truncation < 20:
        raise ValueError("continued-fraction truncation must be at least 20")
    kappa = _bound_kappa(alpha, omega)
    if abs(kappa) == 0.0:
        return complex(np.inf, np.inf)
    angular = spheroidal_eigenvalue(
        chi * chi * (omega * omega - alpha * alpha),
        state,
        settings.angular_lmax,
    )
    alpha_n, beta_n, gamma_n, _ = _leaver_recurrence(
        alpha, chi, state, omega, angular
    )

    tiny = 1.0e-300
    fraction = beta_n(0)
    if abs(fraction) < tiny:
        fraction = complex(tiny, tiny)
    c_value = fraction
    d_value = 0.0 + 0.0j
    for index in range(1, settings.truncation + 1):
        numerator = -alpha_n(index - 1) * gamma_n(index)
        denominator = beta_n(index)
        d_value = denominator + numerator * d_value
        if abs(d_value) < tiny:
            d_value = complex(tiny, tiny)
        c_value = denominator + numerator / c_value
        if abs(c_value) < tiny:
            c_value = complex(tiny, tiny)
        d_value = 1.0 / d_value
        update = c_value * d_value
        fraction *= update
    return fraction


def radial_continued_fraction_residual_backward(
    alpha: float,
    chi: float,
    state: State,
    omega: complex,
    settings: CFSettings = CFSettings(),
) -> complex:
    """Evaluate the same Leaver fraction by backward finite recursion.

    This evaluator has numerical failure modes independent of modified Lentz.
    Agreement of the two is therefore part of accepting a CF root.
    """
    if not 0.0 < alpha <= 0.5:
        raise ValueError("continued-fraction calibration supports 0 < alpha <= 0.5")
    if not 0.0 <= chi < 1.0:
        raise ValueError("continued-fraction calibration requires 0 <= chi < 1")
    if settings.truncation < 20:
        raise ValueError("continued-fraction truncation must be at least 20")
    kappa = _bound_kappa(alpha, omega)
    if abs(kappa) == 0.0:
        return complex(np.inf, np.inf)
    angular = spheroidal_eigenvalue(
        chi * chi * (omega * omega - alpha * alpha),
        state,
        settings.angular_lmax,
    )
    alpha_n, beta_n, gamma_n, _ = _leaver_recurrence(
        alpha, chi, state, omega, angular
    )
    tiny = 1.0e-300
    denominator = beta_n(settings.truncation)
    if abs(denominator) < tiny:
        denominator = complex(tiny, tiny)
    for index in range(settings.truncation - 1, 0, -1):
        denominator = (
            beta_n(index)
            - alpha_n(index) * gamma_n(index + 1) / denominator
        )
        if abs(denominator) < tiny:
            denominator = complex(tiny, tiny)
    return beta_n(0) - alpha_n(0) * gamma_n(1) / denominator


def _cf_root_quality(
    alpha: float,
    chi: float,
    state: State,
    omega: complex,
    settings: CFSettings,
) -> tuple[float, float]:
    lentz = radial_continued_fraction_residual(
        alpha, chi, state, omega, settings
    )
    backward = radial_continued_fraction_residual_backward(
        alpha, chi, state, omega, settings
    )
    scale = max(1.0, abs(omega))
    residual = max(abs(lentz), abs(backward)) / scale
    evaluator_difference = abs(lentz - backward) / scale
    return float(residual), float(evaluator_difference)


def solve_quasibound_cf(
    alpha: float,
    chi: float,
    state: State,
    settings: CFSettings = CFSettings(),
    initial_guess: complex | None = None,
) -> CFResult:
    if initial_guess is None:
        if alpha <= 0.3 and chi > 0.0:
            initial_guess = complex(
                omega_real_M(alpha, chi, state),
                gamma_detweiler_M(alpha, chi, state),
            )
        else:
            bracket = 1.0 - alpha**2 / (2.0 * state.n**2)
            initial_guess = complex(alpha * bracket, 0.0)
    omega_scale = max(alpha**5, 1.0e-8)

    def unpack(values: np.ndarray) -> complex:
        return complex(
            initial_guess.real + values[0] * omega_scale,
            values[1] * omega_scale,
        )

    def equations(values: np.ndarray) -> np.ndarray:
        residual = radial_continued_fraction_residual_backward(
            alpha, chi, state, unpack(values), settings
        )
        return np.asarray([residual.real, residual.imag], dtype=float)

    start = np.asarray([0.0, initial_guess.imag / omega_scale], dtype=float)
    candidates: list[tuple[complex, float, float, str]] = []

    def record(values: np.ndarray, message: str) -> bool:
        frequency = unpack(np.asarray(values, dtype=float))
        if not (np.isfinite(frequency.real) and np.isfinite(frequency.imag)):
            return False
        try:
            residual, evaluator_difference = _cf_root_quality(
                alpha, chi, state, frequency, settings
            )
        except (ArithmeticError, ValueError, ZeroDivisionError):
            return False
        candidates.append((frequency, residual, evaluator_difference, message))
        return bool(
            0.0 < frequency.real < alpha
            and residual <= settings.residual_tolerance
            and evaluator_difference <= 1.0e-10
        )

    record(start, "initial guess")

    # A short explicit Newton stage is essential for weak high-l clouds.  In
    # scaled coordinates their root can lie only ~2e-5 from an excellent
    # multiplet seed, while generic MINPACK step selection either sees a zero
    # Jacobian or steps across the very narrow CF pole.  The centered
    # derivative below corresponds to an omega increment of alpha^5*1e-5 and
    # resolves that local root without ever scaling by the physical Gamma.
    newton = start.copy()
    best_newton = newton.copy()
    best_norm = float(np.linalg.norm(equations(newton)))
    for _ in range(12):
        value = equations(newton)
        value_norm = float(np.linalg.norm(value))
        if not np.isfinite(value_norm):
            break
        if value_norm < best_norm:
            best_norm = value_norm
            best_newton = newton.copy()
        if value_norm <= settings.residual_tolerance * 0.25:
            break
        step = 1.0e-5 * max(1.0, float(np.max(np.abs(newton))))
        columns = []
        for index in range(2):
            offset = np.zeros(2, dtype=float)
            offset[index] = step
            columns.append(
                (equations(newton + offset) - equations(newton - offset))
                / (2.0 * step)
            )
        jacobian = np.column_stack(columns)
        if not np.all(np.isfinite(jacobian)):
            break
        try:
            correction = np.linalg.solve(jacobian, -value)
        except np.linalg.LinAlgError:
            correction = np.linalg.lstsq(jacobian, -value, rcond=None)[0]
        correction_norm = float(np.linalg.norm(correction))
        if not np.isfinite(correction_norm):
            break
        if correction_norm > 10.0:
            correction *= 10.0 / correction_norm
        improved = False
        for damping in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = newton + damping * correction
            trial_norm = float(np.linalg.norm(equations(trial)))
            if np.isfinite(trial_norm) and trial_norm < value_norm:
                newton = trial
                improved = True
                break
        if not improved:
            break
    record(best_newton, "scaled centered-Newton")
    start = best_newton
    hybr = root(
        equations,
        start,
        method="hybr",
        # The coordinates are already scaled by alpha^5.  SciPy's default
        # finite-difference increment would perturb omega by only ~1e-13 at
        # alpha=0.1 and numerically zero the high-l Jacobian.  An explicit
        # scaled-coordinate step keeps the physical perturbation near 1e-8.
        options={
            "maxfev": settings.max_function_evaluations,
            "xtol": 1.0e-11,
            "eps": 1.0e-6,
        },
    )
    accepted = record(hybr.x, f"hybr: {hybr.message}")
    best_values = hybr.x
    if candidates:
        best_frequency = min(candidates, key=lambda value: value[1])[0]
        best_values = np.asarray(
            [
                (best_frequency.real - initial_guess.real) / omega_scale,
                best_frequency.imag / omega_scale,
            ]
        )
    if not accepted:
        lm = root(
            equations,
            best_values,
            method="lm",
            options={
                "maxiter": settings.max_function_evaluations,
                "ftol": 1.0e-13,
                "eps": 1.0e-6,
            },
        )
        accepted = record(lm.x, f"lm: {lm.message}")
        best_values = lm.x
    if not accepted:
        lower = np.asarray(
            [(np.finfo(float).eps - initial_guess.real) / omega_scale, -alpha / omega_scale]
        )
        upper = np.asarray(
            [(alpha - np.finfo(float).eps - initial_guess.real) / omega_scale, alpha / omega_scale]
        )
        clipped = np.minimum(np.maximum(best_values, lower + 1.0e-14), upper - 1.0e-14)
        least = least_squares(
            equations,
            clipped,
            bounds=(lower, upper),
            xtol=1.0e-13,
            ftol=1.0e-13,
            gtol=1.0e-13,
            diff_step=1.0e-3,
            max_nfev=settings.max_function_evaluations,
        )
        record(least.x, f"least_squares: {least.message}")
    physical_candidates = [
        candidate for candidate in candidates if 0.0 < candidate[0].real < alpha
    ]
    selected = min(
        physical_candidates or candidates,
        key=lambda value: (value[1], value[2]),
    )
    frequency, residual, evaluator_difference, solver_message = selected
    bound_state = 0.0 < frequency.real < alpha
    converged = bool(
        np.isfinite(residual)
        and residual <= settings.residual_tolerance
        and evaluator_difference <= 1.0e-10
        and bound_state
    )
    return CFResult(
        alpha=alpha,
        chi=chi,
        state=state,
        frequency_M=frequency,
        residual=float(residual),
        truncation=settings.truncation,
        angular_lmax=settings.angular_lmax,
        converged=converged,
        message=(
            f"{solver_message}; backward/Lentz difference="
            f"{evaluator_difference:.3e}; scale={omega_scale:.3e}"
        ),
    )


def quasibound_frequency_cf(
    alpha: float,
    chi: float,
    state: State,
    settings: CFSettings = CFSettings(),
) -> complex:
    result = solve_quasibound_cf(alpha, chi, state, settings)
    if not result.converged:
        raise RuntimeError(
            f"continued-fraction calibration failed for {state.label}: "
            f"residual={result.residual:.3e}; {result.message}"
        )
    return result.frequency_M


def leaver_radial_coefficients(
    alpha: float,
    chi: float,
    state: State,
    omega: complex,
    angular: complex,
    terms: int = 250,
) -> np.ndarray:
    """Construct the minimal radial solution by backward Miller ratios."""
    if terms < 1:
        raise ValueError("terms must be positive")
    alpha_n, beta_n, gamma_n, _ = _leaver_recurrence(
        alpha, chi, state, omega, angular
    )
    # Match the finite-fraction boundary used to obtain the root.  Extending
    # the tail beyond that boundary changes R_1 by more than the tiny QBS
    # residual at low alpha, even though both tails converge asymptotically.
    tail = terms
    ratios = np.empty(tail + 1, dtype=complex)
    tiny = 1.0e-300
    tail_denominator = beta_n(tail)
    if abs(tail_denominator) < tiny:
        tail_denominator = complex(tiny, tiny)
    ratios[tail] = -gamma_n(tail) / tail_denominator
    for index in range(tail - 1, 0, -1):
        denominator = beta_n(index) + alpha_n(index) * ratios[index + 1]
        if abs(denominator) < tiny:
            denominator = complex(tiny, tiny)
        ratios[index] = -gamma_n(index) / denominator
    coefficients = np.ones(terms, dtype=complex)
    for index in range(1, terms):
        coefficients[index] = coefficients[index - 1] * ratios[index]
    return coefficients


def angular_mode_values(
    theta: np.ndarray | float,
    state: State,
    angular: complex,
    l_values: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate S, dS/dtheta and d2S/dtheta2 at phi=0."""
    theta_array = np.asarray(theta, dtype=float)
    value = np.zeros_like(theta_array, dtype=complex)
    derivative = np.zeros_like(theta_array, dtype=complex)
    second = np.zeros_like(theta_array, dtype=complex)
    for l_value, coefficient in zip(l_values, coefficients, strict=True):
        l_int = int(l_value)
        harmonic, first_derivatives, second_derivatives = sph_harm_y(
            l_int, state.m, theta_array, 0.0, diff_n=2
        )
        value += coefficient * harmonic
        derivative += coefficient * first_derivatives[..., 0]
        second += coefficient * second_derivatives[..., 0, 0]
    return value, derivative, second


def angular_mode_second_derivative(
    theta: np.ndarray | float,
    state: State,
    alpha: float,
    chi: float,
    omega: complex,
    angular: complex,
    value: np.ndarray,
    derivative: np.ndarray,
) -> np.ndarray:
    theta_array = np.asarray(theta, dtype=float)
    sine = np.sin(theta_array)
    safe = np.where(np.abs(sine) > 1.0e-14, sine, 1.0e-14)
    c_squared = chi * chi * (omega * omega - alpha * alpha)
    return (
        -np.cos(theta_array) / safe * derivative
        + (state.m * state.m / safe**2 - c_squared * np.cos(theta_array) ** 2 - angular)
        * value
    )


def angular_mode_residual(
    theta: np.ndarray,
    state: State,
    alpha: float,
    chi: float,
    omega: complex,
    angular: complex,
    l_values: np.ndarray,
    coefficients: np.ndarray,
    lmax: int,
) -> float:
    """Independent spectral and collocation residual for an angular mode."""
    value, first, second = angular_mode_values(
        theta, state, angular, l_values, coefficients
    )
    sine = np.sin(theta)
    term_one = second
    term_two = np.cos(theta) / sine * first
    term_three = (
        chi * chi * (omega * omega - alpha * alpha) * np.cos(theta) ** 2
        - state.m * state.m / sine**2
        + angular
    ) * value
    collocation = float(
        np.max(
            np.abs(term_one + term_two + term_three)
            / (
                np.abs(term_one)
                + np.abs(term_two)
                + np.abs(term_three)
                + np.abs(value)
                + 1.0e-30
            )
        )
    )
    matrix_l_values, matrix = _spheroidal_matrix(
        chi * chi * (omega * omega - alpha * alpha), state, lmax
    )
    if not np.array_equal(matrix_l_values, l_values):
        raise ValueError("angular coefficients do not match the requested basis")
    denominator = (
        np.linalg.norm(matrix, ord=2) + abs(angular) + 1.0e-30
    ) * np.linalg.norm(coefficients)
    spectral = float(
        np.linalg.norm(matrix @ coefficients - angular * coefficients)
        / denominator
    )
    return max(collocation, spectral)


def radial_mode_values(
    radius: np.ndarray | float,
    alpha: float,
    chi: float,
    state: State,
    omega: complex,
    coefficients: np.ndarray,
    advanced: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the Leaver radial series and its first two derivatives."""
    radius_array = np.asarray(radius, dtype=float)
    b_value = math.sqrt(1.0 - chi * chi)
    r_plus = 1.0 + b_value
    r_minus = 1.0 - b_value
    gap = r_plus - r_minus
    x = radius_array - r_plus
    y = radius_array - r_minus
    if np.any(x <= 0.0):
        raise ValueError("radial series is defined outside the outer horizon")
    kappa = _bound_kappa(alpha, omega)
    sigma = 2.0 * r_plus * (omega - state.m * chi / (2.0 * r_plus)) / gap
    coulomb = (alpha * alpha - 2.0 * omega * omega) / kappa
    exponent_x = -1j * sigma
    exponent_y = 1j * sigma + coulomb - 1.0
    exponential = kappa
    if advanced:
        exponent_x = 0.0
        exponent_y += -2j * r_minus * omega / gap + 1j * state.m * chi / gap
        exponential += 1j * omega
    z_value = x / y
    z_first = gap / y**2
    z_second = -2.0 * gap / y**3
    powers = np.arange(len(coefficients), dtype=float)
    flat_z = np.ravel(z_value)
    polynomial = np.polynomial.polynomial.polyval(flat_z, coefficients).reshape(z_value.shape)
    first_coefficients = coefficients[1:] * powers[1:]
    second_coefficients = coefficients[2:] * powers[2:] * (powers[2:] - 1.0)
    polynomial_z = np.polynomial.polynomial.polyval(flat_z, first_coefficients).reshape(z_value.shape)
    polynomial_zz = np.polynomial.polynomial.polyval(flat_z, second_coefficients).reshape(z_value.shape)
    prefactor = x**exponent_x * y**exponent_y * np.exp(exponential * radius_array)
    log_first = exponent_x / x + exponent_y / y + exponential
    log_second = -exponent_x / x**2 - exponent_y / y**2
    prefactor_first = prefactor * log_first
    prefactor_second = prefactor * (log_first**2 + log_second)
    value = prefactor * polynomial
    first = prefactor_first * polynomial + prefactor * polynomial_z * z_first
    second = (
        prefactor_second * polynomial
        + 2.0 * prefactor_first * polynomial_z * z_first
        + prefactor
        * (polynomial_zz * z_first**2 + polynomial_z * z_second)
    )
    return value, first, second


def build_radial_ode_evaluator(
    alpha: float,
    chi: float,
    state: State,
    omega: complex,
    separation_constant: complex,
    coefficients: np.ndarray,
    *,
    outer_decay_lengths: float = 20.0,
    rtol: float = 1.0e-10,
):
    """Match the horizon series to an independently integrated radial ODE.

    The Leaver power series converges slowly when ``z -> 1``.  Evaluating a
    finite series throughout a low-alpha cloud therefore creates spurious
    N-dependence in ``r_99``.  We use it only in its exponentially converged
    overlap region.  An outward IVP covers the cloud core, while a second IVP
    is integrated inward from the decaying Coulomb asymptotic solution and is
    matched in the exponentially small tail.  The two-sided construction
    avoids the growing-solution contamination that otherwise appears after
    roughly twelve decay lengths and genuinely supports the 28/40 atlas
    outer-range convergence test.
    """
    b_value = math.sqrt(1.0 - chi * chi)
    r_plus = 1.0 + b_value
    gap = 2.0 * b_value
    decay = max(-_bound_kappa(alpha, omega).real, 1.0e-5)
    match_offset = max(1.0e-3, gap * len(coefficients) / 45.0)
    match = r_plus + match_offset
    # Transition integrals use a common physical outer radius.  For a pair
    # with different principal quantum numbers this can correspond to more
    # than forty decay lengths for the more compact mode, even though the
    # joint tail is still cut at 28/40 lengths of the slower mode.
    requested_lengths = min(float(outer_decay_lengths), 80.0)
    outer = r_plus + requested_lengths / decay
    # Outward propagation remains clean through the cloud support used by the
    # Ward audit (roughly <=11 decay lengths for the strongest pilot).  Match
    # at twelve lengths so the artificial two-sided stitch is always outside
    # the compact gauge support and its exponentially small derivative
    # mismatch cannot masquerade as a gauge-variation boundary term.
    stitch_lengths = min(requested_lengths, 12.0)
    stitch = r_plus + stitch_lengths / decay
    initial = radial_mode_values(
        match, alpha, chi, state, omega, coefficients, advanced=False
    )

    def equation(radius: float, vector: np.ndarray) -> np.ndarray:
        delta = radius * radius - 2.0 * radius + chi * chi
        potential = (
            (
                omega**2 * (radius * radius + chi * chi) ** 2
                - 4.0 * radius * chi * state.m * omega
                + state.m**2 * chi**2
            )
            / delta
            - omega**2 * chi**2
            - alpha**2 * radius**2
            - separation_constant
        )
        return np.asarray([
            vector[1],
            -(2.0 * (radius - 1.0) * vector[1] + potential * vector[0])
            / delta,
        ], dtype=complex)

    outward = solve_ivp(
        equation,
        (match, stitch),
        np.asarray(initial[:2], dtype=complex),
        method="DOP853",
        rtol=min(float(rtol), 1.0e-10),
        atol=1.0e-13 * max(abs(initial[0]), abs(initial[1]), 1.0e-30),
        dense_output=True,
    )
    if not outward.success or outward.sol is None:
        raise RuntimeError(f"outward radial ODE propagation failed: {outward.message}")

    inward = None
    inward_scale = 1.0 + 0.0j
    matching_residual = 0.0
    if outer > stitch * (1.0 + 1.0e-14):
        kappa = _bound_kappa(alpha, omega)
        coulomb_power = (alpha * alpha - 2.0 * omega * omega) / kappa - 1.0
        # An arbitrary unit amplitude is sufficient.  Boundary-condition
        # errors in this leading Coulomb logarithmic derivative are
        # exponentially suppressed by the long inward propagation.
        asymptotic = np.asarray(
            [1.0 + 0.0j, kappa + coulomb_power / outer], dtype=complex
        )
        inward = solve_ivp(
            equation,
            (outer, stitch),
            asymptotic,
            method="DOP853",
            rtol=min(float(rtol), 1.0e-10),
            atol=1.0e-13,
            dense_output=True,
        )
        if not inward.success or inward.sol is None:
            raise RuntimeError(
                f"inward radial ODE propagation failed: {inward.message}"
            )
        outward_at_stitch = outward.sol(stitch)
        inward_at_stitch = inward.sol(stitch)
        if abs(inward_at_stitch[0]) <= 1.0e-300:
            raise RuntimeError("inward radial solution vanished at the matching point")
        inward_scale = outward_at_stitch[0] / inward_at_stitch[0]
        matching_residual = float(
            abs(outward_at_stitch[1] - inward_scale * inward_at_stitch[1])
            / max(
                abs(outward_at_stitch[1]),
                abs(inward_scale * inward_at_stitch[1]),
                decay * abs(outward_at_stitch[0]),
                1.0e-300,
            )
        )

    def evaluate(radius: np.ndarray | float):
        radius_array = np.asarray(radius, dtype=float)
        if np.any(radius_array <= r_plus) or np.any(radius_array > outer):
            raise ValueError("hybrid radial evaluator called outside its domain")
        flat = radius_array.reshape(-1)
        values = np.empty(flat.shape, dtype=complex)
        first = np.empty(flat.shape, dtype=complex)
        second = np.empty(flat.shape, dtype=complex)
        inner = flat <= match
        if np.any(inner):
            local = radial_mode_values(
                flat[inner], alpha, chi, state, omega, coefficients,
                advanced=False,
            )
            values[inner], first[inner], second[inner] = local
        middle = (flat > match) & (flat <= stitch)
        if np.any(middle):
            propagated = outward.sol(flat[middle])
            values[middle], first[middle] = propagated
        tail = flat > stitch
        if np.any(tail):
            if inward is None or inward.sol is None:
                raise RuntimeError("radial tail requested without an inward solution")
            propagated = inward_scale * inward.sol(flat[tail])
            values[tail], first[tail] = propagated
        propagated_mask = ~inner
        if np.any(propagated_mask):
            for index, radius_value in zip(
                np.flatnonzero(propagated_mask), flat[propagated_mask], strict=True
            ):
                second[index] = equation(
                    float(radius_value),
                    np.asarray([values[index], first[index]], dtype=complex),
                )[1]
        shape = radius_array.shape
        output = (
            values.reshape(shape), first.reshape(shape), second.reshape(shape)
        )
        if radius_array.ndim == 0:
            return tuple(complex(item) for item in output)
        return output

    evaluate.match_radius = match
    evaluate.stitch_radius = stitch
    evaluate.outer_radius = outer
    evaluate.matching_residual = matching_residual
    return evaluate


def _complex_quad(function, lower: float, upper: float, rtol: float) -> complex:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        real = quad(lambda value: float(np.real(function(value))), lower, upper, epsabs=1e-11, epsrel=rtol, limit=300)[0]
        imag = quad(lambda value: float(np.imag(function(value))), lower, upper, epsabs=1e-11, epsrel=rtol, limit=300)[0]
    messages = [str(item.message) for item in caught
                if issubclass(item.category, IntegrationWarning)]
    if messages:
        raise RuntimeError("radial finite-part quadrature warning: " + "; ".join(messages))
    return complex(real, imag)


def _finite_part_radial_norm(
    alpha: float,
    chi: float,
    state: State,
    omega: complex,
    angular_gamma: complex,
    coefficients: np.ndarray,
    settings: KerrModeSettings,
    radial_evaluator=None,
) -> complex:
    b_value = math.sqrt(1.0 - chi * chi)
    r_plus = 1.0 + b_value
    gap = 2.0 * b_value
    decay = max(-_bound_kappa(alpha, omega).real, 1.0e-5)
    outer = r_plus + min(settings.outer_decay_lengths, 20.0) / decay
    if radial_evaluator is not None:
        outer = min(outer, float(radial_evaluator.outer_radius))
    sigma = 2.0 * r_plus * (omega - state.m * chi / (2.0 * r_plus)) / gap
    exponent = -2j * sigma

    def integrand(radius: float) -> complex:
        radial = (
            radial_mode_values(
                radius, alpha, chi, state, omega, coefficients, advanced=False
            )[0]
            if radial_evaluator is None
            else radial_evaluator(radius)[0]
        )
        delta = radius * radius - 2.0 * radius + chi * chi
        kernel = (
            2.0
            * omega
            * ((radius * radius + chi * chi) ** 2 - angular_gamma * chi * chi * delta)
            - 4.0 * radius * chi * state.m
        )
        return complex(kernel / delta * radial * radial)

    cutoff = settings.horizon_cutoff
    # Integrate the singular Frobenius sector in log distance from the
    # horizon.  Directly asking QUADPACK to resolve x**(exponent - 1) over
    # the full cloud interval can trigger a roundoff/non-convergence warning
    # at small cutoffs even though the Hadamard finite part is well behaved.
    # The logarithmic Jacobian removes the 1/x envelope.  The remaining
    # interval is split at every radial-provider matching surface so each
    # quadrature sees a smooth branch.
    near_extent = min(
        outer - r_plus,
        max(100.0 * cutoff, 0.05 * gap, 0.02),
    )
    log_lower = math.log(cutoff)
    log_upper = math.log(near_extent)
    integral = _complex_quad(
        lambda log_x: integrand(r_plus + math.exp(log_x)) * math.exp(log_x),
        log_lower,
        log_upper,
        settings.radial_rtol,
    )
    boundaries = [r_plus + near_extent, outer]
    if radial_evaluator is not None:
        boundaries.extend((
            float(radial_evaluator.match_radius),
            float(radial_evaluator.stitch_radius),
        ))
    boundaries = sorted({
        min(max(float(value), r_plus + near_extent), outer)
        for value in boundaries
    })
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        if upper > lower:
            integral += _complex_quad(integrand, lower, upper, settings.radial_rtol)
    # Fit the smooth factor multiplying x^(exponent-1), then analytically add
    # the omitted [0, cutoff] piece.  This is a local Hadamard finite part.
    order = max(settings.counterterm_order, 1)
    samples = cutoff * np.geomspace(0.35, 1.4, order + 3)
    smooth = np.asarray(
        [integrand(r_plus + x_value) * x_value ** (1.0 - exponent) for x_value in samples]
    )
    vandermonde = np.vander(samples, N=order + 1, increasing=True)
    local = np.linalg.lstsq(vandermonde, smooth, rcond=None)[0]
    counterterm = 0.0j
    for power, coefficient in enumerate(local):
        denominator = exponent + power
        if abs(denominator) < 1.0e-12:
            counterterm += coefficient * cmath.log(cutoff)
        else:
            counterterm += coefficient * cutoff**denominator / denominator
    return 1j * (integral + counterterm)


def radial_ode_crosscheck(
    alpha: float,
    chi: float,
    state: State,
    omega: complex,
    separation_constant: complex,
    radial_coefficients: np.ndarray,
    settings: KerrModeSettings,
) -> float:
    """Compare the Miller/series mode with an independent DOP853 IVP.

    The IVP is initialized at one interior point and propagated across the
    cloud core.  It therefore checks the radial recurrence as a function,
    rather than substituting its derivatives back into the same ODE.
    """
    r_plus = 1.0 + math.sqrt(1.0 - chi * chi)
    decay = max(-_bound_kappa(alpha, omega).real, 1.0e-5)
    start = r_plus + max(50.0 * settings.horizon_cutoff, 0.02 / decay)
    stop = r_plus + 0.50 / decay
    if not stop > start:
        stop = start + max(1.0, 0.10 / decay)
    samples = np.linspace(start, stop, 24)
    value, first, _ = radial_mode_values(
        start,
        alpha,
        chi,
        state,
        omega,
        radial_coefficients,
        advanced=False,
    )

    def equation(radius: float, vector: np.ndarray) -> np.ndarray:
        delta = radius * radius - 2.0 * radius + chi * chi
        potential = (
            (
                omega**2 * (radius * radius + chi * chi) ** 2
                - 4.0 * radius * chi * state.m * omega
                + state.m**2 * chi**2
            )
            / delta
            - omega**2 * chi**2
            - alpha**2 * radius**2
            - separation_constant
        )
        return np.asarray([
            vector[1],
            -(2.0 * (radius - 1.0) * vector[1] + potential * vector[0])
            / delta,
        ], dtype=complex)

    solution = solve_ivp(
        equation,
        (start, stop),
        np.asarray([value, first], dtype=complex),
        t_eval=samples,
        method="DOP853",
        rtol=min(settings.radial_rtol, 1.0e-10),
        atol=1.0e-12 * max(abs(value), abs(first), 1.0e-30),
    )
    if not solution.success or solution.y.shape[1] != len(samples):
        return math.inf
    reference = radial_mode_values(
        samples,
        alpha,
        chi,
        state,
        omega,
        radial_coefficients,
        advanced=False,
    )[0]
    scale = np.maximum.reduce((
        np.abs(reference),
        np.abs(solution.y[0]),
        np.full_like(np.abs(reference), 1.0e-30),
    ))
    return float(np.max(np.abs(reference - solution.y[0]) / scale))


def solve_kerr_mode(
    alpha: float,
    chi: float,
    state: State,
    settings: KerrModeSettings = KerrModeSettings(),
    initial_guess: complex | None = None,
) -> KerrModeResult:
    """Solve and bilinearly normalize a massive scalar Kerr QBS."""
    cf = solve_quasibound_cf(
        alpha, chi, state, settings.cf_settings(), initial_guess
    )
    if not cf.converged:
        seed_chi = max(1.0e-5, chi - 2.0e-3)
        seed = solve_quasibound_cf(alpha, seed_chi, state, settings.cf_settings())
        if seed.converged or (
            np.isfinite(seed.residual) and seed.residual < 1.0e-6
        ):
            cf = solve_quasibound_cf(
                alpha, chi, state, settings.cf_settings(), seed.frequency_M
            )
    if not cf.converged:
        return KerrModeResult(
            alpha, chi, state, cf.frequency_M, complex(np.nan), np.array([], int),
            np.array([], complex), np.array([], complex), complex(np.nan), math.nan,
            cf.residual, math.inf, math.inf, False, f"CF failed: {cf.message}"
        )
    omega = cf.frequency_M
    angular, l_values, angular_coefficients = spheroidal_eigensystem(
        chi * chi * (omega * omega - alpha * alpha), state, settings.angular_lmax
    )
    radial_coefficients = leaver_radial_coefficients(
        alpha, chi, state, omega, angular, settings.series_terms
    )
    theta = np.linspace(2.0e-4, math.pi - 2.0e-4, settings.angular_nodes)
    angular_value, _, _ = angular_mode_values(
        theta, state, angular, l_values, angular_coefficients
    )
    angular_residual = angular_mode_residual(
        theta,
        state,
        alpha,
        chi,
        omega,
        angular,
        l_values,
        angular_coefficients,
        settings.angular_lmax,
    )
    gamma_angular = 2.0 * math.pi * np.trapz(
        np.sin(theta) ** 3 * angular_value * angular_value, theta
    )
    radial_evaluator = build_radial_ode_evaluator(
        alpha,
        chi,
        state,
        omega,
        angular,
        radial_coefficients,
        outer_decay_lengths=settings.outer_decay_lengths,
        rtol=settings.radial_rtol,
    )
    bilinear_norm = _finite_part_radial_norm(
        alpha, chi, state, omega, gamma_angular, radial_coefficients, settings,
        radial_evaluator,
    )
    normalization = cmath.sqrt(bilinear_norm)
    radial_coefficients = radial_coefficients / normalization
    radial_evaluator = build_radial_ode_evaluator(
        alpha,
        chi,
        state,
        omega,
        angular,
        radial_coefficients,
        outer_decay_lengths=settings.outer_decay_lengths,
        rtol=settings.radial_rtol,
    )
    bilinear_norm = _finite_part_radial_norm(
        alpha, chi, state, omega, gamma_angular, radial_coefficients, settings,
        radial_evaluator,
    )
    b_value = math.sqrt(1.0 - chi * chi)
    r_plus = 1.0 + b_value
    decay = max(-_bound_kappa(alpha, omega).real, 1.0e-5)
    radial_grid = r_plus + np.geomspace(
        settings.horizon_cutoff,
        radial_evaluator.match_radius - r_plus,
        1600,
    )
    radial_value, radial_first, radial_second = radial_mode_values(
        radial_grid, alpha, chi, state, omega, radial_coefficients, advanced=False
    )
    delta = radial_grid**2 - 2.0 * radial_grid + chi * chi
    radial_term_one = delta * radial_second
    radial_term_two = 2.0 * (radial_grid - 1.0) * radial_first
    radial_term_three = (
            (
                omega**2 * (radial_grid**2 + chi**2) ** 2
                - 4.0 * radial_grid * chi * state.m * omega
                + state.m**2 * chi**2
            )
            / delta
            - omega**2 * chi**2
            - alpha**2 * radial_grid**2
            - angular
        ) * radial_value
    radial_equation = radial_term_one + radial_term_two + radial_term_three
    radial_residual = float(
        np.max(
            np.abs(radial_equation[10:])
            / (
                np.abs(radial_term_one[10:])
                + np.abs(radial_term_two[10:])
                + np.abs(radial_term_three[10:])
                + 1.0e-30
            )
        )
    )
    radial_ode_residual = radial_ode_crosscheck(
        alpha,
        chi,
        state,
        omega,
        angular,
        radial_coefficients,
        settings,
    )
    probability_grid = r_plus + np.geomspace(
        settings.horizon_cutoff,
        radial_evaluator.outer_radius - r_plus,
        4000,
    )
    bl_value = radial_evaluator(probability_grid)[0]
    sigma = 2.0 * r_plus * (
        omega - state.m * chi / (2.0 * r_plus)
    ) / (2.0 * b_value)
    ratio = np.exp(
        1j * sigma * np.log(
            (probability_grid - r_plus) / (probability_grid - (1.0 - b_value))
        )
        + 1j * omega * probability_grid
    )
    advanced_value = bl_value * ratio
    density = (probability_grid**2 + chi**2) * np.abs(advanced_value) ** 2
    cumulative = cumulative_trapezoid(density, probability_grid, initial=0.0)
    if cumulative[-1] <= 0.0 or not np.isfinite(cumulative[-1]):
        r_99 = math.nan
    else:
        r_99 = float(
            np.interp(0.99 * cumulative[-1], cumulative, probability_grid)
        )
    converged = bool(
        np.isfinite(abs(bilinear_norm))
        and abs(bilinear_norm - 1.0) < 1.0e-5
        and angular_residual < 1.0e-8
        and radial_residual < 1.0e-8
        and radial_ode_residual < 1.0e-6
    )
    message = "ok" if converged else "mode-function or normalization residual failed"
    return KerrModeResult(
        alpha, chi, state, omega, angular, l_values, angular_coefficients,
        radial_coefficients, bilinear_norm, r_99, cf.residual, radial_residual,
        angular_residual, converged, message, settings.truncation, (),
        radial_ode_residual,
    )


def _mode_pair_convergence(
    candidate: KerrModeResult,
    reference: KerrModeResult,
) -> tuple[bool, dict[str, float | bool | int]]:
    real_relative = abs(candidate.frequency_M.real - reference.frequency_M.real) / max(
        abs(reference.frequency_M.real), 1.0e-30
    )
    gamma_absolute = abs(candidate.frequency_M.imag - reference.frequency_M.imag)
    gamma_scale = max(
        abs(candidate.frequency_M.imag), abs(reference.frequency_M.imag)
    )
    gamma_metric = (
        gamma_absolute / gamma_scale if gamma_scale >= 1.0e-14 else gamma_absolute
    )
    gamma_limit = 1.0e-3 if gamma_scale >= 1.0e-14 else 1.0e-15
    r99_relative = abs(candidate.r_99_M - reference.r_99_M) / max(
        abs(reference.r_99_M), 1.0e-30
    )
    passed = bool(
        candidate.converged
        and reference.converged
        and real_relative < 1.0e-8
        and gamma_metric < gamma_limit
        and r99_relative < 1.0e-4
    )
    return passed, {
        "candidate_N": int(candidate.selected_truncation or 0),
        "reference_N": int(reference.selected_truncation or 0),
        "real_relative": float(real_relative),
        "gamma_metric": float(gamma_metric),
        "gamma_limit": float(gamma_limit),
        "r99_relative": float(r99_relative),
        "candidate_mode_converged": bool(candidate.converged),
        "reference_mode_converged": bool(reference.converged),
        "passed": passed,
    }


def solve_kerr_mode_adaptive(
    alpha: float,
    chi: float,
    state: State,
    settings: KerrModeSettings = KerrModeSettings(),
    seed: complex | KerrModeResult | None = None,
) -> KerrModeResult:
    """Select the lowest mode truncation certified by the next rung.

    ``N=1600`` is reference-only.  Returning it as a production mode would
    hide a failure to meet the advertised ``N<=1200`` resource contract.
    """
    ladder = tuple(sorted(set(int(value) for value in settings.adaptive_truncations)))
    if len(ladder) < 2 or ladder[-1] < 1600:
        raise ValueError("adaptive truncations must contain a reference rung >= 1600")
    if any(value < 20 for value in ladder):
        raise ValueError("all adaptive truncations must be at least 20")
    initial_guess = (
        seed.frequency_M if isinstance(seed, KerrModeResult) else seed
    )
    modes: list[KerrModeResult] = []
    history: list[dict[str, float | bool | int]] = []
    for truncation in ladder:
        rung_settings = replace(
            settings,
            truncation=truncation,
            series_terms=truncation,
        )
        mode = solve_kerr_mode(
            alpha,
            chi,
            state,
            rung_settings,
            initial_guess=initial_guess,
        )
        mode = replace(mode, selected_truncation=truncation)
        modes.append(mode)
        if mode.converged:
            initial_guess = mode.frequency_M
        if len(modes) >= 2:
            passed, row = _mode_pair_convergence(modes[-2], modes[-1])
            history.append(row)
            candidate_n = modes[-2].selected_truncation or 0
            if passed and candidate_n <= 1200:
                return replace(
                    modes[-2],
                    selected_truncation=candidate_n,
                    convergence_history=tuple(history),
                    message=(
                        f"ok; adaptive N={candidate_n}, reference N={truncation}"
                    ),
                )
    last = modes[-1]
    return replace(
        last,
        converged=False,
        convergence_history=tuple(history),
        message="adaptive truncation failed before the N=1600 reference rung",
    )


def solve_saturation_mode_cf(
    alpha: float,
    state: State,
    settings: KerrModeSettings = KerrModeSettings(),
    seed: complex | None = None,
) -> SaturationResult:
    """Solve ``Re(omega)=m Omega_H`` with an outer bracketed root.

    The inner CF solve is continued from the nearest already evaluated spin.
    This keeps the radial overtone fixed without using the vanishing physical
    growth rate as a numerical coordinate.
    """
    if state.m <= 0:
        raise ValueError("saturation requires a prograde state")
    cf_settings = settings.cf_settings()
    seed_real = alpha * (1.0 - alpha**2 / (2.0 * state.n**2))
    initial_seed = seed if seed is not None else complex(seed_real, 0.0)
    solved: dict[float, CFResult] = {}

    def evaluate(chi: float) -> float:
        if solved:
            nearest = min(solved, key=lambda value: abs(value - chi))
            guess = solved[nearest].frequency_M
        else:
            guess = initial_seed
        result = solve_quasibound_cf(
            alpha, chi, state, cf_settings, guess
        )
        if not result.converged:
            # A hydrogenic seed is independent of continuation and prevents a
            # single poor neighbor from poisoning the whole branch.
            fallback = solve_quasibound_cf(
                alpha,
                chi,
                state,
                cf_settings,
                complex(
                    omega_real_M(alpha, chi, state)
                    if alpha <= 0.3
                    else seed_real,
                    0.0,
                ),
            )
            if fallback.residual < result.residual:
                result = fallback
        if not result.converged:
            raise RuntimeError(
                f"CF saturation branch failed for {state.label} at chi={chi:.9f}: "
                f"residual={result.residual:.3e}"
            )
        solved[float(chi)] = result
        return float(result.frequency_M.real - state.m * horizon_omega_M(chi))

    approximate_ratio = seed_real / state.m
    approximate_chi = 4.0 * approximate_ratio / (1.0 + 4.0 * approximate_ratio**2)
    lower = max(1.0e-8, 0.6 * approximate_chi)
    # Keep the first bracket local to the hydrogenic threshold estimate.
    # The former 1.35 factor unnecessarily evaluated moderate-spin branches
    # at chi=0.999999, where the CF Jacobian is ill-conditioned even though
    # the physical root lies near chi~0.7.
    upper = min(0.999999, 1.10 * approximate_chi + 0.03)
    lower_value = evaluate(lower)
    upper_value = evaluate(upper)
    if lower_value * upper_value > 0.0:
        lower = 1.0e-8
        upper = 0.999999
        lower_value = evaluate(lower)
        upper_value = evaluate(upper)
    if lower_value * upper_value > 0.0:
        raise RuntimeError(f"could not bracket saturation spin for {state.label}")
    chi = float(
        brentq(
            evaluate,
            lower,
            upper,
            xtol=2.0e-13,
            rtol=4.0 * np.finfo(float).eps,
            maxiter=100,
        )
    )
    nearest = min(solved, key=lambda value: abs(value - chi))
    final = solve_quasibound_cf(
        alpha, chi, state, cf_settings, solved[nearest].frequency_M
    )
    saturation_residual = abs(
        final.frequency_M.real - state.m * horizon_omega_M(chi)
    )
    converged = bool(
        final.converged
        and saturation_residual < 1.0e-12
        and abs(final.frequency_M.imag) < 1.0e-10
    )
    return SaturationResult(
        alpha=alpha,
        state=state,
        chi=chi,
        frequency_M=final.frequency_M,
        cf_residual=final.residual,
        saturation_residual=float(saturation_residual),
        truncation=settings.truncation,
        converged=converged,
        message="ok" if converged else "final saturation root failed acceptance",
    )


def trace_saturation_branch_cf(
    alphas: np.ndarray | list[float] | tuple[float, ...],
    state: State,
    settings: KerrModeSettings = KerrModeSettings(),
) -> list[SaturationResult]:
    """Trace a saturation branch in both directions and cross-check labels."""
    requested = [float(value) for value in alphas]
    if len(set(requested)) != len(requested):
        raise ValueError("saturation alpha grid must not contain duplicates")

    def trace(values: list[float]) -> dict[float, SaturationResult]:
        rows: dict[float, SaturationResult] = {}
        seed: complex | None = None
        previous_alpha: float | None = None
        for alpha_value in values:
            continued_seed = seed
            if seed is not None and previous_alpha is not None:
                current_hydrogenic = alpha_value * (
                    1.0 - alpha_value**2 / (2.0 * state.n**2)
                )
                previous_hydrogenic = previous_alpha * (
                    1.0 - previous_alpha**2 / (2.0 * state.n**2)
                )
                continued_seed = seed + (
                    current_hydrogenic - previous_hydrogenic
                )
            row = solve_saturation_mode_cf(
                alpha_value, state, settings, seed=continued_seed
            )
            if not row.converged:
                raise RuntimeError(
                    f"saturation branch failed for {state.label} at alpha={alpha_value}"
                )
            rows[alpha_value] = row
            seed = row.frequency_M
            previous_alpha = alpha_value
        return rows

    ascending_values = sorted(requested)
    ascending = trace(ascending_values)
    descending = trace(list(reversed(ascending_values)))
    output: list[SaturationResult] = []
    for alpha_value in requested:
        forward = ascending[alpha_value]
        reverse = descending[alpha_value]
        difference = abs(forward.chi - reverse.chi)
        if difference >= 1.0e-10:
            raise RuntimeError(
                f"saturation branch direction mismatch for {state.label} "
                f"at alpha={alpha_value}: {difference:.3e}"
            )
        output.append(
            replace(
                forward,
                message=f"ok; direction_difference={difference:.3e}",
            )
        )
    return output


def saturation_spin_cf(
    alpha: float,
    state: State,
    settings: KerrModeSettings = KerrModeSettings(),
) -> float:
    """Compatibility wrapper returning only the certified saturation spin."""
    result = solve_saturation_mode_cf(alpha, state, settings)
    if not result.converged:
        raise RuntimeError(
            f"CF saturation branch failed for {state.label}: "
            f"residual={result.cf_residual:.3e}"
        )
    return result.chi


def solve_quasibound_cf_sympy(
    alpha: float,
    chi: float,
    state: State,
    settings: CFSettings = CFSettings(truncation=100),
    digits: int = 60,
    initial_guess: complex | None = None,
) -> CFResult:
    """Low-alpha arbitrary-precision cross-check using SymPy ``nsolve``.

    For the l=1 branch the angular eigenvalue is expanded through c^4; the
    returned root is accepted only if it also satisfies the full double-
    precision spectral-angular continued fraction.  This is a cross-check,
    not a silent replacement for a failed calibrated point.
    """
    if state.l != 1:
        raise NotImplementedError("the high-precision cross-check supports l=1")
    if digits < 40:
        raise ValueError("digits must be at least 40")
    import sympy as sp

    if initial_guess is None:
        initial_guess = complex(
            omega_real_M(alpha, chi, state),
            gamma_detweiler_M(alpha, chi, state),
        )
    precision = digits
    alpha_sp = sp.Float(alpha, precision)
    chi_sp = sp.Float(chi, precision)
    b_value = sp.sqrt(1 - chi_sp**2)
    omega = sp.Symbol("omega")
    kappa = -sp.sqrt(alpha_sp**2 - omega**2)
    c_squared = chi_sp**2 * (omega**2 - alpha_sp**2)
    angular = sp.Integer(2) - c_squared / 5 - 4 * c_squared**2 / 875
    shifted = omega - sp.I * kappa
    horizon_shift = omega - chi_sp * state.m / 2
    c0 = 1 - 2 * sp.I * omega - 2 * sp.I * horizon_shift / b_value
    c1 = (
        -4
        + 4 * sp.I * (omega - sp.I * kappa * (1 + b_value))
        + 4 * sp.I * horizon_shift / b_value
        - 2 * (omega**2 + kappa**2) / kappa
    )
    c2 = (
        3
        - 2 * sp.I * omega
        - 2 * (kappa**2 - omega**2) / kappa
        - 2 * sp.I * horizon_shift / b_value
    )
    c3 = (
        2 * sp.I * shifted**3 / kappa
        + 2 * shifted**2 * b_value
        + kappa**2 * chi_sp**2
        + 2 * sp.I * kappa * chi_sp * state.m
        - angular
        - 1
        - shifted**2 / kappa
        + 2 * kappa * b_value
        + 2 * sp.I / b_value * (shifted**2 / kappa + 1) * horizon_shift
    )
    c4 = (
        shifted**4 / kappa**2
        + 2 * sp.I * omega * shifted**2 / kappa
        - 2 * sp.I / b_value * shifted**2 / kappa * horizon_shift
    )

    def alpha_n(index: int):
        return index**2 + (c0 + 1) * index + c0

    def beta_n(index: int):
        return -2 * index**2 + (c1 + 2) * index + c3

    def gamma_n(index: int):
        return index**2 + (c2 - 3) * index + c4

    truncation = min(settings.truncation, 120)
    denominator = beta_n(truncation)
    for index in range(truncation - 1, -1, -1):
        denominator = beta_n(index) - alpha_n(index) * gamma_n(index + 1) / denominator
    try:
        omega_real, omega_imag = sp.symbols("omega_real omega_imag", real=True)
        split_residual = denominator.subs(
            omega, omega_real + sp.I * omega_imag
        )
        root_value = sp.nsolve(
            (sp.re(split_residual), sp.im(split_residual)),
            (omega_real, omega_imag),
            (
                sp.Float(initial_guess.real, precision),
                sp.Float(initial_guess.imag, precision),
            ),
            tol=sp.Float(10, precision) ** (-(digits - 10)),
            maxsteps=100,
            prec=precision,
        )
        frequency = complex(float(root_value[0]), float(root_value[1]))
        residual = abs(
            radial_continued_fraction_residual(
                alpha, chi, state, frequency, settings
            )
        ) / max(1.0, abs(frequency))
        converged = bool(
            np.isfinite(residual)
            and residual <= settings.residual_tolerance
            and 0.0 < frequency.real < alpha
        )
        message = (
            f"SymPy nsolve ({digits} digits, angular c^4 cross-check); "
            f"full residual={residual:.3e}"
        )
    except (ValueError, ZeroDivisionError) as error:
        frequency = initial_guess
        residual = math.inf
        converged = False
        message = f"SymPy nsolve failed: {error}"
    return CFResult(
        alpha=alpha,
        chi=chi,
        state=state,
        frequency_M=frequency,
        residual=float(residual),
        truncation=settings.truncation,
        angular_lmax=settings.angular_lmax,
        converged=converged,
        message=message,
    )
