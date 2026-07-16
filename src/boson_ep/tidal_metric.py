"""Adiabatic quadrupolar Kerr tide in ingoing radiation gauge.

The implementation follows the l=2 specialization of Katagiri and Cardoso
(2026), arXiv:2601.14979.  It returns the complex coefficient multiplying
``exp[-2 i (phi - Omega v)]`` per unit tidal strength epsilon.  The physical
metric is obtained by adding the complex conjugate harmonic.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
import sympy as sp

from .models import GaugeMetricConfig, GaugeMetricResult, GaugeVectorSpec


def kerr_metric_advanced(radius: float, theta: float, chi: float) -> np.ndarray:
    """Kerr metric in advanced coordinates (v,r,theta,phi), with M=1."""
    sine = math.sin(theta)
    sigma = radius * radius + chi * chi * math.cos(theta) ** 2
    metric = np.zeros((4, 4), dtype=float)
    metric[0, 0] = -(1.0 - 2.0 * radius / sigma)
    metric[0, 1] = metric[1, 0] = 1.0
    metric[0, 3] = metric[3, 0] = -2.0 * radius * chi * sine**2 / sigma
    metric[1, 3] = metric[3, 1] = -chi * sine**2
    metric[2, 2] = sigma
    metric[3, 3] = (
        radius * radius
        + chi * chi
        + 2.0 * radius * chi * chi * sine**2 / sigma
    ) * sine**2
    return metric


@lru_cache(maxsize=131072)
def kerr_metric_boyer_lindquist(
    radius: float, theta: float, chi: float
) -> np.ndarray:
    """Kerr metric in Boyer-Lindquist coordinates (t,r,theta,phi)."""
    sine = math.sin(theta)
    sigma = radius * radius + chi * chi * math.cos(theta) ** 2
    delta = radius * radius - 2.0 * radius + chi * chi
    metric = np.zeros((4, 4), dtype=float)
    metric[0, 0] = -(1.0 - 2.0 * radius / sigma)
    metric[0, 3] = metric[3, 0] = -2.0 * radius * chi * sine**2 / sigma
    metric[1, 1] = sigma / delta
    metric[2, 2] = sigma
    metric[3, 3] = (
        radius * radius
        + chi * chi
        + 2.0 * radius * chi * chi * sine**2 / sigma
    ) * sine**2
    return metric


@lru_cache(maxsize=1)
def _bl_christoffel_function():
    """Build an analytic Boyer-Lindquist connection once with SymPy."""
    r, theta, a = sp.symbols("r theta a", real=True)
    sine = sp.sin(theta)
    sigma = r**2 + a**2 * sp.cos(theta) ** 2
    delta = r**2 - 2 * r + a**2
    metric = sp.MutableDenseMatrix.zeros(4, 4)
    metric[0, 0] = -(1 - 2 * r / sigma)
    metric[0, 3] = metric[3, 0] = -2 * r * a * sine**2 / sigma
    metric[1, 1] = sigma / delta
    metric[2, 2] = sigma
    metric[3, 3] = (
        r**2 + a**2 + 2 * r * a**2 * sine**2 / sigma
    ) * sine**2
    inverse = metric.inv()
    coordinates = (None, r, theta, None)
    connection: list[sp.Expr] = []
    for upper in range(4):
        for first in range(4):
            for second in range(4):
                value = 0
                for lower in range(4):
                    derivative_one = (
                        sp.diff(metric[lower, second], coordinates[first])
                        if coordinates[first] is not None
                        else 0
                    )
                    derivative_two = (
                        sp.diff(metric[lower, first], coordinates[second])
                        if coordinates[second] is not None
                        else 0
                    )
                    derivative_three = (
                        sp.diff(metric[first, second], coordinates[lower])
                        if coordinates[lower] is not None
                        else 0
                    )
                    value += inverse[upper, lower] * (
                        derivative_one + derivative_two - derivative_three
                    ) / 2
                connection.append(sp.simplify(value))
    return sp.lambdify((r, theta, a), connection, modules="numpy", cse=True)


@lru_cache(maxsize=131072)
def boyer_lindquist_christoffel(
    radius: float, theta: float, chi: float
) -> np.ndarray:
    """Return ``Gamma^a_bc`` for Kerr in Boyer-Lindquist coordinates."""
    values = np.asarray(
        _bl_christoffel_function()(radius, theta, chi), dtype=float
    )
    return values.reshape((4, 4, 4))


def _bump_value_and_derivative(
    radius: float, support_inner: float, support_outer: float
) -> tuple[float, float]:
    """C-infinity compact bump, normalized to one at the support midpoint."""
    center = 0.5 * (support_inner + support_outer)
    half_width = 0.5 * (support_outer - support_inner)
    x_value = (radius - center) / half_width
    if abs(x_value) >= 1.0:
        return 0.0, 0.0
    denominator = 1.0 - x_value * x_value
    value = math.exp(1.0 - 1.0 / denominator)
    derivative = value * (-2.0 * x_value / denominator**2) / half_width
    return value, derivative


def compact_support_gauge_metric(
    config: GaugeMetricConfig,
    vector_spec: GaugeVectorSpec,
) -> GaugeMetricResult:
    """Return the harmonic coefficient of ``2 nabla_(a xi_b)``.

    The covector coefficient carries the common harmonic
    ``exp(2 i Omega t-2 i phi)``.  ``absolute_amplitude`` is deliberately
    separate from the requested relative amplitude because the Ward audit
    calibrates the former against the physical IRG tide on each support.
    """
    radius = config.radius
    theta = config.theta
    bump, bump_first = _bump_value_and_derivative(
        radius, vector_spec.support_inner, vector_spec.support_outer
    )
    if bump == 0.0:
        return GaugeMetricResult(
            np.zeros((4, 4), dtype=complex),
            np.zeros(4, dtype=complex),
            0.0,
            False,
        )

    sine = math.sin(theta)
    cosine = math.cos(theta)
    normalization = math.sqrt(15.0 / (32.0 * math.pi))
    harmonic = normalization * sine**2
    harmonic_first = 2.0 * normalization * sine * cosine
    harmonic_second = 2.0 * normalization * (cosine**2 - sine**2)

    angular = np.zeros(4, dtype=complex)
    angular_first = np.zeros(4, dtype=complex)
    if vector_spec.kind == "temporal":
        angular[0] = harmonic
        angular_first[0] = harmonic_first
    elif vector_spec.kind == "radial":
        angular[1] = harmonic
        angular_first[1] = harmonic_first
    elif vector_spec.kind == "polar":
        angular[2] = harmonic_first
        angular[3] = -2j * harmonic
        angular_first[2] = harmonic_second
        angular_first[3] = -2j * harmonic_first
    else:
        angular[2] = -2j * normalization * sine
        angular[3] = -2.0 * normalization * sine**2 * cosine
        angular_first[2] = -2j * normalization * cosine
        angular_first[3] = -2.0 * normalization * (
            2.0 * sine * cosine**2 - sine**3
        )

    amplitude = vector_spec.absolute_amplitude
    covector = amplitude * bump * angular
    partial = np.zeros((4, 4), dtype=complex)
    partial[0] = 2j * config.omega_orb_M * covector
    partial[1] = amplitude * bump_first * angular
    partial[2] = amplitude * bump * angular_first
    partial[3] = -2j * covector
    connection = boyer_lindquist_christoffel(radius, theta, config.chi)
    covariant = partial - np.einsum("cab,c->ab", connection, covector)
    perturbation = covariant + covariant.T
    return GaugeMetricResult(perturbation, covector, bump, True)


def kinnersley_l_advanced(radius: float, chi: float) -> np.ndarray:
    delta = radius * radius - 2.0 * radius + chi * chi
    return np.asarray(
        [2.0 * (radius * radius + chi * chi) / delta, 1.0, 0.0, 2.0 * chi / delta]
    )


@lru_cache(maxsize=1)
def _metric_function():
    r, theta, a, omega = sp.symbols("r theta a omega", real=True)
    imaginary = sp.I
    sine = sp.sin(theta)
    cosine = sp.cos(theta)
    delta_kerr = r**2 - 2 * r + a**2
    sigma = r**2 + a**2 * cosine**2
    zeta = r - imaginary * a * cosine
    bar_zeta = r + imaginary * a * cosine
    beta = sp.cot(theta) / (2 * sp.sqrt(2) * bar_zeta)
    pi_value = imaginary * a * sine / (sp.sqrt(2) * zeta**2)
    bar_pi = -imaginary * a * sine / (sp.sqrt(2) * bar_zeta**2)
    tau = -imaginary * a * sine / (sp.sqrt(2) * zeta * bar_zeta)
    # The l=2 Hertz coefficient includes the factor of two in D^4 Psi=2 psi_0.
    # The /8 normalization is independently fixed by the explicit
    # Schwarzschild equatorial metric: h_vv^(m=-2)=3(r-2)^2/4 per epsilon.
    hertz = -delta_kerr**2 * (1 - cosine) ** 2 / 8

    l_v = 2 * (r**2 + a**2) / delta_kerr
    l_phi = 2 * a / delta_kerr
    m_v = imaginary * a * sine / (sp.sqrt(2) * bar_zeta)
    m_theta = 1 / (sp.sqrt(2) * bar_zeta)
    m_phi = imaginary / (sp.sqrt(2) * bar_zeta * sine)
    dv_phase = 2 * imaginary * omega
    dphi_phase = -2 * imaginary

    def d_op(expression):
        return sp.diff(expression, r) + (l_v * dv_phase + l_phi * dphi_phase) * expression

    def delta_op(expression):
        return (
            m_theta * sp.diff(expression, theta)
            + (m_v * dv_phase + m_phi * dphi_phase) * expression
        )

    def shifted_delta(expression, shift):
        return delta_op(expression) + shift * expression

    def shifted_d(expression, shift):
        return d_op(expression) + shift * expression

    h_nn = -shifted_delta(
        shifted_delta(hertz, 4 * beta + 3 * tau),
        2 * beta + bar_pi - tau,
    )
    h_nmb = -sp.Rational(1, 2) * (
        shifted_delta(shifted_d(hertz, -3 / zeta), 4 * beta - 2 * bar_pi - tau)
        + shifted_d(shifted_delta(hertz, 4 * beta + 3 * tau), -1 / bar_zeta + 1 / zeta)
    )
    h_mbmb = -shifted_d(shifted_d(hertz, -3 / zeta), 1 / zeta)
    x_value = h_nmb / bar_zeta
    y_value = h_mbmb / bar_zeta**2
    re_x = x_value / 2
    im_x = x_value / (2 * imaginary)
    re_y = y_value / 2
    im_y = y_value / (2 * imaginary)
    h = sp.MutableDenseMatrix.zeros(4, 4)
    h[0, 0] = h_nn + 2 * sp.sqrt(2) * a * sine * im_x - a**2 * sine**2 * re_y
    h[0, 1] = -2 * sigma / delta_kerr * (h_nn + sp.sqrt(2) * a * sine * im_x)
    h[0, 2] = sp.sqrt(2) * sigma * re_x + a * sigma * sine * im_y
    h[0, 3] = (
        -sp.sqrt(2) * sine * (sigma + 2 * a**2 * sine**2) * im_x
        - a * sine**2 * (h_nn - (r**2 + a**2) * re_y)
    )
    h[1, 1] = 4 * sigma**2 / delta_kerr**2 * h_nn
    h[1, 2] = -2 * sp.sqrt(2) * sigma**2 / delta_kerr * re_x
    h[1, 3] = (
        2 * sigma / delta_kerr * a * sine**2 * h_nn
        + 2 * sp.sqrt(2) * sigma / delta_kerr * (r**2 + a**2) * sine * im_x
    )
    h[2, 2] = sigma**2 * re_y
    h[2, 3] = -sp.sqrt(2) * a * sigma * sine**2 * re_x - sigma * sine * (r**2 + a**2) * im_y
    h[3, 3] = (
        a**2 * sine**4 * h_nn
        + 2 * sp.sqrt(2) * a * sine**3 * (r**2 + a**2) * im_x
        - (r**2 + a**2) ** 2 * sine**2 * re_y
    )
    for row in range(4):
        for column in range(row):
            h[row, column] = h[column, row]
    flattened = [h[row, column] for row in range(4) for column in range(4)]
    function = sp.lambdify((r, theta, a, omega), flattened, modules="numpy", cse=True)
    return function


def irg_metric_coefficient(
    radius: float,
    theta: float,
    chi: float,
    omega_orb_M: float,
) -> np.ndarray:
    """Return the complex l=2,m=-2 IRG metric coefficient per epsilon."""
    if radius <= 1.0 + math.sqrt(1.0 - chi * chi):
        raise ValueError("IRG tide is evaluated outside the future horizon")
    values = np.asarray(
        _metric_function()(radius, theta, chi, omega_orb_M), dtype=complex
    )
    return values.reshape((4, 4))


def _advanced_phase(radius: float, chi: float, omega_orb_M: float) -> complex:
    """Radial phase relating the advanced and BL tidal harmonics."""
    root = math.sqrt(1.0 - chi * chi)
    r_plus = 1.0 + root
    r_minus = 1.0 - root
    gap = r_plus - r_minus
    x = radius - r_plus
    y = radius - r_minus
    tortoise = (
        radius
        + 2.0 * r_plus / gap * math.log(x)
        - 2.0 * r_minus / gap * math.log(y)
    )
    azimuth_shift = chi / gap * math.log(x / y)
    return np.exp(2j * omega_orb_M * tortoise - 2j * azimuth_shift)


@lru_cache(maxsize=131072)
def irg_metric_coefficient_bl(
    radius: float,
    theta: float,
    chi: float,
    omega_orb_M: float,
) -> np.ndarray:
    """Transform the IRG harmonic coefficient to Boyer-Lindquist coordinates."""
    delta = radius * radius - 2.0 * radius + chi * chi
    jacobian = np.eye(4)
    jacobian[0, 1] = (radius * radius + chi * chi) / delta
    jacobian[3, 1] = chi / delta
    advanced = irg_metric_coefficient(radius, theta, chi, omega_orb_M)
    return (
        jacobian.T
        @ advanced
        @ jacobian
        * _advanced_phase(radius, chi, omega_orb_M)
    )


def schwarzschild_rw_metric_coefficient(
    radius: float,
    theta: float,
    omega_orb_M: float = 0.0,
) -> np.ndarray:
    """Schwarzschild l=2,m=-2 tide in Regge-Wheeler gauge per epsilon.

    This is Eq. (tidally deformed Schwarzschild metric) of Katagiri and
    Cardoso, with the complex harmonic coefficient from their Eq. (TidalStrength).
    The metric is expressed in static coordinates (t,r,theta,phi).
    """
    del omega_orb_M  # Time dependence lives in the external harmonic.
    if radius <= 2.0:
        raise ValueError("RW tide is evaluated outside the Schwarzschild horizon")
    sine = math.sin(theta)
    f_value = 1.0 - 2.0 / radius
    tidal_harmonic = -0.75 * sine**2
    perturbation = np.zeros((4, 4), dtype=complex)
    perturbation[0, 0] = -radius**2 * f_value**2 * tidal_harmonic
    perturbation[1, 1] = -radius**2 * tidal_harmonic
    angular_factor = -radius**4 * (1.0 - 2.0 / radius**2) * tidal_harmonic
    perturbation[2, 2] = angular_factor
    perturbation[3, 3] = angular_factor * sine**2
    return perturbation


def irg_algebraic_residuals(
    radius: float,
    theta: float,
    chi: float,
    omega_orb_M: float,
) -> tuple[float, float]:
    """Return normalized trace and l^a h_ab residuals."""
    metric = kerr_metric_advanced(radius, theta, chi)
    inverse = np.linalg.inv(metric)
    perturbation = irg_metric_coefficient(radius, theta, chi, omega_orb_M)
    scale = max(float(np.max(np.abs(perturbation))), 1.0e-30)
    trace = abs(np.einsum("ab,ab->", inverse, perturbation)) / scale
    tetrad = kinnersley_l_advanced(radius, chi) @ perturbation
    tetrad_residual = float(np.max(np.abs(tetrad)) / scale)
    return float(trace), tetrad_residual
