"""Physical scalar currents on spacelike, horizon-penetrating Kerr slices.

Units are M=1 and the signature is (-,+,+,+). Coordinates are
(T=v-r, r, theta, phi_tilde). These integrals use complex conjugation;
they are not the analytically continued QBS bilinear product.
"""

from __future__ import annotations

import math

import numpy as np

from .models import KerrModeResult
from .relativity import _bound_kappa, angular_mode_values, build_radial_ode_evaluator


def ingoing_kerr_inverse(radius, theta, chi: float) -> np.ndarray:
    """Return g^{ab} in (T,r,theta,phi_tilde), including r=r_+."""
    r, angle = np.broadcast_arrays(np.asarray(radius), np.asarray(theta))
    sigma = r * r + chi * chi * np.cos(angle) ** 2
    inverse = np.zeros(r.shape + (4, 4))
    inverse[..., 0, 0] = -(sigma + 2 * r)
    inverse[..., 0, 1] = inverse[..., 1, 0] = 2 * r
    inverse[..., 1, 1] = r * r - 2 * r + chi * chi
    inverse[..., 1, 3] = inverse[..., 3, 1] = chi
    inverse[..., 2, 2] = 1
    inverse[..., 3, 3] = 1 / np.sin(angle) ** 2
    return inverse / sigma[..., None, None]


def scalar_currents(value, gradient, inverse, mass: float) -> np.ndarray:
    """Return the number, energy and axial angular currents, shape (...,3,4).

    j^a=Im(Psi* grad^a Psi), E^a=-T^a_T, J^a=T^a_phi.
    A real scalar's fast-time-averaged stress has half this normalization
    when it is defined as Re(Psi); fixing its physical mass removes it.
    """
    value, gradient = np.asarray(value), np.asarray(gradient)
    raised = np.einsum("...ab,...b->...a", inverse, gradient)
    trace_term = 0.5 * (
        np.einsum("...a,...a->...", gradient.conj(), raised).real
        + mass * mass * abs(value) ** 2
    )
    number = (value.conj()[..., None] * raised).imag
    energy = -(gradient[..., 0].conj()[..., None] * raised).real
    angular = (gradient[..., 3].conj()[..., None] * raised).real
    energy[..., 0] += trace_term
    angular[..., 3] -= trace_term
    return np.stack((number, energy, angular), axis=-2)


def axial_metric_source(value, gradient, inverse, mass: float, dphi_metric):
    """Return T^{ab} partial_phi(g_ab)/2, without the volume element.

    For an approximate scalar field with KG residual R there is also
    Re(R* partial_phi(Psi)) in the current divergence. This function does
    not assume that a truncated mode expansion has zero residual.
    """
    raised = np.einsum("...ab,...b->...a", inverse, gradient)
    trace_term = 0.5 * (
        np.einsum("...a,...a->...", np.conjugate(gradient), raised).real
        + mass * mass * abs(np.asarray(value)) ** 2
    )
    stress = (raised[..., :, None] * raised.conj()[..., None, :]).real
    stress -= trace_term[..., None, None] * inverse
    return 0.5 * np.einsum("...ab,...ab->...", stress, dphi_metric)


def ingoing_radial_provider(mode: KerrModeResult, outer_decay_lengths=28.0):
    """Reuse the two-sided radial ODE and remove the horizon phase analytically."""
    a, omega = mode.chi, mode.frequency_M
    root = math.sqrt(1 - a * a)
    rp, rm, gap = 1 + root, 1 - root, 2 * root
    kappa = _bound_kappa(mode.alpha, omega)
    power = (mode.alpha**2 - 2 * omega**2) / kappa - 1 + 2j * omega
    coefficients = mode.radial_coefficients
    derivative_coefficients = np.arange(1, len(coefficients)) * coefficients[1:]
    bl = build_radial_ode_evaluator(
        mode.alpha, a, mode.state, omega, mode.separation_constant, coefficients,
        outer_decay_lengths=outer_decay_lengths,
    )

    def evaluate(radius):
        r = np.asarray(radius, dtype=float)
        if np.any(r < rp) or np.any(r > bl.outer_radius):
            raise ValueError("ingoing radial evaluator called outside its domain")
        flat = r.reshape(-1)
        value = np.empty(flat.shape, dtype=complex)
        first = np.empty(flat.shape, dtype=complex)
        inner = flat <= bl.match_radius
        # Cancellation of all (r-r_+) powers precedes numerical evaluation.
        if np.any(inner):
            local = flat[inner]
            y = local - rm
            z = (local - rp) / y
            polynomial = np.polynomial.polynomial.polyval(z, coefficients)
            derivative = np.polynomial.polynomial.polyval(z, derivative_coefficients)
            prefactor = y**power * np.exp(kappa * local)
            value[inner] = prefactor * polynomial
            first[inner] = prefactor * (
                (power / y + kappa) * polynomial + derivative * gap / y**2
            )
        if np.any(~inner):
            local = flat[~inner]
            x, y = local - rp, local - rm
            time_shift = 2 * rp / gap * np.log(x) - 2 * rm / gap * np.log(y)
            phi_shift = a / gap * np.log(x / y)
            phase = np.exp(1j * omega * time_shift - 1j * mode.state.m * phi_shift)
            phase_derivative = 1j * (2 * local * omega - mode.state.m * a) / (x * y)
            radial, derivative, _ = bl(local)
            value[~inner] = phase * radial
            first[~inner] = phase * (derivative + phase_derivative * radial)
        return value.reshape(r.shape), first.reshape(r.shape)

    evaluate.r_plus = rp
    evaluate.match_radius = bl.match_radius
    evaluate.stitch_radius = bl.stitch_radius
    evaluate.outer_radius = bl.outer_radius
    evaluate.matching_residual = bl.matching_residual
    return evaluate


def mode_current_integrals(mode: KerrModeResult, *, radial_nodes=32,
                           angular_nodes=24, outer_decay_lengths=28.0) -> dict:
    """Integrate Q,E,J and both boundary fluxes of one stored Kerr mode.

    Charges and fluxes use the stored radial amplitude, with no independent
    unit-charge or unit-energy normalization. Positive horizon flux enters
    the hole; positive outer flux leaves the integration volume.
    """
    radial = ingoing_radial_provider(mode, outer_decay_lengths)
    u, angular_weights = np.polynomial.legendre.leggauss(angular_nodes)
    theta = np.arccos(u)
    angular, angular_first, _ = angular_mode_values(
        theta, mode.state, mode.separation_constant,
        mode.angular_l_values, mode.angular_coefficients,
    )

    def fields(radii):
        r = np.atleast_1d(radii)
        value, first = radial(r)
        psi = value[:, None] * angular[None, :]
        gradient = np.stack((
            -1j * mode.frequency_M * psi,
            first[:, None] * angular[None, :],
            value[:, None] * angular_first[None, :],
            1j * mode.state.m * psi,
        ), axis=-1)
        inverse = ingoing_kerr_inverse(r[:, None], theta[None, :], mode.chi)
        sigma = r[:, None]**2 + mode.chi**2 * u[None, :]**2
        currents = scalar_currents(psi, gradient, inverse, mode.alpha)
        return 2 * math.pi * np.einsum("u,ru,ruab->rab", angular_weights, sigma, currents)

    decay = -_bound_kappa(mode.alpha, mode.frequency_M).real
    # Split at the series/ODE and ODE/ODE joins; sample each decay length.
    breaks = [radial.r_plus, radial.r_plus + 0.1, radial.r_plus + 1,
              radial.match_radius, radial.stitch_radius, radial.outer_radius]
    breaks += list(radial.r_plus + np.arange(1, math.ceil(outer_decay_lengths)) / decay)
    breaks = sorted(set(x for x in breaks if radial.r_plus <= x <= radial.outer_radius))
    nodes, weights = np.polynomial.legendre.leggauss(radial_nodes)
    charges = np.zeros(3)
    for left, right in zip(breaks[:-1], breaks[1:], strict=True):
        radii = (left + right) / 2 + (right - left) / 2 * nodes
        charges += (right - left) / 2 * np.einsum("r,ra->a", weights, fields(radii)[..., 0])
    horizon = -fields(radial.r_plus)[0, :, 1]
    outer = fields(radial.outer_radius)[0, :, 1]
    omega, m = mode.frequency_M, mode.state.m
    horizon_amplitude = abs(radial(radial.r_plus)[0])**2
    angular_norm = 2 * math.pi * np.dot(angular_weights, abs(angular)**2)
    factor = float(2 * radial.r_plus * horizon_amplitude * angular_norm)
    omega_h = mode.chi / (2 * radial.r_plus)
    analytic = factor * np.array([
        omega.real - m * omega_h,
        abs(omega)**2 - m * omega_h * omega.real,
        m * (omega.real - m * omega_h),
    ])
    residual = 2 * omega.imag * charges + horizon + outer
    names = ("number", "energy", "angular_momentum")
    return {
        "state": mode.state.label, "alpha": mode.alpha, "chi": mode.chi,
        "frequency": [omega.real, omega.imag],
        "charges": dict(zip(names, charges.tolist(), strict=True)),
        "horizon_flux": dict(zip(names, horizon.tolist(), strict=True)),
        "outer_flux": dict(zip(names, outer.tolist(), strict=True)),
        "analytic_horizon_flux": dict(zip(names, analytic.tolist(), strict=True)),
        "balance_residual": dict(zip(names, residual.tolist(), strict=True)),
        "angular_norm": float(angular_norm),
        "radial_matching_residual": radial.matching_residual,
        "outer_radius": radial.outer_radius,
        "quadrature": {"radial_nodes": radial_nodes, "angular_nodes": angular_nodes,
                       "outer_decay_lengths": outer_decay_lengths},
    }
