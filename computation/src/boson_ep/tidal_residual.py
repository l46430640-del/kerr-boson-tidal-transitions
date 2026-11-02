"""Pointwise linearized Ricci tensor for a harmonic tidal perturbation.

Only the two non-Killing coordinates are differenced. Time and azimuthal
derivatives act on the supplied Fourier coefficient exactly.
"""

from functools import lru_cache

import numpy as np


def _connection(inverse, derivative):
    return 0.5 * np.einsum("cd,abd->cab", inverse,
                           derivative + derivative.swapaxes(0, 1) - derivative.transpose(1, 2, 0))


def _variation_data(background, perturbation, radius, theta, frequency, azimuthal_number,
                    step_scale):
    steps = {1: step_scale * 1e-3 * radius, 2: step_scale * 1e-3}
    factors = {0: -1j * frequency, 3: 1j * azimuthal_number}

    def partials(provider, r, angle, harmonic):
        value = provider(r, angle)
        result = np.zeros((4, *value.shape), dtype=complex)
        if harmonic:
            for coordinate, factor in factors.items():
                result[coordinate] = factor * value
        for coordinate, step in steps.items():
            def at(offset):
                return provider(r + offset if coordinate == 1 else r,
                                angle + offset if coordinate == 2 else angle)
            result[coordinate] = (at(-2 * step) - 8 * at(-step) + 8 * at(step) - at(2 * step)) / (12 * step)
        return result

    @lru_cache(maxsize=None)
    def geometry(r, angle):
        metric = np.asarray(background(r, angle), dtype=complex)
        inverse = np.linalg.inv(metric)
        gamma = _connection(inverse, partials(background, r, angle, False))
        return inverse, gamma

    @lru_cache(maxsize=None)
    def variation(r, angle):
        inverse, gamma = geometry(r, angle)
        h = np.asarray(perturbation(r, angle), dtype=complex)
        return (_connection(inverse, partials(perturbation, r, angle, True))
                - np.einsum("cd,de,eab->cab", inverse, h, gamma))

    _, gamma = geometry(radius, theta)
    delta_gamma = variation(radius, theta)
    derivative = partials(variation, radius, theta, True)
    return gamma, delta_gamma, derivative, geometry, partials


def linearized_ricci(background, perturbation, radius, theta, frequency, azimuthal_number,
                     step_scale=1.0):
    """Return delta R_ab for h_ab exp(-i frequency v+i m phi).

    Providers take (r, theta) and return covariant 4x4 coefficients, ordered
    (v,r,theta,phi). The background must be stationary and axisymmetric.
    A fourth-order stencil gives a convergence check without a large CAS.
    """
    gamma, delta_gamma, derivative, _, _ = _variation_data(
        background, perturbation, radius, theta, frequency, azimuthal_number, step_scale)
    ricci = np.einsum("ccab->ab", derivative) - np.einsum("bcac->ab", derivative)
    ricci += np.einsum("ccd,dab->ab", delta_gamma, gamma) + np.einsum("ccd,dab->ab", gamma, delta_gamma)
    ricci -= np.einsum("cbd,dac->ab", delta_gamma, gamma) + np.einsum("cbd,dac->ab", gamma, delta_gamma)
    return ricci


def linearized_riemann(background, perturbation, radius, theta, frequency, azimuthal_number,
                       step_scale=1.0):
    """Return the all-covariant delta R_abcd, including h_ae R^e_bcd."""
    gamma, delta_gamma, derivative, geometry, partials = _variation_data(
        background, perturbation, radius, theta, frequency, azimuthal_number, step_scale)
    background_derivative = partials(lambda r,t:geometry(r,t)[1], radius, theta, False)
    curvature = background_derivative.transpose(1,3,0,2)-background_derivative.transpose(1,3,2,0)
    curvature += np.einsum("ace,edb->abcd",gamma,gamma)-np.einsum("ade,ecb->abcd",gamma,gamma)
    variation = derivative.transpose(1,3,0,2)-derivative.transpose(1,3,2,0)
    variation += np.einsum("ace,edb->abcd",delta_gamma,gamma)+np.einsum("ace,edb->abcd",gamma,delta_gamma)
    variation -= np.einsum("ade,ecb->abcd",delta_gamma,gamma)+np.einsum("ade,ecb->abcd",gamma,delta_gamma)
    return (np.einsum("ae,ebcd->abcd",background(radius,theta),variation)
            +np.einsum("ae,ebcd->abcd",perturbation(radius,theta),curvature))


def component_norm(tensor, radius, theta):
    """Coordinate-rescaled component norm, not a scalar curvature invariant."""
    scale = np.array([1, 1, radius, radius * np.sin(theta)])
    return float(np.max(np.abs(tensor / np.outer(scale, scale))))
