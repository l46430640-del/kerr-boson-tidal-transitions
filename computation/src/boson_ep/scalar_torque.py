"""Scalar stress overlaps for a real, rotating quadrupolar tidal harmonic.

The metric coefficient multiplies exp(2i Omega T-2i phi_tilde) and its
complex conjugate. Scalar coefficients refer to the rotating two-mode
basis. No identification with a modal population-transfer rate is made.
"""

import math

import numpy as np

from .relativity import _bound_kappa, angular_mode_values
from .scalar_balance import ingoing_kerr_inverse, ingoing_radial_provider


def advanced_tide_to_ingoing(metric, radius, omega):
    """Transform a covariant harmonic coefficient from v to T=v-r."""
    jacobian = np.eye(4)
    jacobian[0, 1] = 1
    return np.exp(2j * omega * radius) * (jacobian.T @ metric @ jacobian)


def torque_operator(overlap, temporal_generator):
    """Return K such that c^dagger K c is the torque delivered to the scalar.

    The full time derivative of field harmonic j is -i (W c)_j, where W
    includes its carrier and envelope derivative. The four overlaps are
    the time/time, time/space, space/time and space/space action terms.
    """
    a, b, c, d = np.asarray(overlap, dtype=complex)
    generator = np.asarray(temporal_generator, dtype=complex)
    initial, final = np.eye(2, dtype=complex)
    di, df = -1j * generator[0], -1j * generator[1]
    product = (a * np.outer(df.conj(), di) + b * np.outer(df.conj(), initial)
               + c * np.outer(final, di) + d * np.outer(final, initial))
    return -1j * product + 1j * product.conj().T


def torque_entry_bounds(overlap_bounds, frequencies, forward_bound, reverse_bound):
    """Triangle bounds on the entries of the physical torque operator."""
    a, b, c, d = np.asarray(overlap_bounds, dtype=float)
    wi, wf = np.abs(frequencies)
    if min(a, b, c, d, forward_bound, reverse_bound) < 0:
        raise ValueError("Coefficient bounds must be nonnegative")
    offdiagonal = a*(wi*wf + forward_bound*reverse_bound) + b*wf + c*wi + d
    return np.array([[2*forward_bound*(a*wi+b), offdiagonal],
                     [offdiagonal, 2*reverse_bound*(a*wf+c)]])


def metric_angular_operator(overlap, temporal_generator, azimuthal_numbers, *, inward_flux=False):
    """First metric variation of a charge or inward boundary flux.

    For the charge use the volume overlaps with the first tensor index T.
    For a boundary use surface overlaps with the first tensor index r.
    """
    a, b, c = np.asarray(overlap, dtype=complex)[:3]
    generator = np.asarray(temporal_generator, dtype=complex)
    initial, final = np.eye(2, dtype=complex)
    mi, mf = azimuthal_numbers
    di, df = -1j * generator[0], -1j * generator[1]
    product = -.5 * (-1j * mf * a * np.outer(final, di)
                     + 1j * mi * a * np.outer(df.conj(), initial)
                     + (-1j * mf * b + 1j * mi * c) * np.outer(final, initial))
    result = product + product.conj().T
    return -result if inward_flux else result


def background_angular_operators(weights, radial_current, horizon_amplitudes,
                                 azimuthal_numbers, chi, temporal_generator):
    """Angular charge and inward horizon flux, including envelope derivatives.

    These use the Kerr metric. Perturbations of the metric and boundary are
    separate from these operators and cannot be inferred from their balance.
    """
    generator = np.asarray(temporal_generator, dtype=complex)
    rp = 1 + math.sqrt(1 - chi * chi)
    charge, flux = np.zeros((2, 2), complex), np.zeros((2, 2), complex)
    for j, m in enumerate(azimuthal_numbers):
        unit = np.eye(2)[j]
        energy_row = np.outer(unit, generator[j])
        energy_row = (energy_row + energy_row.conj().T) / 2
        diagonal = np.outer(unit, unit)
        charge += m * (weights[j] * energy_row + radial_current[j] * diagonal)
        flux += m * horizon_amplitudes[j] * (2 * rp * energy_row - chi * m * diagonal)
    return charge, flux


def background_energy_operators(weights, isolated_energies, frequencies, horizon_amplitudes,
                                azimuthal_numbers, chi, temporal_generator):
    """Kerr energy and inward energy flux with the full temporal derivative."""
    generator = np.asarray(temporal_generator, dtype=complex)
    rp = 1 + math.sqrt(1 - chi * chi)
    charge, flux = np.zeros((2, 2), complex), np.zeros((2, 2), complex)
    for j, m in enumerate(azimuthal_numbers):
        unit = np.eye(2)[j]
        quadratic = np.outer(generator[j].conj(), generator[j])
        row = np.outer(unit, generator[j])
        real_row = (row + row.conj().T) / 2
        spatial = isolated_energies[j] - .5 * weights[j] * abs(frequencies[j])**2
        charge += .5 * weights[j] * quadratic + spatial * np.outer(unit, unit)
        flux += horizon_amplitudes[j] * (2 * rp * quadratic - chi * m * real_row)
    return charge, flux


def background_angular_outer_flux(amplitudes, radial_flux, azimuthal_numbers,
                                  chi, radius, temporal_generator):
    """Outward Kerr angular flux across a fixed outer radial boundary."""
    generator = np.asarray(temporal_generator, dtype=complex)
    flux = np.zeros((2, 2), complex)
    for j, m in enumerate(azimuthal_numbers):
        unit = np.eye(2)[j]
        row = np.outer(unit, generator[j])
        real_row = (row + row.conj().T) / 2
        flux += m*(-2*radius*amplitudes[j]*real_row
                   + (radial_flux[j]+chi*m*amplitudes[j])*np.outer(unit,unit))
    return flux


def stress_overlaps(modes, metric_providers, *, radial_nodes=20, angular_nodes=16,
                    outer_decay_lengths=24.0):
    """Integrate four stress overlaps for each supplied ingoing metric coefficient.

    The two modes must have m_i-m_f=2. Providers already include any metric
    coordinate transformation, but may omit the binary amplitude epsilon.
    """
    initial, final = modes
    if initial.state.m - final.state.m != 2:
        raise ValueError("The implemented tide requires m_i-m_f=2")
    if initial.alpha != final.alpha or initial.chi != final.chi:
        raise ValueError("Modes must share the same scalar mass and Kerr background")
    providers = [ingoing_radial_provider(m, outer_decay_lengths) for m in modes]
    rp, outer = providers[0].r_plus, min(p.outer_radius for p in providers)
    nodes, weights = np.polynomial.legendre.leggauss(radial_nodes)
    u, uw = np.polynomial.legendre.leggauss(angular_nodes)
    theta = np.arccos(u)
    angular = [angular_mode_values(theta, m.state, m.separation_constant,
                                   m.angular_l_values, m.angular_coefficients)[:2] for m in modes]
    decay = min(-_bound_kappa(m.alpha, m.frequency_M).real for m in modes)
    breaks = [rp, rp + .01, rp + .1, rp + 1, outer]
    breaks += [p.match_radius for p in providers] + [p.stitch_radius for p in providers]
    breaks += list(rp + np.arange(1, math.ceil(outer_decay_lengths)) / decay)
    breaks = sorted(set(x for x in breaks if rp <= x <= outer))
    overlaps = {key: np.zeros(4, complex) for key in metric_providers}
    time_weights, radial_current = np.zeros(2), np.zeros(2)
    for left, right in zip(breaks[:-1], breaks[1:], strict=True):
        radii = (left + right) / 2 + (right - left) / 2 * nodes
        radial = [p(radii) for p in providers]
        for ir, (r, rw) in enumerate(zip(radii, weights, strict=True)):
            inverse = ingoing_kerr_inverse(r, theta, initial.chi)
            sigma = r*r + initial.chi**2 * u*u
            measure = math.pi * (right - left) * rw * uw * sigma
            fields, spatial = [], []
            for j, mode in enumerate(modes):
                value = radial[j][0][ir] * angular[j][0]
                first = radial[j][1][ir] * angular[j][0]
                derivatives = np.stack((first, radial[j][0][ir] * angular[j][1],
                                        1j * mode.state.m * value), axis=-1)
                fields.append(value)
                spatial.append(derivatives)
                time_weights[j] += np.dot(measure, -inverse[:, 0, 0] * abs(value)**2)
                radial_current[j] += np.dot(measure, inverse[:, 0, 1] * (value.conj() * first).imag)
            fi, ff = fields
            si, sf = spatial
            for key, provider in metric_providers.items():
                metric = np.array([provider(float(r), float(t)) for t in theta])
                trace = np.einsum("uab,uab->u", inverse, metric)
                tensor = np.einsum("uab,ubc,ucd->uad", inverse, metric, inverse)
                tensor -= .5 * trace[:, None, None] * inverse
                integrands = [
                    ff.conj() * fi * tensor[:, 0, 0],
                    ff.conj() * np.einsum("ua,ua->u", tensor[:, 0, 1:], si),
                    fi * np.einsum("ua,ua->u", sf.conj(), tensor[:, 1:, 0]),
                    np.einsum("ua,uab,ub->u", sf.conj(), tensor[:, 1:, 1:], si)
                    - .5 * trace * initial.alpha**2 * ff.conj() * fi,
                ]
                overlaps[key] += np.array([np.dot(measure, x) for x in integrands])
    horizon = [float(2 * math.pi * abs(p(rp)[0])**2 * np.dot(uw, abs(a[0])**2))
               for p, a in zip(providers, angular, strict=True)]
    boundary_overlaps = {}
    for offset in (1e-3, 5e-4, None):
        r = rp + offset if offset is not None else outer
        inverse = ingoing_kerr_inverse(r, theta, initial.chi)
        sigma = r*r + initial.chi**2 * u*u
        measure = 2 * math.pi * uw * sigma
        fields, spatial = [], []
        for mode, provider, angles in zip(modes, providers, angular, strict=True):
            value, first = provider(r)
            psi = value * angles[0]
            fields.append(psi)
            spatial.append(np.stack((first * angles[0], value * angles[1],
                                     1j * mode.state.m * psi), axis=-1))
        fi, ff = fields
        si, sf = spatial
        key_offset = str(offset) if offset is not None else "outer"
        boundary_overlaps[key_offset] = {}
        for key, provider in metric_providers.items():
            metric = np.array([provider(float(r), float(t)) for t in theta])
            trace = np.einsum("uab,uab->u", inverse, metric)
            tensor = np.einsum("uab,ubc,ucd->uad", inverse, metric, inverse)
            tensor -= .5 * trace[:, None, None] * inverse
            values = [ff.conj() * fi * tensor[:, 1, 0],
                      ff.conj() * np.einsum("ua,ua->u", tensor[:, 1, 1:], si),
                      fi * np.einsum("ua,ua->u", sf.conj(), tensor[:, 1:, 1])]
            boundary_overlaps[key_offset][key] = np.array([np.dot(measure, x) for x in values])
    outer_values = [p(outer) for p in providers]
    angular_norms = [2*math.pi*np.dot(uw,abs(a[0])**2) for a in angular]
    outer_amplitudes = [float(abs(v[0])**2*n) for v,n in zip(outer_values,angular_norms,strict=True)]
    delta = outer*outer-2*outer+initial.chi**2
    outer_radial_flux = [float(delta*(v[0].conj()*v[1]).imag*n)
                         for v,n in zip(outer_values,angular_norms,strict=True)]
    return {"overlaps": overlaps, "time_weights": time_weights,
            "radial_current": radial_current, "horizon_amplitudes": horizon,
            "boundary_overlaps": boundary_overlaps,
            "outer_amplitudes": outer_amplitudes, "outer_radial_flux": outer_radial_flux,
            "outer_radius": outer,
            "quadrature": {"radial_nodes": radial_nodes, "angular_nodes": angular_nodes,
                           "outer_decay_lengths": outer_decay_lengths}}
