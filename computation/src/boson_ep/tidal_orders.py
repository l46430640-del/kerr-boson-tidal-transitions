"""Distinct expansion scales for a circular equatorial tidal field."""

import math


def tidal_expansion_scales(alpha, principal_n, radius_M, omega_M, q):
    """Return order estimates, not error bounds, at a specified cloud radius.

    The electric-quadrupole PN coefficients use the black-hole-frame frequency
    and harmonic separation of Poisson and Corrigan, arXiv:1804.01848v2, Sec. IX.
    All orbital quantities on the right are needed only at Newtonian order.
    """
    if not all(math.isfinite(x) and x > 0 for x in (alpha, principal_n, radius_M, omega_M, q)):
        raise ValueError("Expansion scales require finite positive inputs")
    separation = (1 + q)**(1 / 3) * omega_M**(-2 / 3)
    spatial = radius_M / separation
    orbital_pn = (1 + q) / separation
    mass_fraction = 1 / (1 + q)
    symmetric_ratio = q / (1 + q)**2
    cloud_velocity = alpha / principal_n
    return {
        "separation_M": separation,
        "r_cloud_over_b": spatial,
        "M_over_r_cloud": 1 / radius_M,
        "Omega_r_cloud": omega_M * radius_M,
        "quadrupole_frequency_r_cloud": 2 * omega_M * radius_M,
        "orbital_pn": orbital_pn,
        "electric_l4_order": spatial**2,
        "magnetic_l3_order": math.sqrt(orbital_pn) * cloud_velocity * spatial,
        "cloud_velocity_estimate": cloud_velocity,
        "tidal_nonlinearity_order": q / (1 + q) * (omega_M * radius_M)**2,
        "electric_22_pn_at_fixed_b": -0.5 * (4 - mass_fraction) * orbital_pn,
        "electric_22_pn_at_fixed_local_frequency": (1 + symmetric_ratio + 0.5 * mass_fraction) * orbital_pn,
        "local_frequency_pn_at_fixed_b": -0.5 * (3 + symmetric_ratio) * orbital_pn,
    }
