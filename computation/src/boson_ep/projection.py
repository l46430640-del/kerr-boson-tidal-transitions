"""Source projection for the convention B(phi, phi) = 2 i omega int(u*u)."""

from __future__ import annotations

import cmath
import math

import numpy as np


def source_to_hamiltonian(source: complex, norm: complex = 1.0) -> complex:
    """Convert int(sqrt(-g) J(phi_f) delta(D) phi_i) to i dc_f/dt.

    The conserved-current identity gives norm * dc_f/dt = -source.
    The source and the final-mode bilinear norm must use the same modes,
    integration slice and radial regularization. No complex conjugation is used.
    """
    norm = complex(norm)
    source = complex(source)
    if norm == 0 or not all(math.isfinite(value) for value in
                            (norm.real, norm.imag, source.real, source.imag)):
        raise ValueError("Source and nonzero bilinear norm must be finite")
    return -1j * source / norm


def schrodinger_product_factor(omega_i: complex, omega_f: complex) -> complex:
    """Convert two unit-B positive-frequency modes to Schrodinger factors."""
    return 2j * cmath.sqrt(omega_i * omega_f)


def transform_amplitude_generator(generator, basis_map, basis_rate):
    """Generator for c'=G c: H'=(G H+i dG/dt) G^{-1}.

    This finite-dimensional identity does not assume Hermiticity or real mode
    frequencies. Endpoint amplitudes must be transformed by the same G.
    """
    generator, basis_map, basis_rate = (
        np.asarray(value, dtype=complex) for value in (generator, basis_map, basis_rate)
    )
    if (generator.ndim != 2 or generator.shape[0] != generator.shape[1]
            or basis_map.shape != generator.shape or basis_rate.shape != generator.shape):
        raise ValueError("Generator, basis map and its derivative must be equally sized square matrices")
    if not all(np.isfinite(value).all() for value in (generator, basis_map, basis_rate)):
        raise ValueError("Amplitude transformation matrices must be finite")
    return np.linalg.solve(basis_map.T, (basis_map @ generator + 1j * basis_rate).T).T
