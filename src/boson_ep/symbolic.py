"""SymPy derivation of the primary transition coefficients."""

from __future__ import annotations

import sympy as sp
from sympy.physics.wigner import gaunt


def derive_primary_symbolics() -> dict[str, sp.Expr]:
    x = sp.symbols("x", positive=True, real=True)
    radial = x * sp.exp(-x / 2) / sp.sqrt(24)
    normalization = sp.integrate(x**2 * radial**2, (x, 0, sp.oo))
    radial_inner = sp.integrate(x**4 * radial**2, (x, 0, sp.oo))
    angular = (-1) ** 1 * gaunt(1, 2, 1, -1, 2, -1)
    y22_equator = sp.sqrt(sp.Rational(15, 32) / sp.pi)
    angular_prefactor = sp.simplify(4 * sp.pi / 5 * y22_equator * angular)
    total_coefficient = sp.simplify(angular_prefactor * radial_inner)
    return {
        "radial_21": radial,
        "normalization": sp.simplify(normalization),
        "radial_inner_infinite": sp.simplify(radial_inner),
        "angular_gaunt": sp.simplify(angular),
        "y22_equator": sp.simplify(y22_equator),
        "angular_prefactor": angular_prefactor,
        "total_far_tide_coefficient": total_coefficient,
    }

