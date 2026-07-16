from __future__ import annotations

import math
import unittest

import numpy as np
import sympy as sp
from scipy.integrate import dblquad
from scipy.special import sph_harm_y

from boson_ep.hydrogen import (
    primary_radial_integrals,
    primary_radial_integrals_quad,
    radial_normalization,
)
from boson_ep.symbolic import derive_primary_symbolics


class SymbolicBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.values = derive_primary_symbolics()

    def test_exact_symbolic_coefficients(self) -> None:
        expected_angular = -sp.sqrt(30) / (10 * sp.sqrt(sp.pi))
        self.assertEqual(self.values["normalization"], 1)
        self.assertEqual(self.values["radial_inner_infinite"], 30)
        self.assertEqual(sp.simplify(self.values["angular_gaunt"] - expected_angular), 0)
        self.assertEqual(self.values["angular_prefactor"], -sp.Rational(3, 10))
        self.assertEqual(self.values["total_far_tide_coefficient"], -9)

    def test_scipy_radial_normalization(self) -> None:
        value, _ = radial_normalization(2, 1)
        self.assertLess(abs(value - 1.0), 1.0e-10)

    def test_incomplete_gamma_matches_quad(self) -> None:
        for x_star in (0.4, 3.0, 30.0):
            exact = primary_radial_integrals(x_star)
            numerical = primary_radial_integrals_quad(x_star)
            for exact_piece, numerical_piece in zip(exact, numerical, strict=True):
                scale = max(abs(exact_piece), 1.0e-300)
                self.assertLess(abs(exact_piece - numerical_piece) / scale, 1.0e-10)

    def test_scipy_angular_integral_matches_sympy(self) -> None:
        def real_integrand(phi: float, theta: float) -> float:
            product = (
                -sph_harm_y(1, -1, theta, phi)
                * sph_harm_y(2, 2, theta, phi)
                * sph_harm_y(1, -1, theta, phi)
                * math.sin(theta)
            )
            return float(np.real(product))

        def imag_integrand(phi: float, theta: float) -> float:
            product = (
                -sph_harm_y(1, -1, theta, phi)
                * sph_harm_y(2, 2, theta, phi)
                * sph_harm_y(1, -1, theta, phi)
                * math.sin(theta)
            )
            return float(np.imag(product))

        real, _ = dblquad(real_integrand, 0.0, math.pi, 0.0, 2.0 * math.pi)
        imaginary, _ = dblquad(imag_integrand, 0.0, math.pi, 0.0, 2.0 * math.pi)
        expected = float(self.values["angular_gaunt"])
        self.assertLess(abs(real - expected) / abs(expected), 1.0e-10)
        self.assertLess(abs(imaginary), 1.0e-12)


if __name__ == "__main__":
    unittest.main()
