import math
import unittest

import numpy as np
import sympy as sp

from boson_ep.dynamic_tide import IngoingHertzTide, angular_mode, angular_values, schwarzschild_frequency_derivative
from boson_ep.tidal_metric import irg_metric_coefficient, kerr_metric_advanced, kinnersley_l_advanced
from boson_ep.tidal_residual import component_norm, linearized_ricci


class DynamicTideTests(unittest.TestCase):
    def test_static_limit_reproduces_original_metric(self):
        for chi in (0, 0.6, 0.95):
            tide = IngoingHertzTide(chi, 0, 100)
            for radius, theta in ((3, 0.7), (12, 1.1), (50, 2.3)):
                np.testing.assert_allclose(tide.metric(radius, theta), irg_metric_coefficient(radius, theta, chi, 0), rtol=2e-13, atol=1e-10)

    def test_angular_equation_and_first_frequency_order(self):
        for c in (-0.02, 0.02):
            eigenvalue, coefficients = angular_mode(c)
            for theta in (0.4, 1.1, 2.4):
                value, first, second = angular_values(theta, coefficients)
                u = math.cos(theta)
                potential = c*c*u*u + 4*c*u - (-2-2*u)**2/(1-u*u) - 2 + eigenvalue
                residual = second + u/math.sin(theta)*first + potential*value
                self.assertLess(abs(residual), 2e-11)
        step = 1e-5
        plus, _ = angular_mode(step)
        minus, _ = angular_mode(-step)
        self.assertAlmostEqual((plus-minus)/(2*step), 8/3, places=8)
        theta = 1.1
        _, p = angular_mode(step)
        _, m = angular_mode(-step)
        derivative = (angular_values(theta, p)[0] - angular_values(theta, m)[0]) / (2*step)
        expected = (1-math.cos(theta))**2 * (4/9 + 2*math.cos(theta)/3)
        self.assertAlmostEqual(derivative, expected, places=9)

    def test_schwarzschild_frequency_derivative(self):
        step = 1e-7
        plus = IngoingHertzTide(0, -step/2, 100, rtol=1e-13)
        minus = IngoingHertzTide(0, step/2, 100, rtol=1e-13)
        for radius in (2.01, 5, 50):
            difference = (plus.radial_values(radius)[0] - minus.radial_values(radius)[0]) / (2*step)
            self.assertLess(abs(difference-schwarzschild_frequency_derivative(radius)), 2e-8)

    def test_exact_schwarzschild_linear_radial_equation(self):
        r = sp.symbols("r", positive=True)
        f = sp.I*(5*r/3 + 2*sp.log(r/2) - 2/r - 2/(3*r*r) - sp.Rational(13, 6))
        self.assertEqual(sp.simplify((r*r-2*r)*sp.diff(f, r, 2) + 6*(r-1)*sp.diff(f, r) - 10*sp.I*r), 0)
        self.assertEqual(sp.simplify(f.subs(r, 2)), 0)

    def test_frequency_solution_passes_full_vacuum_equation(self):
        for chi in (0, 0.6):
            tide = IngoingHertzTide(chi, 1e-5, 110)
            for radius in (5, 50):
                theta = 1.1
                residual = linearized_ricci(lambda r,t:kerr_metric_advanced(r,t,chi),
                                             tide.electric_metric, radius, theta, tide.frequency, -2)
                scaled = radius**2 * component_norm(residual, radius, theta) / component_norm(tide.electric_metric(radius, theta), radius, theta)
                self.assertLess(scaled, 2e-7)

    def test_irg_constraints_and_electric_reflection(self):
        tide = IngoingHertzTide(0.8, 2e-5, 100)
        reflection = np.diag([1, 1, -1, 1])
        for radius, theta in ((4, 0.7), (50, 1.1)):
            metric = tide.electric_metric(radius, theta)
            scale = np.max(np.abs(metric))
            inverse = np.linalg.inv(kerr_metric_advanced(radius, theta, tide.chi))
            self.assertLess(abs(np.einsum("ab,ab->", inverse, metric))/scale, 1e-13)
            self.assertLess(np.max(np.abs(kinnersley_l_advanced(radius,tide.chi)@metric))/scale, 1e-13)
            np.testing.assert_allclose(metric, reflection@tide.electric_metric(radius, math.pi-theta)@reflection, rtol=1e-13, atol=1e-9)

    def test_stationary_electric_equatorial_metric(self):
        tide = IngoingHertzTide(0,0,100)
        for r in (5,50):
            # Independent Schwarzschild specialization of the explicit IRG metric.
            reference = np.zeros((4,4),dtype=complex)
            reference[0,0] = 3*(r-2)**2/4
            reference[0,1] = reference[1,0] = -3*r*(r-2)/2
            reference[1,1] = 3*r*r
            reference[0,3] = reference[3,0] = 1j*r*r*(r-2)/2
            reference[1,3] = reference[3,1] = -1j*r**3
            reference[2,2] = r*r*(r*r-2)/4
            reference[3,3] = -reference[2,2]
            np.testing.assert_allclose(tide.electric_metric(r,math.pi/2),reference,rtol=2e-14,atol=1e-9)

    def test_boundary_and_angular_refinements(self):
        base = IngoingHertzTide(0.9, 1e-4, 100)
        refined = IngoingHertzTide(0.9, 1e-4, 100, ell_max=10, horizon_offset=5e-5, series_order=10, rtol=3e-13, max_step=0.5)
        for radius in (3, 50):
            np.testing.assert_allclose(base.radial_values(radius), refined.radial_values(radius), rtol=1e-8, atol=2e-12)
            np.testing.assert_allclose(base.electric_metric(radius,1.1), refined.electric_metric(radius,1.1), rtol=2e-10, atol=1e-9)

    def test_invalid_inputs(self):
        for chi, omega, outer in ((1,0,100), (0,float("nan"),100), (0,0,1)):
            with self.assertRaises(ValueError):
                IngoingHertzTide(chi, omega, outer)
        with self.assertRaises(ValueError):
            IngoingHertzTide(0,0,10).metric(11,1.1)


if __name__ == "__main__":
    unittest.main()
