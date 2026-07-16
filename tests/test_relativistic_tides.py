import math
import unittest

from boson_ep import (
    schwarzschild_irg_tidal_kernel_M,
    schwarzschild_rw_tidal_kernel_M,
)
from boson_ep.models import KerrModeSettings, RelativisticTideSettings, State
from boson_ep.relativity import saturation_spin_cf, spheroidal_eigensystem
from boson_ep.relativistic_tides import (
    _density_tensor_bl,
    hydrogenic_newtonian_kernel_M,
)
from boson_ep.tidal_metric import (
    irg_algebraic_residuals,
    irg_metric_coefficient,
    schwarzschild_rw_metric_coefficient,
)


class RelativisticTideTests(unittest.TestCase):
    def test_finite_frequency_rw_kernel_path_is_rejected(self):
        with self.assertRaises(ValueError):
            _density_tensor_bl(10.0, 1.0, 0.0, 1.0e-3, "rw")

    def test_spherical_limit_and_bilinear_normalization(self):
        state = State(4, 3, 1)
        eigenvalue, l_values, coefficients = spheroidal_eigensystem(
            0.0j, state, 9
        )
        self.assertAlmostEqual(eigenvalue.real, 12.0, places=13)
        self.assertLess(abs(sum(coefficients * coefficients) - 1.0), 1.0e-13)
        self.assertIn(3, l_values)

    def test_hydrogenic_211_quadrupole_coefficient(self):
        alpha = 0.1
        kernel = hydrogenic_newtonian_kernel_M(
            alpha, State(2, 1, 1), State(2, 1, -1)
        )
        self.assertLess(abs(kernel / (9.0 / alpha**3) - 1.0), 2.0e-12)

    def test_forbidden_delta_m_is_zero(self):
        kernel = hydrogenic_newtonian_kernel_M(
            0.1, State(2, 1, 1), State(2, 1, 0)
        )
        self.assertEqual(kernel, 0.0j)

    def test_irg_algebraic_conditions(self):
        trace, tetrad = irg_algebraic_residuals(8.0, 1.0, 0.4, 1.0e-3)
        self.assertLess(trace, 1.0e-12)
        self.assertLess(tetrad, 1.0e-12)

    def test_metric_harmonic_normalization(self):
        theta = math.pi / 2.0
        irg = irg_metric_coefficient(10.0, theta, 0.0, 0.0)
        rw = schwarzschild_rw_metric_coefficient(10.0, theta)
        expected = 0.75 * (10.0 - 2.0) ** 2
        self.assertAlmostEqual(irg[0, 0].real, expected, places=12)
        self.assertAlmostEqual(rw[0, 0].real, expected, places=12)

    def test_schwarzschild_irg_rw_kernel_agreement(self):
        settings = RelativisticTideSettings(
            mode=KerrModeSettings(
                truncation=150, series_terms=150, angular_nodes=30,
                horizon_cutoff=3.0e-5, counterterm_order=4,
            ),
            angular_nodes=6,
            radial_nodes=80,
            horizon_cutoffs=(3.0e-5,),
        )
        initial = State(2, 1, 1)
        final = State(2, 1, -1)
        irg = schwarzschild_irg_tidal_kernel_M(
            0.1, initial, final, settings
        )
        rw = schwarzschild_rw_tidal_kernel_M(
            0.1, initial, final, settings
        )
        self.assertLess(abs(irg / rw - 1.0), 1.0e-8)

    def test_high_l_saturation_branch(self):
        chi = saturation_spin_cf(
            0.45,
            State(4, 3, 3),
            KerrModeSettings(truncation=80, angular_lmax=12, series_terms=80),
        )
        self.assertGreater(chi, 0.5)
        self.assertLess(chi, 0.6)


if __name__ == "__main__":
    unittest.main()
