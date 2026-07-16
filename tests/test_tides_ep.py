from __future__ import annotations

import unittest

from boson_ep.ep import analytic_far_tide_q_ep, find_ep_roots
from boson_ep.spectrum import saturation_spin_numeric
from boson_ep.tides import (
    orbital_radius_M,
    resonance_frequency_M,
    tidal_eta_M,
    tidal_eta_far_M,
)


class TidalAndEPTests(unittest.TestCase):
    def test_full_tide_reaches_far_limit(self) -> None:
        alpha = 0.1
        chi = 0.99
        q = 0.5
        omega = resonance_frequency_M(alpha, chi)
        x_star = alpha**2 * orbital_radius_M(omega, q)
        self.assertGreater(x_star, 30.0)
        full = tidal_eta_M(alpha, chi, q)
        far = tidal_eta_far_M(alpha, chi, q)
        self.assertLess(abs(full - far) / far, 1.0e-8)

    def test_physical_ep_root_and_residuals(self) -> None:
        roots = find_ep_roots(0.1, 0.99)
        physical = [root for root in roots if root.status == "physical_root"]
        self.assertEqual(len(physical), 1)
        root = physical[0]
        self.assertIsNotNone(root.q)
        self.assertLessEqual(float(root.q), 1.0)
        self.assertLess(float(root.residual), 1.0e-10)
        self.assertLess(float(root.discriminant_normalized), 1.0e-10)
        self.assertGreater(float(root.radius_over_cloud), 10.0)

    def test_numeric_root_matches_far_analytic_root(self) -> None:
        alpha = 0.1
        chi = 0.99
        root = find_ep_roots(alpha, chi)[0]
        analytic = analytic_far_tide_q_ep(alpha, chi)
        self.assertIsNotNone(root.q)
        self.assertIsNotNone(analytic)
        self.assertLess(abs(float(root.q) - float(analytic)) / float(analytic), 1.0e-6)

    def test_numeric_saturation_is_marginal_not_superradiant(self) -> None:
        for alpha in (0.05, 0.1, 0.2, 0.3):
            chi = saturation_spin_numeric(alpha)
            result = find_ep_roots(alpha, chi)[0]
            self.assertEqual(result.status, "not_superradiant")
            self.assertIsNone(result.q)


if __name__ == "__main__":
    unittest.main()
