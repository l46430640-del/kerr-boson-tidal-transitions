from __future__ import annotations

import unittest

from boson_ep.models import PRIMARY_FINAL, PRIMARY_INITIAL
from boson_ep.spectrum import (
    detweiler_coefficient,
    gamma_detweiler_M,
    horizon_radius_M,
    horizon_omega_M,
    omega_real_M,
    saturation_spin_numeric,
)
from boson_ep.tides import energy_gap_M, resonance_frequency_M


class SpectrumTests(unittest.TestCase):
    def test_primary_frequency_gap(self) -> None:
        for alpha, chi in ((0.03, 0.3), (0.1, 0.9), (0.3, 0.99)):
            expected_gap = -chi * alpha**6 / 6.0
            expected_resonance = chi * alpha**6 / 12.0
            self.assertAlmostEqual(energy_gap_M(alpha, chi), expected_gap, delta=2.0e-16)
            self.assertAlmostEqual(
                resonance_frequency_M(alpha, chi),
                expected_resonance,
                delta=1.0e-16,
            )

    def test_corrected_primary_detweiler_coefficient(self) -> None:
        alpha = 0.1
        chi = 0.91
        r_plus = horizon_radius_M(chi)
        for state in (PRIMARY_INITIAL, PRIMARY_FINAL):
            expected = (
                1.0
                - chi**2
                + (chi * state.m - 2.0 * r_plus * alpha) ** 2
            ) / 48.0
            self.assertAlmostEqual(
                detweiler_coefficient(alpha, chi, state), expected, places=15
            )

    def test_growth_and_absorption_signs(self) -> None:
        self.assertGreater(gamma_detweiler_M(0.1, 0.99, PRIMARY_INITIAL), 0.0)
        self.assertLess(gamma_detweiler_M(0.1, 0.99, PRIMARY_FINAL), 0.0)

    def test_numeric_saturation_condition(self) -> None:
        for alpha in (0.03, 0.1, 0.2, 0.3):
            chi = saturation_spin_numeric(alpha)
            residual = horizon_omega_M(chi) - omega_real_M(alpha, chi, PRIMARY_INITIAL)
            self.assertLess(abs(residual), 1.0e-13)


if __name__ == "__main__":
    unittest.main()
