import unittest
import numpy as np
from boson_ep.scalar_balance import axial_metric_source, ingoing_kerr_inverse, ingoing_radial_provider, mode_current_integrals, scalar_currents
from boson_ep.relativity import radial_mode_values
from boson_ep.tidal_metric import kerr_metric_advanced
from calculations import load_mode_pair

class ScalarCurrentTests(unittest.TestCase):

    def test_inverse_is_the_spacelike_time_transform_of_advanced_kerr(self):
        transform = np.eye(4)
        transform[0, 1] = 1
        for a in (0.0, 0.7):
            for r in (1 + np.sqrt(1 - a * a), 4.0, 120.0):
                metric = transform.T @ kerr_metric_advanced(r, 1.1, a) @ transform
                inverse = ingoing_kerr_inverse(r, 1.1, a)
                np.testing.assert_allclose(inverse @ metric, np.eye(4), atol=2e-15)
                self.assertLess(inverse[0, 0], 0)

    def test_flat_on_shell_plane_wave(self):
        mass, k = (0.3, 0.4)
        omega = np.hypot(mass, k)
        gradient = np.array([-1j * omega, 1j * k, 0, 0])
        currents = scalar_currents(1 + 0j, gradient, np.diag([-1.0, 1, 1, 1]), mass)
        np.testing.assert_allclose(currents[0], [omega, k, 0, 0])
        np.testing.assert_allclose(currents[1], [omega ** 2, omega * k, 0, 0])

    def test_metric_source_factor_and_scalar_rescaling(self):
        inverse = np.diag([-1.0, 1, 1, 1])
        gradient = np.array([-0.5j, 0.4j, 0, 0])
        dphi = np.zeros((4, 4))
        dphi[0, 0] = 0.8
        source = axial_metric_source(1, gradient, inverse, 0.3, dphi)
        self.assertAlmostEqual(source, 0.5 * 0.5 ** 2 * 0.8)
        amplitude = 2 - 3j
        self.assertAlmostEqual(axial_metric_source(amplitude, amplitude * gradient, inverse, 0.3, dphi), abs(amplitude) ** 2 * source)
        self.assertEqual(axial_metric_source(1, gradient, inverse, 0.3, 0 * dphi), 0)

    def test_newtonian_potential_has_the_classical_torque_sign(self):
        mass, potential_derivative = (0.2, 0.03)
        inverse = np.diag([-1.0, 1, 1, 1])
        gradient = np.array([-1j * mass, 0, 0, 0])
        dphi = np.zeros((4, 4))
        dphi[0, 0] = -2 * potential_derivative
        number_density = scalar_currents(1, gradient, inverse, mass)[0, 0]
        self.assertAlmostEqual(axial_metric_source(1, gradient, inverse, mass, dphi), -mass * number_density * potential_derivative)

    def test_complex_frequency_horizon_energy_keeps_width_squared(self):
        a, angle, m, omega = (0.7, 1.1, 1, 0.2 - 0.03j)
        rp = 1 + np.sqrt(1 - a * a)
        sigma = rp * rp + a * a * np.cos(angle) ** 2
        psi = 0.8 + 0.6j
        gradient = np.array([-1j * omega * psi, 0.3 + 0.1j, -0.2j, 1j * m * psi])
        currents = scalar_currents(psi, gradient, ingoing_kerr_inverse(rp, angle, a), 0.3)
        flux = -sigma * currents[:, 1]
        expected = np.array([2 * rp * omega.real - a * m, 2 * rp * abs(omega) ** 2 - a * m * omega.real, m * (2 * rp * omega.real - a * m)]) * abs(psi) ** 2
        np.testing.assert_allclose(flux, expected, rtol=2e-14, atol=2e-16)
        self.assertAlmostEqual(flux[1] - omega.real * flux[0], 2 * rp * omega.imag ** 2 * abs(psi) ** 2)

class KerrModeCurrentTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.modes = [m for p in ('a0.200000_21+1_21-1', 'a0.400000_32+2_32+0') for m in load_mode_pair(p)[1]]
        cls.records = [mode_current_integrals(m) for m in cls.modes]

    def test_regular_series_matches_existing_advanced_radial_mode(self):
        for mode in self.modes:
            radial = ingoing_radial_provider(mode)
            radii = radial.r_plus + np.array([1e-05, 0.1, 1.0])
            advanced, first, _ = radial_mode_values(radii, mode.alpha, mode.chi, mode.state, mode.frequency_M, mode.radial_coefficients, advanced=True)
            phase = np.exp(-1j * mode.frequency_M * radii)
            regular, regular_first = radial(radii)
            np.testing.assert_allclose(regular, phase * advanced, rtol=2e-14)
            np.testing.assert_allclose(regular_first, phase * (first - 1j * mode.frequency_M * advanced), rtol=2e-13)
            self.assertTrue(np.all(np.isfinite(radial(radial.r_plus))))
            with self.assertRaises(ValueError):
                radial(radial.r_plus - 0.0001)

    def test_mode_current_has_physical_m_charge_relation(self):
        for mode, record in zip(self.modes, self.records, strict=True):
            charge = record['charges']
            self.assertGreater(charge['energy'], 0)
            self.assertAlmostEqual(charge['angular_momentum'], mode.state.m * charge['number'], places=14)
            for field in ('horizon_flux', 'outer_flux'):
                self.assertAlmostEqual(record[field]['angular_momentum'], mode.state.m * record[field]['number'], places=20)

    def test_damped_modes_close_energy_number_and_angular_balance(self):
        for record in self.records[1::2]:
            for name, residual in record['balance_residual'].items():
                flux = record['horizon_flux'][name]
                if flux:
                    self.assertLess(abs(residual / flux), 2e-08)
                    self.assertLess(abs(record['outer_flux'][name] / flux), 1e-15)
                    self.assertAlmostEqual(flux / record['analytic_horizon_flux'][name], 1, places=14)
                else:
                    self.assertEqual(residual, 0)

    def test_saturation_uses_absolute_not_zero_flux_relative_error(self):
        for record in self.records[::2]:
            for name, residual in record['balance_residual'].items():
                self.assertLess(abs(residual / record['charges'][name]), 2e-16)

    def test_zero_m_mode_absorbs_energy_but_no_axial_angular_momentum(self):
        record = self.records[-1]
        self.assertGreater(record['horizon_flux']['energy'], 0)
        self.assertEqual(record['horizon_flux']['angular_momentum'], 0)
        self.assertLess(self.records[1]['horizon_flux']['angular_momentum'], 0)
if __name__ == '__main__':
    unittest.main()
