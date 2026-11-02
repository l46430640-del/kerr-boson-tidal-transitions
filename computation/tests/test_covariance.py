import types
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp

from boson_ep.certification import _pure_gauge_green_kernel
from boson_ep.projection import transform_amplitude_generator


class GaugeCovarianceTests(unittest.TestCase):
    def test_harmonic_basis_shift_gives_full_complex_mismatch(self):
        t = sp.symbols("t", real=True)
        delta, overlap = sp.symbols("delta B", complex=True)
        shift = overlap * sp.exp(-sp.I * delta * t)
        self.assertEqual(sp.simplify(sp.I * sp.diff(shift, t) - delta * shift), 0)

    def test_real_resonance_retains_width_difference(self):
        initial = types.SimpleNamespace(frequency_M=0.3 + 0j)
        final = types.SimpleNamespace(frequency_M=0.2 - 0.02j)
        overlap = 0.4 + 0.1j
        with patch("boson_ep.certification.gauge_amplitude_overlap", return_value=overlap):
            real = _pure_gauge_green_kernel(initial, final, 0.05, None, None)
            pole = _pure_gauge_green_kernel(initial, final, (initial.frequency_M - final.frequency_M) / 2,
                                             None, None)
        self.assertAlmostEqual(real, 0.02j * overlap)
        self.assertEqual(pole, 0)

    def test_generator_connection_has_the_required_sign(self):
        generator = np.array([[0.1j, 0.3], [0.2j, -0.4j]])
        basis = np.array([[1, 0.2j], [-0.1, 1]])
        rate = np.array([[0, 0.07], [0.03j, 0]])
        transformed = transform_amplitude_generator(generator, basis, rate)
        np.testing.assert_allclose(transformed @ basis, basis @ generator + 1j * rate, atol=1e-14)

    def test_dissipative_propagator_and_endpoint_population_transform_together(self):
        delta = 0.1 + 0.17j

        def generator(t):
            return np.array([[0, 0.18], [0.18, 0.3 * t - 0.2j]], dtype=complex)

        def maps(t):
            off = np.array([[0, (0.2 - 0.1j) * np.exp(1j * delta * t)],
                            [(-0.4 + 0.1j) * np.exp(-1j * delta * t), 0]])
            rate = off * np.array([[0, 1j * delta], [-1j * delta, 0]])
            return np.eye(2) + 0.1 * off, 0.1 * rate

        def evolve(transformed):
            def rhs(t, flat):
                h = generator(t)
                if transformed:
                    h = transform_amplitude_generator(h, *maps(t))
                return (-1j * h @ flat.reshape(2, 2)).ravel()
            solution = solve_ivp(rhs, (-4, 4), np.eye(2, dtype=complex).ravel(),
                                 method="DOP853", rtol=2e-12, atol=1e-13)
            self.assertTrue(solution.success)
            return solution.y[:, -1].reshape(2, 2)

        original, shifted = evolve(False), evolve(True)
        start, end = maps(-4)[0], maps(4)[0]
        np.testing.assert_allclose(shifted @ start, end @ original, rtol=1e-10, atol=1e-11)
        state = original[:, 0]
        shifted_state = shifted @ start[:, 0]
        projector = np.diag([1, 0])
        inverse = np.linalg.inv(end)
        shifted_projector = inverse.conj().T @ projector @ inverse
        original_population = np.vdot(state, projector @ state)
        shifted_population = np.vdot(shifted_state, shifted_projector @ shifted_state)
        self.assertAlmostEqual(original_population, shifted_population, places=10)
        self.assertGreater(abs(abs(shifted_state[0])**2 - original_population), 1e-4)

    def test_invalid_basis_map_is_rejected(self):
        with self.assertRaises(ValueError):
            transform_amplitude_generator(np.eye(2), np.ones(2), np.eye(2))
        with self.assertRaises(ValueError):
            transform_amplitude_generator(np.eye(2), np.eye(2), np.full((2, 2), np.nan))
        with self.assertRaises(np.linalg.LinAlgError):
            transform_amplitude_generator(np.eye(2), np.zeros((2, 2)), np.eye(2))


if __name__ == "__main__":
    unittest.main()
