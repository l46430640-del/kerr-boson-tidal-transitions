from __future__ import annotations

import math
import unittest

import numpy as np

from boson_ep.models import CFSettings, KerrModeSettings, State
from boson_ep.relativity import (
    angular_mode_residual,
    radial_continued_fraction_residual,
    radial_continued_fraction_residual_backward,
    solve_kerr_mode_adaptive,
    solve_quasibound_cf,
    solve_saturation_mode_cf,
    spheroidal_eigensystem,
    trace_saturation_branch_cf,
)


class ContinuedFractionV2Tests(unittest.TestCase):
    def test_backward_fraction_agrees_at_root(self) -> None:
        state = State(2, 1, 1)
        settings = CFSettings(truncation=400, angular_lmax=14)
        result = solve_quasibound_cf(0.42, 0.99, state, settings)
        self.assertTrue(result.converged)
        forward = radial_continued_fraction_residual(
            0.42, 0.99, state, result.frequency_M, settings
        )
        backward = radial_continued_fraction_residual_backward(
            0.42, 0.99, state, result.frequency_M, settings
        )
        self.assertLess(abs(forward - backward), 1.0e-10)
        self.assertIn("scale=", result.message)

    def test_all_previously_failed_saturation_points(self) -> None:
        settings = KerrModeSettings(
            truncation=400,
            series_terms=400,
            angular_lmax=14,
        )
        points = [
            (0.150, State(3, 2, 2)),
            (0.175, State(3, 2, 2)),
            (0.175, State(4, 3, 3)),
            (0.225, State(4, 3, 3)),
            (0.250, State(4, 3, 3)),
            (0.275, State(4, 3, 3)),
            (0.300, State(4, 3, 3)),
            (0.375, State(4, 3, 3)),
        ]
        for alpha, state in points:
            with self.subTest(alpha=alpha, state=state.label):
                result = solve_saturation_mode_cf(alpha, state, settings)
                self.assertTrue(result.converged)
                self.assertLess(result.cf_residual, 1.0e-10)
                self.assertLess(result.saturation_residual, 1.0e-12)

    def test_saturation_trace_is_direction_independent(self) -> None:
        settings = KerrModeSettings(
            truncation=400,
            series_terms=400,
            angular_lmax=14,
        )
        rows = trace_saturation_branch_cf(
            [0.15, 0.175], State(3, 2, 2), settings
        )
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertTrue(row.converged)
            difference = float(row.message.rsplit("=", 1)[1])
            self.assertLess(difference, 1.0e-10)

    def test_independent_angular_residual_detects_perturbation(self) -> None:
        alpha = 0.25
        chi = 0.5
        state = State(4, 3, 1)
        omega = complex(0.247, -1.0e-8)
        c_squared = chi * chi * (omega * omega - alpha * alpha)
        angular, l_values, coefficients = spheroidal_eigensystem(
            c_squared, state, 14
        )
        theta = np.linspace(0.02, math.pi - 0.02, 96)
        clean = angular_mode_residual(
            theta,
            state,
            alpha,
            chi,
            omega,
            angular,
            l_values,
            coefficients,
            14,
        )
        perturbed = coefficients.copy()
        perturbed[-1] += 1.0e-4
        dirty = angular_mode_residual(
            theta,
            state,
            alpha,
            chi,
            omega,
            angular,
            l_values,
            perturbed,
            14,
        )
        self.assertLess(clean, 1.0e-8)
        self.assertGreater(dirty, 1.0e-7)


class AdaptiveModeRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = KerrModeSettings(
            truncation=400,
            series_terms=400,
            angular_lmax=14,
            angular_nodes=72,
            horizon_cutoff=1.0e-4,
            counterterm_order=3,
        )

    def test_three_previously_failed_modes(self) -> None:
        points = [
            (0.100, 0.19791155474884012, State(3, 0, 0), 1200),
            (0.125, 0.24594549892517448, State(3, 0, 0), 1200),
            (0.450, 0.990615014283061, State(2, 1, -1), 800),
        ]
        for alpha, chi, state, maximum_n in points:
            with self.subTest(alpha=alpha, state=state.label):
                result = solve_kerr_mode_adaptive(
                    alpha, chi, state, self.settings
                )
                self.assertTrue(result.converged, result.message)
                self.assertIsNotNone(result.selected_truncation)
                self.assertLessEqual(result.selected_truncation, maximum_n)
                self.assertLess(result.cf_residual, 1.0e-10)
                self.assertLess(result.radial_residual, 1.0e-8)
                self.assertLess(result.angular_residual, 1.0e-8)
                self.assertIsNotNone(result.radial_ode_residual)
                self.assertLess(result.radial_ode_residual, 1.0e-6)
                self.assertTrue(result.convergence_history[-1]["passed"])

    def test_small_cutoff_norm_uses_log_horizon_quadrature(self) -> None:
        settings = KerrModeSettings(
            truncation=400,
            series_terms=400,
            angular_lmax=14,
            angular_nodes=72,
            horizon_cutoff=1.0e-5,
            counterterm_order=3,
        )
        alpha = 0.275
        chi = 0.839861984452313
        cases = (
            (State(2, 1, 1), complex(0.27218757483346534, -3.1180561296674623e-19)),
            (State(2, 1, -1), complex(0.2720772171127697, -1.2477696561685896e-5)),
        )
        for state, seed in cases:
            with self.subTest(state=state.label):
                result = solve_kerr_mode_adaptive(
                    alpha, chi, state, settings, seed=seed
                )
                self.assertTrue(result.converged, result.message)
                self.assertLess(abs(result.bilinear_norm - 1.0), 1.0e-5)


if __name__ == "__main__":
    unittest.main()
