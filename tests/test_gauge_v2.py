import types
import unittest
from unittest.mock import patch

import numpy as np

from boson_ep.certification import (
    _direct_gauge_kernels,
    direct_ward_audit,
    operator_form_audit,
)
from boson_ep.models import (
    GaugeVectorSpec,
    RelativisticTideSettings,
    State,
    TransitionConfig,
)


def _mode(state, frequency=0.2, *, r_99=20.0, converged=True):
    return types.SimpleNamespace(
        alpha=0.1,
        chi=0.5,
        state=state,
        frequency_M=complex(frequency),
        r_99_M=r_99,
        converged=converged,
    )


class DirectWardTests(unittest.TestCase):
    def setUp(self):
        self.initial = State(2, 1, 1)
        self.final = State(2, 1, -1)
        self.config = TransitionConfig(
            0.1,
            self.initial,
            self.final,
            chi=0.5,
            settings=RelativisticTideSettings(radial_nodes=8, angular_nodes=8),
        )
        self.modes = (_mode(self.initial, 0.2), _mode(self.final, 0.1))

    def test_metric_triplet_uses_production_projection_and_same_domain(self):
        spec = GaugeVectorSpec("radial", 3.0, 7.0, 0.1, 0.25, "core")
        calls = []

        def projection(initial, final, omega, settings, metric, **kwargs):
            calls.append((metric, kwargs))
            if metric == "irg":
                return 2.0 + 0.0j
            matrix = metric(5.0, 1.0)
            self.assertEqual(matrix.shape, (4, 4))
            return complex(np.max(np.abs(matrix)))

        with patch(
            "boson_ep.certification.tidal_kernel_from_modes_M",
            side_effect=projection,
        ):
            physical, pure, shifted = _direct_gauge_kernels(
                *self.modes, 0.05, self.config.settings, spec
            )

        self.assertEqual(physical, 2.0)
        self.assertGreater(abs(pure), 0.0)
        self.assertGreater(abs(shifted), 0.0)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call[1]["radial_domain"] == (3.0, 7.0) for call in calls))
        self.assertEqual(calls[0][0], "irg")
        self.assertTrue(callable(calls[1][0]))
        self.assertTrue(callable(calls[2][0]))

    def test_direct_audit_reports_measured_residuals(self):
        specs = (
            GaugeVectorSpec("radial", 3.0, 7.0, 0.1, 1.0, "core"),
            GaugeVectorSpec("radial", 3.0, 7.0, 1.0, 1.0, "core"),
        )

        calls = {}

        def support_kernel(initial, final, omega, settings, spec, metric):
            if metric == "irg":
                return 0.2 + 0.0j
            key = complex(omega)
            calls[key] = calls.get(key, 0) + 1
            pure = 1.0e-10 + abs(omega - 0.05)
            if abs(omega - 0.05) < 1.0e-14 and calls[key] == 2:
                return 0.2 + pure
            return pure

        def green(initial, final, omega, settings, spec):
            return spec.relative_amplitude * (1.0e-10 + abs(omega - 0.05))

        with (
            patch(
                "boson_ep.certification._calibrated_spec",
                side_effect=lambda config, omega, spec: spec,
            ),
            patch(
                "boson_ep.certification._direct_support_kernel",
                side_effect=support_kernel,
            ),
            patch(
                "boson_ep.certification._pure_gauge_green_kernel",
                side_effect=green,
            ),
            patch(
                "boson_ep.certification.tidal_kernel_from_modes_M",
                return_value=1.0 + 0.0j,
            ),
        ):
            result = direct_ward_audit(self.config, self.modes, specs)

        self.assertEqual(result.status, "ok")
        self.assertGreater(result.maximum_pure_gauge_residual, 0.0)
        self.assertLess(result.maximum_pure_gauge_residual, 1.0e-8)
        self.assertGreater(result.off_resonance_ratio, 100.0)
        self.assertEqual(result.rows[0]["ward_evaluation"], "direct production delta Box")
        self.assertIn("green_cross_residual", result.rows[0])

    def test_nonzero_direct_pure_gauge_failure_is_not_hidden(self):
        spec = GaugeVectorSpec("radial", 3.0, 7.0, 1.0, 1.0, "core")
        with (
            patch(
                "boson_ep.certification._calibrated_spec",
                side_effect=lambda config, omega, value: value,
            ),
            patch(
                "boson_ep.certification._direct_support_kernel",
                side_effect=(0.2 + 0.0j, 0.02 + 0.0j, 0.22 + 0.0j),
            ),
            patch(
                "boson_ep.certification._pure_gauge_green_kernel",
                return_value=0.0j,
            ),
            patch(
                "boson_ep.certification.tidal_kernel_from_modes_M",
                return_value=1.0 + 0.0j,
            ),
        ):
            result = direct_ward_audit(
                self.config, self.modes, (spec,), include_off_resonance=False
            )
        self.assertEqual(result.status, "gauge_audit_failed")
        self.assertAlmostEqual(result.maximum_pure_gauge_residual, 0.02)
        self.assertGreater(result.maximum_invariance_residual, 0.0)


class OperatorAuditTests(unittest.TestCase):
    def setUp(self):
        initial = State(2, 1, 1)
        final = State(2, 1, -1)
        self.config = TransitionConfig(0.1, initial, final, chi=0.5)
        self.modes = (_mode(initial, 0.2), _mode(final, 0.1))

    def test_default_audit_uses_twelve_points_and_richardson(self):
        # Same continuum value with different O(h^2) stencil errors.
        connection = lambda mode, r, t, o, h: 2.0 + 3.0 * h**4
        divergence = lambda mode, r, t, o, h: 2.0 - 5.0 * h**4
        with (
            patch(
                "boson_ep.certification._delta_box_connection_bl",
                side_effect=lambda mode, r, t, o, gauge, h: connection(mode, r, t, o, h),
            ),
            patch(
                "boson_ep.certification._delta_box_divergence_bl",
                side_effect=lambda mode, r, t, o, gauge, h: divergence(mode, r, t, o, h),
            ),
        ):
            maximum, rows = operator_form_audit(
                self.config, resolved_modes=self.modes
            )
        self.assertEqual(len(rows), 12)
        self.assertLess(maximum, 1.0e-14)
        self.assertTrue(all("connection_stencil_residual" in row for row in rows))

    def test_continuum_disagreement_is_detected(self):
        with (
            patch(
                "boson_ep.certification._delta_box_connection_bl",
                side_effect=lambda mode, r, t, o, gauge, h: 2.0 + h**2,
            ),
            patch(
                "boson_ep.certification._delta_box_divergence_bl",
                side_effect=lambda mode, r, t, o, gauge, h: 2.1 - h**4,
            ),
        ):
            maximum, _ = operator_form_audit(
                self.config, ((5.0, 1.0),), self.modes
            )
        self.assertGreater(maximum, 0.04)


if __name__ == "__main__":
    unittest.main()
