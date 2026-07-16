import math
import unittest

import numpy as np

from boson_ep import (
    GaugeMetricConfig,
    GaugeVectorSpec,
    compact_support_gauge_metric,
)
from boson_ep.certification import (
    compare_schwarzschild_benchmark,
    fit_weak_field_limit,
)


class CertificationTests(unittest.TestCase):
    def test_compact_gauge_metric_support_and_linearity(self):
        config = GaugeMetricConfig(5.0, 1.0, 0.3, 1.0e-3)
        small = compact_support_gauge_metric(
            config, GaugeVectorSpec("polar", 3.0, 7.0, 0.1, 0.25)
        )
        large = compact_support_gauge_metric(
            config, GaugeVectorSpec("polar", 3.0, 7.0, 1.0, 2.5)
        )
        outside = compact_support_gauge_metric(
            GaugeMetricConfig(7.0, 1.0, 0.3, 1.0e-3),
            GaugeVectorSpec("polar", 3.0, 7.0, 1.0),
        )
        self.assertTrue(small.active)
        self.assertLess(
            np.max(np.abs(large.metric_coefficient - 10.0 * small.metric_coefficient)),
            1.0e-12,
        )
        self.assertFalse(outside.active)
        self.assertEqual(float(np.max(np.abs(outside.metric_coefficient))), 0.0)

    def test_weak_field_fit_accepts_quartic_baseline(self):
        rows = []
        for alpha in (0.03, 0.04, 0.05, 0.06, 0.075, 0.10):
            rows.append(
                {
                    "alpha": alpha,
                    "kernel_ratio": 1.0 - 2.85 * alpha**2 + 0.4 * alpha**4,
                    "gauge_residual": 1.0e-12,
                }
            )
        result = fit_weak_field_limit(rows)
        self.assertEqual(result.status, "ok")
        self.assertLess(abs(result.c0 - 1.0), 1.0e-12)
        self.assertLess(abs(result.c2 + 2.85), 1.0e-10)

    def test_benchmark_uses_digitization_envelope(self):
        reference = [
            {"alpha": 0.1, "relative_correction": 0.017, "sigma_digitization": 0.002}
        ]
        observed = [
            {"alpha": 0.1, "relative_correction": 0.026, "gauge_residual": 1.0e-12}
        ]
        result = compare_schwarzschild_benchmark(observed, reference)
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.rows[0]["passed"])


if __name__ == "__main__":
    unittest.main()
