import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from boson_ep.models import (
    KernelErrorBudgetResult,
    KerrModeResult,
    RelativisticTideSettings,
    State,
    TransitionConfig,
    TransitionKernelResult,
)
from boson_ep.relativistic_tides import (
    compute_kernel_error_budget,
    compute_transition_kernel,
    evaluate_transition_at_q,
)


def _load_scanner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "scan_relativistic_atlas_v2.py"
    spec = importlib.util.spec_from_file_location("atlas_v2_scanner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mode(state, frequency, r99=11.0):
    return KerrModeResult(
        alpha=0.1, chi=0.9, state=state, frequency_M=frequency,
        separation_constant=2.0 + 0j, angular_l_values=np.asarray([state.l]),
        angular_coefficients=np.asarray([1.0 + 0j]),
        radial_coefficients=np.asarray([1.0 + 0j]), bilinear_norm=1.0 + 0j,
        r_99_M=r99, cf_residual=1.0e-12, radial_residual=1.0e-10,
        angular_residual=1.0e-10, converged=True, message="ok",
        selected_truncation=400,
    )


class AtlasV2PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.initial = State(2, 1, 1)
        self.final = State(2, 1, -1)
        self.modes = (
            _mode(self.initial, 0.099 + 1.0e-10j),
            _mode(self.final, 0.097 + 0j),
        )
        self.config = TransitionConfig(
            0.1, self.initial, self.final, chi=0.9,
            settings=RelativisticTideSettings(radial_nodes=56, angular_nodes=72),
        )

    def test_kernel_is_q_independent_and_q_validity_crosses(self):
        with (
            patch("boson_ep.relativistic_tides.hydrogenic_newtonian_kernel_M", return_value=9.0 + 0j),
            patch("boson_ep.relativistic_tides._kerr_newtonian_kernel", return_value=8.0 + 0j),
            patch("boson_ep.relativistic_tides.tidal_kernel_from_modes_M", return_value=8.1 + 0j),
        ):
            kernel = compute_transition_kernel(self.config, self.modes)
        self.assertEqual(kernel.kernel_status, "ok")
        low = evaluate_transition_at_q(kernel, 1.0e-4)
        high = evaluate_transition_at_q(kernel, 1.0)
        self.assertFalse(low.tidal_valid)
        self.assertTrue(high.tidal_valid)
        self.assertEqual(kernel.kernel_status, "ok")
        self.assertNotEqual(low.eta_covariant_M, high.eta_covariant_M)

    def test_all_three_physics_layers_and_measured_error_band(self):
        kernel = TransitionKernelResult(
            "point", 0.1, 0.9, self.initial, self.final, "ok", None,
            1.0e-3, 5.0, 9.0 + 0j, 8.0 + 0j, 7.0 + 0j,
        )
        budget = KernelErrorBudgetResult(
            "point", "ok", (), 0.1, 0.1, "radial_nodes", 0.7, True
        )
        result = evaluate_transition_at_q(kernel, 0.1, budget)
        self.assertLess(result.z_covariant, result.z_hydrogenic)
        self.assertAlmostEqual(result.power_ratio_covariant_to_hydrogenic, 49 / 81)
        self.assertLess(result.error_lower, result.depletion_covariant)
        self.assertGreater(result.error_upper, result.depletion_covariant)

    def test_real_oat_budget_has_no_alpha_proxy(self):
        central = TransitionKernelResult(
            "point", 0.1, 0.9, self.initial, self.final, "ok", None,
            1.0e-3, 11.0, 9.0 + 0j, 8.0 + 0j, 10.0 + 0j,
        )

        def variant_kernel(_i, _f, _o, settings, _metric, **kwargs):
            delta = 0.0
            delta += (settings.radial_nodes - 56) * 1.0e-4
            delta += (settings.angular_nodes - 72) * 1.0e-5
            delta += (settings.mode.angular_lmax - 14) * 1.0e-4
            delta += (settings.mode.truncation - 250) * 1.0e-6
            delta += (settings.mode.counterterm_order - 2) * 1.0e-4
            delta += (settings.mode.outer_decay_lengths - 28.0) * 1.0e-5
            if kwargs.get("radial_strategy") == "quad":
                delta += 2.0e-4
            return 10.0 + delta + 0j

        with (
            patch("boson_ep.relativistic_tides.compute_transition_kernel", return_value=central),
            patch("boson_ep.relativistic_tides.solve_kerr_mode", side_effect=lambda a, c, s, st: _mode(s, 0.099 + 0j)),
            patch("boson_ep.relativistic_tides.tidal_kernel_from_modes_M", side_effect=variant_kernel),
            patch("boson_ep.certification.operator_form_audit", return_value=(0.0, [])),
        ):
            budget = compute_kernel_error_budget(self.config, self.modes)
        self.assertTrue(budget.complete)
        self.assertEqual(len(budget.sources), 9)
        self.assertNotEqual(budget.worst_source, "alpha_proxy")
        self.assertGreaterEqual(budget.systematic_error_abs, 0.0)


class AtlasV2CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scanner = _load_scanner()

    def test_strict_json_replaces_nonfinite_and_rejects_nan_encoding(self):
        payload = {"nan": math.nan, "inf": math.inf, "array": np.asarray([1.0])}
        encoded = self.scanner.canonical_json(payload)
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)
        self.assertEqual(json.loads(encoded)["nan"], None)

    def test_staged_cache_checks_all_fingerprints_and_payload_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            cache = self.scanner.StagedCache(path, "code", "config", "deps")
            cache.store("kernels", "point", "ok", {"value": 3})
            self.assertEqual(cache.load("kernels", "point"), {"value": 3})
            other = self.scanner.StagedCache(path, "changed", "config", "deps")
            self.assertIsNone(other.load("kernels", "point"))
            file_path = cache._path("kernels", "point")
            envelope = json.loads(file_path.read_text(encoding="utf-8"))
            envelope["payload"]["value"] = 4
            file_path.write_text(json.dumps(envelope), encoding="utf-8")
            self.assertIsNone(cache.load("kernels", "point"))

    def test_formal_output_cardinalities_are_frozen(self):
        self.assertEqual(
            self.scanner.EXPECTED_COUNTS,
            {"saturation": 45, "modes": 120, "kernels": 75,
             "audits": 75, "error_budgets": 75, "phenomenology": 2475},
        )
        self.assertEqual(
            self.scanner.PILOT_IDS,
            {"a0.100000_21+1_21-1", "a0.250000_32+2_30+0",
             "a0.450000_43+3_41+1"},
        )


if __name__ == "__main__":
    unittest.main()
