import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import sph_harm_y

from boson_ep.models import ContourKernelSettings, State
from boson_ep.kernel_validation import _angular_bvp, _chirp, _separation


class KernelValidationPhysicsTests(unittest.TestCase):
    def test_point_companion_multipoles_reconstruct_direct_potential(self):
        rho, theta, phi = 0.2, 1.1, -0.7
        direct = 1.0 / math.sqrt(
            1.0 + rho * rho - 2.0 * rho * math.sin(theta) * math.cos(phi)
        )
        reconstructed = 0.0j
        for ell in range(21):
            coefficient = sum(
                sph_harm_y(ell, m, theta, phi)
                * np.conjugate(sph_harm_y(ell, m, math.pi / 2.0, 0.0))
                for m in range(-ell, ell + 1)
            )
            reconstructed += 4.0 * math.pi / (2 * ell + 1) * rho**ell * coefficient
        self.assertLess(abs(reconstructed - direct) / direct, 1.0e-10)

    def test_odd_electric_m2_coefficients_vanish_without_parity_branch(self):
        quadrupole = abs(sph_harm_y(2, 2, math.pi / 2.0, 0.0))
        for ell in (3, 5, 7, 9):
            ratio = abs(sph_harm_y(ell, 2, math.pi / 2.0, 0.0)) / quadrupole
            self.assertLess(ratio, 1.0e-12)

    def test_regular_angular_bvp_has_independent_spherical_limit(self):
        state = State(2, 1, 1)
        eigenvalue, _, residual = _angular_bvp(
            0.1, 0.0, state, 0.099 + 0.0j, ContourKernelSettings()
        )
        self.assertLess(abs(eigenvalue - 2.0), 1.0e-10)
        self.assertLess(residual, 1.0e-8)

    def test_1pn_orbit_is_distinct_and_finite(self):
        omega, q = 1.0e-3, 0.2
        newtonian_b = _separation(omega, q, "newtonian")
        one_pn_b = _separation(omega, q, "1pn")
        newtonian_chirp = _chirp(omega, q, "newtonian")
        one_pn_chirp = _chirp(omega, q, "1pn")
        self.assertTrue(0.0 < one_pn_b < newtonian_b)
        self.assertTrue(0.0 < one_pn_chirp < newtonian_chirp)


class KernelValidationCacheTests(unittest.TestCase):
    def test_cache_rejects_any_fingerprint_change_and_json_is_strict(self):
        from scripts.validate_kernel_results import KernelCache, _atomic_json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = KernelCache(root / "cache", "code-a", "config-a", "deps-a")
            cache.store("stage", "point", "ok", {"value": 1.0})
            self.assertEqual(cache.load("stage", "point"), {"value": 1.0})
            self.assertIsNone(
                KernelCache(root / "cache", "code-b", "config-a", "deps-a")
                .load("stage", "point")
            )
            self.assertIsNone(
                KernelCache(root / "cache", "code-a", "config-b", "deps-a")
                .load("stage", "point")
            )
            self.assertIsNone(
                KernelCache(root / "cache", "code-a", "config-a", "deps-b")
                .load("stage", "point")
            )
            output = root / "strict.json"
            _atomic_json(output, {"failed": math.inf, "nan": math.nan})
            self.assertEqual(json.loads(output.read_text()), {"failed": None, "nan": None})


if __name__ == "__main__":
    unittest.main()
