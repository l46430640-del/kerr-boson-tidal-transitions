from __future__ import annotations

import unittest

from boson_ep.ep import find_ep_roots
from boson_ep.timescales import compute_timescales


class TimescaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = find_ep_roots(0.1, 0.99)[0]
        if root.q is None:
            raise AssertionError("reference EP root is required")
        cls.q = root.q

    def test_mass_scaling_is_strictly_linear(self) -> None:
        light = compute_timescales(0.1, 0.99, self.q, 10.0)
        heavy = compute_timescales(0.1, 0.99, self.q, 100.0)
        for name in light.times_M:
            self.assertEqual(light.times_M[name], heavy.times_M[name])
            self.assertAlmostEqual(
                heavy.times_seconds[name] / light.times_seconds[name], 10.0, places=14
            )

    def test_reference_signs_and_hierarchy(self) -> None:
        result = compute_timescales(0.1, 0.99, self.q, 10.0)
        self.assertGreater(result.gamma_grow_M, 0.0)
        self.assertLess(result.gamma_absorb_M, 0.0)
        self.assertEqual(
            result.hierarchy,
            tuple(sorted(result.times_M, key=result.times_M.get)),
        )
        self.assertGreater(result.landau_zener_z, 0.0)


if __name__ == "__main__":
    unittest.main()
