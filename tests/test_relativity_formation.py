from __future__ import annotations

import math
import unittest

from boson_ep import assess_formation_history, find_ep_roots
from boson_ep.formation import FormationConfig
from boson_ep.models import CFSettings, PRIMARY_INITIAL
from boson_ep.relativity import solve_quasibound_cf


class RelativityAndFormationTests(unittest.TestCase):
    def test_dolan_maximum_growth_benchmark(self) -> None:
        result = solve_quasibound_cf(
            0.42,
            0.99,
            PRIMARY_INITIAL,
            CFSettings(truncation=400),
        )
        self.assertTrue(result.converged)
        self.assertLess(abs(result.frequency_M.imag / 1.5e-7 - 1.0), 0.02)

    def test_formation_veto_and_mass_dependence(self) -> None:
        q_value = find_ep_roots(0.1, 0.99, (1.0e-4, 1.0))[0].q
        low_mass = assess_formation_history(
            FormationConfig(0.1, 0.99, q_value, 10.0)
        )
        high_mass = assess_formation_history(
            FormationConfig(0.1, 0.99, q_value, 100.0)
        )
        self.assertEqual(low_mass.status, "formation_veto")
        self.assertGreater(low_mass.pre_resonance_efolds, low_mass.log_required_occupancy)
        self.assertAlmostEqual(
            high_mass.log_required_occupancy - low_mass.log_required_occupancy,
            2.0 * math.log(10.0),
            places=11,
        )
        self.assertAlmostEqual(
            high_mass.pre_resonance_efolds,
            low_mass.pre_resonance_efolds,
            places=8,
        )


if __name__ == "__main__":
    unittest.main()

