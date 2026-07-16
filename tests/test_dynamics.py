from __future__ import annotations

import unittest

from boson_ep import evolve_transition
from boson_ep.models import EvolutionConfig, PRIMARY_FINAL, PRIMARY_INITIAL
from boson_ep.spectrum import gamma_detweiler_M, omega_real_M
from boson_ep.tides import resonance_frequency_M, tidal_eta_at_omega_M, tidal_eta_M


class DynamicsTests(unittest.TestCase):
    def test_eta_at_resonance_matches_baseline(self) -> None:
        alpha = 0.1
        chi = 0.99
        q_value = 0.6
        self.assertAlmostEqual(
            tidal_eta_at_omega_M(
                alpha, chi, q_value, resonance_frequency_M(alpha, chi)
            )
            / tidal_eta_M(alpha, chi, q_value),
            1.0,
            places=13,
        )

    def test_zero_coupling_and_common_growth_shift(self) -> None:
        alpha = 0.3
        chi = 0.99
        q_value = 1.0
        base = EvolutionConfig(
            alpha,
            chi,
            q_value,
            edge_factor=0.1,
            coupling_scale=0.0,
            store_trajectory=False,
        )
        first = evolve_transition(base)
        shift = 2.0e-6j
        initial = complex(
            omega_real_M(alpha, chi, PRIMARY_INITIAL),
            gamma_detweiler_M(alpha, chi, PRIMARY_INITIAL),
        )
        final = complex(
            omega_real_M(alpha, chi, PRIMARY_FINAL),
            gamma_detweiler_M(alpha, chi, PRIMARY_FINAL),
        )
        shifted = evolve_transition(
            EvolutionConfig(
                **{
                    **base.__dict__,
                    "omega_initial_M": initial + shift,
                    "omega_final_M": final + shift,
                }
            )
        )
        self.assertAlmostEqual(first.survival_nh, 1.0, places=12)
        self.assertAlmostEqual(shifted.survival_nh, first.survival_nh, places=12)

    def test_exponential_and_dop853_agree_and_balance(self) -> None:
        result = evolve_transition(
            EvolutionConfig(
                0.3,
                0.99,
                1.0,
                edge_factor=0.1,
                crosscheck=True,
                store_trajectory=False,
            )
        )
        self.assertIsNotNone(result.solver_error)
        self.assertLess(result.solver_error, 1.0e-6)
        self.assertLess(result.probability_balance_residual, 1.0e-8)


if __name__ == "__main__":
    unittest.main()

