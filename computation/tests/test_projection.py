import cmath
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import sympy as sp

from boson_ep.models import KerrModeSettings, RelativisticTideSettings, State
from boson_ep.projection import schrodinger_product_factor, source_to_hamiltonian
from boson_ep.relativistic_tides import _bl_tidal_kernel, hydrogenic_newtonian_kernel_M


class SourceProjectionTests(unittest.TestCase):
    def test_flat_single_mode_reproduces_signed_frequency_shift(self):
        omega, constant, norm_u = 0.7, 0.03, 1.4
        norm = 2j * omega * norm_u
        source = constant * norm_u
        self.assertAlmostEqual(source_to_hamiltonian(source, norm), -constant / (2 * omega))

    def test_constant_source_agrees_with_exact_dispersion_derivative(self):
        epsilon, omega, constant = sp.symbols("epsilon omega C", positive=True)
        exact = sp.sqrt(omega**2 - epsilon * constant)
        first_order = sp.diff(exact, epsilon).subs(epsilon, 0)
        self.assertEqual(sp.simplify(first_order + constant / (2 * omega)), 0)

    def test_background_current_identity_includes_the_source_sign(self):
        t = sp.symbols("t", real=True)
        omega = sp.symbols("omega", positive=True)
        field = sp.Function("phi")(t)
        left = sp.exp(sp.I * omega * t)
        current = field * sp.diff(left, t) - left * sp.diff(field, t)
        d_field = -sp.diff(field, t, 2) - omega**2 * field
        self.assertEqual(sp.simplify(sp.diff(current, t) - left * d_field), 0)

    def test_phase_and_scale_transform_with_the_final_norm(self):
        source, norm = 0.3 + 0.8j, 1.2 - 0.1j
        scale_i, scale_f = 1.7 * cmath.exp(0.4j), 0.6 * cmath.exp(-0.7j)
        transformed = source_to_hamiltonian(source * scale_f * scale_i, norm * scale_f**2)
        self.assertAlmostEqual(transformed, scale_i / scale_f * source_to_hamiltonian(source, norm))
        rephased = source_to_hamiltonian(source * cmath.exp(0.4j - 0.7j), norm * cmath.exp(-1.4j))
        self.assertAlmostEqual(abs(rephased), abs(source_to_hamiltonian(source, norm)))

    def test_unit_bilinear_norm_supplies_the_frequency_factor(self):
        omega = 0.7
        amplitude = 1 / cmath.sqrt(2j * omega)
        self.assertAlmostEqual(source_to_hamiltonian(0.03 * amplitude**2), -0.03 / (2 * omega))
        self.assertAlmostEqual(schrodinger_product_factor(omega, omega) * amplitude**2, 1)

    def test_hydrogenic_quadrupole_has_the_signed_nine_over_alpha_cubed_limit(self):
        alpha = 0.2
        radial_moment = 30 / alpha**4
        angular_moment = -2 / (5 * math.pi)
        potential_matrix = -3 / 8 * 2 * math.pi * radial_moment * angular_moment
        source = -2 * alpha**2 * potential_matrix / (2j * alpha)
        expected = 9 / alpha**3
        self.assertAlmostEqual(source_to_hamiltonian(source).real, expected, places=9)
        self.assertAlmostEqual(hydrogenic_newtonian_kernel_M(alpha, State(2, 1, 1), State(2, 1, -1)).real,
                               expected, places=9)

    def test_production_quadrature_applies_the_conversion(self):
        settings = RelativisticTideSettings(
            radial_nodes=18, angular_nodes=12,
            mode=KerrModeSettings(outer_decay_lengths=28),
        )
        mode = SimpleNamespace(chi=0.0, alpha=0.2, frequency_M=0.19 + 0j)
        # A constant source and final field make the BL volume integral elementary.
        fields = lambda radius, theta: (1.0 + 0j, np.zeros(6, dtype=complex))
        with patch("boson_ep.relativistic_tides._delta_box_divergence_bl", return_value=2.0 + 0j):
            actual = _bl_tidal_kernel(
                mode, mode, 0.0, settings, "irg", radial_domain=(3.0, 4.0),
                radial_breaks=(3.5,), initial_field_provider=fields, final_field_provider=fields,
            )
        integral = 2 * 4 * math.pi * (4**3 - 3**3) / 3
        self.assertAlmostEqual(actual, -1j * integral, places=10)

    def test_invalid_norm_or_source_is_rejected(self):
        for source, norm in ((1, 0), (float("nan"), 1), (1, float("inf"))):
            with self.subTest(source=source, norm=norm):
                with self.assertRaises(ValueError):
                    source_to_hamiltonian(source, norm)


if __name__ == "__main__":
    unittest.main()
