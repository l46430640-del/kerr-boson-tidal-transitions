from types import SimpleNamespace
import math
import unittest

import numpy as np
from scipy.special import sph_harm_y

from boson_ep.models import State
from boson_ep.multipoles import spectral_angular_integral


def mode(l, m, coeffs):
    return SimpleNamespace(state=State(l+1, l, m),
        angular_l_values=np.arange(l, l+2*len(coeffs), 2),
        angular_coefficients=np.asarray(coeffs, complex))


class MultipoleAngularTests(unittest.TestCase):
    def test_spherical_selection_zeros_are_exact(self):
        first, second = mode(1, 1, [1]), mode(1, -1, [1])
        for ell in (3, 4, 6, 8, 10):
            self.assertEqual(spectral_angular_integral(first, second, ell), 0j)
        expected = -math.sqrt(15/(2*math.pi))/(10*math.pi)
        self.assertAlmostEqual(spectral_angular_integral(first, second, 2).real, expected, places=15)

    def test_bilinear_signs_and_complex_coefficients(self):
        u, weights = np.polynomial.legendre.leggauss(96)
        theta = np.arccos(u)
        for mi, mf in ((1, -1), (2, 0), (3, 1), (-1, 1), (0, 0)):
            first, second = mode(abs(mi), mi, [1, .02+.03j]), mode(abs(mf), mf, [1j, -.01j])
            for ell in range(max(2, abs(mi-mf)), 8):
                si = sum(c*sph_harm_y(int(l), mi, theta, 0) for l, c in
                         zip(first.angular_l_values, first.angular_coefficients))
                sf = sum(c*sph_harm_y(int(l), mf, theta, 0) for l, c in
                         zip(second.angular_l_values, second.angular_coefficients))
                direct = np.sum(weights*sf*si*sph_harm_y(ell, mi-mf, theta, 0))
                self.assertLess(abs(direct-spectral_angular_integral(first, second, ell)), 2e-15)

    def test_tiny_higher_moment_scales_without_roundoff_floor(self):
        # ell=4 needs one l=3 coefficient; there is no order-one contribution.
        initial = mode(1, 1, [1, 1e-25j])
        final = mode(1, -1, [1, 0])
        tiny = spectral_angular_integral(initial, final, 4)
        unit = spectral_angular_integral(mode(1, 1, [1, 1j]), final, 4)
        self.assertLess(abs(tiny/(1e-25*unit)-1), 2e-15)
        self.assertEqual(tiny.real, 0)


if __name__ == "__main__":
    unittest.main()
