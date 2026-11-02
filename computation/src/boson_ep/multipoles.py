"""Point-companion multipoles with analytic angular integration.

The finite spheroidal expansion is integrated term by term. This avoids
subtracting order-one angular samples to recover very small higher moments.
"""

from functools import lru_cache
import math

import numpy as np
from scipy.special import sph_harm_y
from sympy.physics.wigner import gaunt

from .kernel_validation import _complex_quad, _gauss_integral
from .projection import schrodinger_product_factor
from .relativity import build_radial_ode_evaluator
from .relativistic_tides import _complex_fsum, _qbs_decay_rate


@lru_cache(maxsize=4096)
def _polar_gaunt(li, mi, lf, mf, ell):
    # At phi=0, Y(l,-m)=(-1)^m Y(l,m); choose azimuths summing to zero.
    return (-1.0)**mi * float(gaunt(lf, li, ell, mf, -mi, mi-mf)) / (2*math.pi)


def spectral_angular_integral(initial, final, ell):
    """Integral of S_f S_i Y_(ell,mi-mf) sin(theta) dtheta, without conjugation."""
    if ell < abs(initial.state.m-final.state.m):
        return 0j
    return _complex_fsum([
        complex(ci*cf*_polar_gaunt(int(li), initial.state.m, int(lf), final.state.m, ell))
        for li, ci in zip(initial.angular_l_values, initial.angular_coefficients, strict=True)
        for lf, cf in zip(final.angular_l_values, final.angular_coefficients, strict=True)
        if abs(int(li)-int(lf)) <= ell <= int(li)+int(lf)
        and (int(li)+int(lf)+ell) % 2 == 0
    ])


class PointCompanionMultipoles:
    """Exact interior/exterior potential, on a finite radial mode domain."""

    def __init__(self, config, pair):
        self.config, self.pair = config, pair
        if (pair[0].state, pair[1].state) != (config.initial, config.final):
            raise ValueError("Mode pair does not match the transition")
        r_plus = 1+math.sqrt(1-pair[0].chi**2)
        decays = [_qbs_decay_rate(config.alpha, mode.frequency_M) for mode in pair]
        self.lower = r_plus+config.settings.horizon_cutoffs[0]
        self.outer = r_plus+config.settings.mode.outer_decay_lengths/min(decays)
        span = 1.001*(self.outer-r_plus)
        self.radial = [build_radial_ode_evaluator(
            config.alpha, mode.chi, mode.state, mode.frequency_M,
            mode.separation_constant, mode.radial_coefficients,
            outer_decay_lengths=span*decay)
            for mode, decay in zip(pair, decays, strict=True)]

    def kernel(self, ell, separation, nodes=112, strategy="gauss"):
        """Kernel before multiplication by q/b**(ell+1), with M=1."""
        mi, mf = (mode.state.m for mode in self.pair)
        harmonic = mi-mf
        if ell < 2 or abs(harmonic) > ell or separation <= 0:
            raise ValueError("Invalid tidal multipole or separation")
        if (ell+harmonic) % 2:
            return 0j
        if strategy == "gauss":
            integrate = lambda f, a, b: _gauss_integral(f, a, b, nodes)
        elif strategy == "quad":
            integrate = lambda f, a, b: _complex_quad(f, a, b, 1e-9)
        else:
            raise ValueError("Unknown radial integration strategy")
        product = lambda r: self.radial[0](r)[0]*self.radial[1](r)[0]
        split = min(max(separation, self.lower), self.outer)
        radial = integrate(lambda r: r**(ell+2)*product(r), self.lower, split)
        radial += separation**(2*ell+1)*integrate(
            lambda r: r**(1-ell)*product(r), split, self.outer)
        source = -4*math.pi/(2*ell+1)*np.conjugate(
            sph_harm_y(ell, harmonic, math.pi/2, 0.0))
        factor = schrodinger_product_factor(*(m.frequency_M for m in self.pair))
        return complex(self.config.alpha*2*math.pi*source*radial
                       * spectral_angular_integral(*self.pair, ell)*factor)
