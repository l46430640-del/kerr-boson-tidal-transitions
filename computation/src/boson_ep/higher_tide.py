"""Leading electric l=4 boundary field and its spin-induced lower multipoles."""

import math

from .dynamic_tide import IngoingHertzTide
from .tidal_metric import advanced_coefficient_to_bl


def static_radial_polynomial(ell, radius, chi):
    """Monic horizon-regular Hertz polynomial at zero frequency, M=1."""
    if ell == 2:
        return 1+0j
    if ell == 3:
        return radius-1-2j*chi/3
    if ell == 4:
        return radius**2-(2+1j*chi)*radius+6/7+1j*chi-chi**2/7
    raise ValueError("Require ell = 2, 3 or 4")


class ElectricHexadecapoleTide:
    """Per unit epsilon_2/b^2, with M=1 and harmonic exp(2i Omega v-2i phi).

    Matches the growing flat-buffer curvature of the static Cartesian STF
    electric l=4 moment. Spin requires l=3 and l=2 Hertz pieces as well.
    Each piece is continued with its ingoing finite-frequency solution;
    boundary terms of relative Omega, orbital PN and higher tidal orders
    are not fixed by this leading-moment calculation.
    """

    def __init__(self, chi, omega_orb_M, outer, **solver_options):
        self.chi, self.omega_orb_M = chi, omega_orb_M
        self.frequency = -2*omega_orb_M
        self.components = {ell:IngoingHertzTide(chi,omega_orb_M,outer,ell=ell,**solver_options)
                           for ell in (2,3,4)}
        horizon = self.components[2].r_plus
        coefficients = {2:55*chi**2/42,3:-.5j*chi,4:-1/6}
        self.factors = {ell:coefficient*static_radial_polynomial(ell,horizon,chi)
                       /math.sqrt((2*ell+1)/5) for ell,coefficient in coefficients.items()}

    def component_metric(self, ell, radius, theta):
        return self.factors[ell]*self.components[ell].electric_metric(radius,theta)

    def metric(self, radius, theta):
        return sum(self.component_metric(ell,radius,theta) for ell in self.components)

    def metric_bl(self, radius, theta):
        return advanced_coefficient_to_bl(self.metric(radius,theta),radius,self.chi,self.omega_orb_M)
