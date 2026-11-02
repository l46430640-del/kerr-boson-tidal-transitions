"""Circular |m|=2 tide with local electric and magnetic STF moments.

The electric quadrupole is normalized through first frequency order. The
magnetic octupole has its leading circular-binary amplitude, including the
spin-induced quadrupole component. Multipliers refer to unit M_* M^2/b^3;
external PN amplitude corrections and higher multipoles are separate.
"""

import math

from .dynamic_tide import IngoingHertzTide
from .tidal_matching import HertzFrequencyResponse
from .tidal_metric import advanced_coefficient_to_bl


class CircularBinaryTide:
    """Vacuum continuation of the first-frequency matched near-zone field.

    Each separated component solves the finite-frequency vacuum equation.
    Boundary amplitudes beyond first frequency order have not been matched.
    All quantities use M=1 and the local orbital frequency Omega.
    """

    def __init__(self, chi, omega_orb_M, outer, *, response=None, **solver_options):
        self.quadrupole = IngoingHertzTide(chi,omega_orb_M,outer,ell=2,**solver_options)
        self.octupole = IngoingHertzTide(chi,omega_orb_M,outer,ell=3,**solver_options)
        self.chi, self.omega_orb_M = chi,omega_orb_M
        self.frequency = -2*omega_orb_M
        if response is None:
            response = HertzFrequencyResponse(chi)
        if response.chi != chi:
            raise ValueError("Matching response and metric must use the same spin")
        self.electric_normalization = response.stf_electric_coefficient()
        self.electric_factor = 1+self.frequency*self.electric_normalization
        self.magnetic_quadrupole_factor = -20*chi*omega_orb_M/9
        # F_3 = r-1-2ia/3 at w=0; the angular eigenvector uses sqrt(7/5) S_3.
        horizon_value = self.octupole.r_plus-1-2j*chi/3
        self.magnetic_octupole_factor = 2j*omega_orb_M*horizon_value/(3*math.sqrt(7/5))

    def electric_metric(self, radius, theta):
        return self.electric_factor*self.quadrupole.electric_metric(radius,theta)

    def magnetic_metric(self, radius, theta):
        return (self.magnetic_quadrupole_factor*self.quadrupole.electric_metric(radius,theta)
                +self.magnetic_octupole_factor*self.octupole.electric_metric(radius,theta))

    def metric(self, radius, theta):
        return self.electric_metric(radius,theta)+self.magnetic_metric(radius,theta)

    def electric_metric_bl(self, radius, theta):
        return advanced_coefficient_to_bl(self.electric_metric(radius,theta),radius,
                                          self.chi,self.omega_orb_M)

    def magnetic_metric_bl(self, radius, theta):
        return advanced_coefficient_to_bl(self.magnetic_metric(radius,theta),radius,
                                          self.chi,self.omega_orb_M)

    def metric_bl(self, radius, theta):
        return advanced_coefficient_to_bl(self.metric(radius,theta),radius,
                                          self.chi,self.omega_orb_M)
