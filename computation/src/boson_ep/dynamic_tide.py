"""Finite-frequency ingoing Hertz solutions, before external tidal matching.

The radial amplitude is normalized at the horizon. This supplies a vacuum
metric but is not yet normalized to the companion's tidal moments.
The production kernels continue to use their original prescribed metric.
"""

from functools import lru_cache
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from scipy.special import eval_jacobi
import sympy as sp

from .tidal_metric import _metric_expressions, advanced_coefficient_to_bl


def angular_mode(spheroidicity, ell_max=8, *, ell=2):
    """s=-2,m=-2 eigenpair with unit spherical coefficient of degree ell."""
    if (not isinstance(ell_max, int) or ell_max < 3
            or not isinstance(ell, int) or not 2 <= ell <= ell_max
            or not math.isfinite(spheroidicity)):
        raise ValueError("Require 2 <= ell <= ell_max, ell_max >= 3, and finite real spheroidicity")
    degrees = np.arange(2, ell_max + 2)
    cosine = np.diag(-4 / (degrees * (degrees + 1)))
    for index, value in enumerate(degrees[1:], start=1):
        cosine[index, index-1] = cosine[index-1, index] = (value*value - 4) / (value * math.sqrt(4*value*value - 1))
    matrix = np.diag(degrees*(degrees+1)-2) - 4*spheroidicity*cosine - spheroidicity**2 * (cosine @ cosine)
    eigenvalues, vectors = eigh(matrix[:-1, :-1])
    index = ell-2
    coefficients = vectors[:, index] / vectors[index, index]
    return float(eigenvalues[index]), coefficients


def angular_values(theta, coefficients):
    """S, dS/dtheta, d2S/dtheta2 in the (1-cos theta)^2 normalization."""
    u = math.cos(theta)
    polynomials = np.zeros(3)
    for index, coefficient in enumerate(coefficients):
        factor = coefficient * math.sqrt((2*index+5)/5)
        polynomials[0] += factor * eval_jacobi(index, 4, 0, u)
        if index >= 1:
            polynomials[1] += factor * (index+5)/2 * eval_jacobi(index-1, 5, 1, u)
        if index >= 2:
            polynomials[2] += factor * (index+5)*(index+6)/4 * eval_jacobi(index-2, 6, 2, u)
    value = (1-u)**2 * polynomials[0]
    first_u = -2*(1-u)*polynomials[0] + (1-u)**2*polynomials[1]
    second_u = 2*polynomials[0] - 4*(1-u)*polynomials[1] + (1-u)**2*polynomials[2]
    return value, -math.sin(theta)*first_u, (1-u*u)*second_u - u*first_u


@lru_cache(maxsize=1)
def _reconstruction_function():
    (r, theta, a, omega), metric = _metric_expressions(separated=True)
    radial, angular = sp.Function("F")(r), sp.Function("S")(theta)
    values = sp.symbols("F0 F1 F2 S0 S1 S2")
    expressions = (radial, sp.diff(radial, r), sp.diff(radial, r, 2),
                   angular, sp.diff(angular, theta), sp.diff(angular, theta, 2))
    metric = metric.xreplace(dict(zip(expressions, values, strict=True)))
    return sp.lambdify((r, theta, a, omega, *values), list(metric), modules="numpy", cse=True)


class IngoingHertzTide:
    """Horizon-normalized m=-2 vacuum tide, harmonic exp(2i Omega v-2i phi).

    ``metric`` returns one Hertz branch and ``electric_metric`` selects its
    equatorially even sector. Neither fixes the amplitude relative to the
    external binary; that requires matching the Weyl curvature.
    """

    def __init__(self, chi, omega_orb_M, outer, *, ell=2, ell_max=8, rtol=2e-12,
                 horizon_offset=1e-4, series_order=8, max_step=None):
        if not (math.isfinite(chi) and abs(chi) < 1 and math.isfinite(omega_orb_M)):
            raise ValueError("Require a subextremal spin and finite real drive frequency")
        self.chi = chi
        self.omega_orb_M = omega_orb_M
        self.frequency = -2*omega_orb_M
        self.r_plus = 1 + math.sqrt(1-chi*chi)
        self.gap = 2*math.sqrt(1-chi*chi)
        self.inner = self.r_plus + horizon_offset
        self.outer = outer
        if not (math.isfinite(outer) and outer > self.inner and 0 < horizon_offset < self.gap/10
                and isinstance(series_order, int) and series_order >= 3 and 0 < rtol < 1
                and (max_step is None or max_step > 0)):
            raise ValueError("Invalid radial interval or Frobenius order")
        self.ell = ell
        self.angular_eigenvalue, self.angular_coefficients = angular_mode(chi*self.frequency, ell_max, ell=ell)
        self.separation = self.angular_eigenvalue + (chi*self.frequency)**2 + 4*chi*self.frequency
        w, rp, gap = self.frequency, self.r_plus, self.gap
        p0 = 3*gap - 2j*((rp*rp+chi*chi)*w+2*chi)
        p1, p2 = 6 - 4j*rp*w, -2j*w
        q0, q1 = 4 - self.separation - 10j*w*rp, -10j*w
        coefficients = [1.0 + 0j]
        for k in range(series_order):
            numerator = (k*(k-1)+p1*k+q0)*coefficients[k]
            if k:
                numerator += (p2*(k-1)+q1)*coefficients[k-1]
            coefficients.append(-numerator / ((k+1)*(gap*k+p0)))
        self.horizon_coefficients = np.array(coefficients)
        self._solution = None
        self.evaluations = 0
        if w != 0 or ell not in (2,3):
            initial = [np.polynomial.polynomial.polyval(horizon_offset, coefficients),
                       np.polynomial.polynomial.polyval(horizon_offset, np.arange(1, len(coefficients))*coefficients[1:])]
            def derivative(r, values):
                f, first = values
                delta = r*r - 2*r + chi*chi
                momentum = (r*r+chi*chi)*w + 2*chi
                second = (-(6*(r-1)-2j*momentum)*first - (4-self.separation-10j*w*r)*f) / delta
                return np.array([first, second])
            result = solve_ivp(derivative, (self.inner, outer), initial, method="DOP853",
                               rtol=rtol, atol=rtol/100, dense_output=True,
                               max_step=max_step or outer/80)
            if not result.success:
                raise RuntimeError(result.message)
            self._solution = result.sol
            self.evaluations = result.nfev

    @lru_cache(maxsize=16384)
    def radial_values(self, radius):
        if not self.r_plus < radius <= self.outer:
            raise ValueError("Radius outside the solved interval")
        if self._solution is None:
            if self.ell == 3:
                denominator = self.r_plus-1-2j*self.chi/3
                return (radius-1-2j*self.chi/3)/denominator, 1/denominator, 0j
            return 1.0+0j, 0j, 0j
        if radius < self.inner:
            x = radius - self.r_plus
            f = np.polynomial.polynomial.polyval(x, self.horizon_coefficients)
            first = np.polynomial.polynomial.polyval(x, np.arange(1, len(self.horizon_coefficients))*self.horizon_coefficients[1:])
        else:
            f, first = self._solution(radius)
        delta = radius*radius - 2*radius + self.chi*self.chi
        momentum = (radius*radius+self.chi*self.chi)*self.frequency + 2*self.chi
        second = (-(6*(radius-1)-2j*momentum)*first - (4-self.separation-10j*self.frequency*radius)*f) / delta
        return f, first, second

    @lru_cache(maxsize=2048)
    def _angular_values(self, theta):
        return angular_values(theta, self.angular_coefficients)

    def metric(self, radius, theta):
        values = (*self.radial_values(radius), *self._angular_values(theta))
        return np.asarray(_reconstruction_function()(radius, theta, self.chi, self.omega_orb_M, *values), dtype=complex).reshape(4, 4)

    def electric_metric(self, radius, theta):
        """Equatorially even sector: electric for l=2, magnetic for l=3.

        Reflection supplies the opposite helicity needed for the even field
        without another radial solve. The method name is retained for the
        original quadrupole API.
        """
        reflection = np.array([1, 1, -1, 1])
        reflected = self.metric(radius, math.pi-theta) * np.outer(reflection, reflection)
        return (self.metric(radius, theta) + reflected) / 2

    def metric_bl(self, radius, theta):
        return advanced_coefficient_to_bl(self.electric_metric(radius, theta), radius,
                                           self.chi, self.omega_orb_M)

def schwarzschild_frequency_derivative(radius):
    """dF/domega at omega=0 with F(r_+)=1, where omega=-2 Omega."""
    if radius < 2:
        raise ValueError("Require r >= 2")
    return 1j * (5*radius/3 + 2*math.log(radius/2) - 2/radius - 2/(3*radius*radius) - 13/6)
