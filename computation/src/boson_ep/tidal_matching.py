"""Curvature normalization of the finite-frequency Hertz tide.

The CPY prescription refers to arXiv:1608.02899v3, Eq. (A13b). Its
first-frequency quadrupole matching is tested here; it is not a complete
binary tide. In particular it omits the independently sourced octupole.
"""

import math

import numpy as np
from scipy.integrate import solve_ivp


def _connection_jet(radius, chi, frequency):
    r, a, w = radius, chi, frequency
    delta, first = r*r-2*r+a*a, 2*(r-1)
    numerator = [(r*r+a*a)*w+2*a, 2*r*w, w, 0]
    coefficients = []
    for k in range(4):
        value = numerator[k]
        if k >= 1:
            value -= first*coefficients[k-1]
        if k >= 2:
            value -= coefficients[k-2]
        coefficients.append(value/delta)
    return np.array([2j*math.factorial(k)*coefficients[k] for k in range(4)])


def _fourth_derivative_weights(jet):
    a, b, c, d = jet
    return np.array([a**4+6*a*a*b+3*b*b+4*a*c+d,
                     4*a**3+12*a*b+4*c, 6*(a*a+b), 4*a, 1])


def _weight_frequency_derivative(jet, derivative):
    a, b, c, _ = jet
    e, f, g, h = derivative
    return np.array([4*a**3*e+12*a*e*b+6*a*a*f+6*b*f+4*e*c+4*a*g+h,
                     12*a*a*e+12*e*b+12*a*f+4*g, 12*a*e+6*f, 4*e, 0])


def _hertz_derivatives(radius, chi, derivatives):
    delta, first = radius**2-2*radius+chi**2, 2*(radius-1)
    polynomial = [delta**2, 2*delta*first, 2*first*first+4*delta, 12*first, 24]
    return np.array([sum(math.comb(k,j)*polynomial[j]*np.conj(derivatives[k-j])
                         for j in range(k+1)) for k in range(5)])


def electric_weyl_transfer(tide, radius):
    """Return D^4[Delta^2 conjugate(F)]/24 for the conjugate Hertz harmonic.

    In the Katagiri convention the equatorially even m=+2 coefficient of
    psi_0 is -3/4 times this transfer times S(theta), per unit epsilon.
    Buffer-region use avoids cancellation in the singular Kinnersley tetrad
    near the horizon. Radial derivatives follow from the separated ODE.
    """
    a, w, r = tide.chi, tide.frequency, radius
    f, first, second = tide.radial_values(r)
    delta, d = r*r-2*r+a*a, 2*(r-1)
    p = 3*d-2j*((r*r+a*a)*w+2*a)
    q = 4-tide.separation-10j*w*r
    p_first, p_second, q_first = 6-4j*r*w, -4j*w, -10j*w
    third = -((d+p)*second+(p_first+q)*first+q_first*f)/delta
    fourth = -((2*d+p)*third+(2+2*p_first+q)*second+(p_second+2*q_first)*first)/delta
    return complex(np.dot(_fourth_derivative_weights(_connection_jet(r,a,w)),
                          _hertz_derivatives(r,a,[f,first,second,third,fourth]))/24)


class HertzFrequencyResponse:
    """dF/dw at w=0, with F(r_plus)=1 and w=-2 Omega.

    Direct variation of the ODE avoids subtracting nearby finite-frequency
    solutions when extracting the constant in the large-radius expansion.
    """

    def __init__(self, chi, outer=800.0, *, rtol=3e-13, horizon_offset=1e-4,
                 series_order=10, max_step=None):
        if not (math.isfinite(chi) and abs(chi) < 1 and math.isfinite(outer) and outer > 20):
            raise ValueError("Require subextremal spin and finite outer radius > 20")
        self.chi, self.outer = chi, outer
        self.r_plus = 1+math.sqrt(1-chi*chi)
        gap = 2*(self.r_plus-1)
        self.inner = self.r_plus+horizon_offset
        if not (0 < horizon_offset < gap/10 and 0 < rtol < 1
                and isinstance(series_order,int) and series_order >= 3
                and (max_step is None or (math.isfinite(max_step) and max_step > 0))):
            raise ValueError("Invalid radial solver parameters")
        p0 = 3*gap-4j*chi
        coefficients = [0j, (20*chi/3+10j*self.r_plus)/p0]
        for k in range(1,series_order):
            coefficients.append(-(k*(k+5)*coefficients[k]-(10j if k == 1 else 0))
                                / ((k+1)*(gap*k+p0)))
        self.horizon_coefficients = np.array(coefficients)
        initial = [np.polynomial.polynomial.polyval(horizon_offset,coefficients),
                   np.polynomial.polynomial.polyval(horizon_offset,
                       np.arange(1,len(coefficients))*coefficients[1:])]

        def derivative(r, values):
            return [values[1], (20*chi/3+10j*r-(6*(r-1)-4j*chi)*values[1])
                    / (r*r-2*r+chi*chi)]

        result = solve_ivp(derivative,(self.inner,outer),initial,method="DOP853",
                           rtol=rtol,atol=rtol/100,dense_output=True,
                           max_step=max_step or outer/160)
        if not result.success:
            raise RuntimeError(result.message)
        self._solution = result.sol
        self.asymptotic_coefficients = [5j/3,2j]
        for j in range(1,5):
            self.asymptotic_coefficients.append(
                ((6-2*j+4j*chi)*self.asymptotic_coefficients[j]
                 +chi*chi*(j-1)*self.asymptotic_coefficients[j-1])/(5-j))

    def radial_derivatives(self, radius):
        r, a = radius, self.chi
        if not self.r_plus < r <= self.outer:
            raise ValueError("Radius outside the solved interval")
        if r < self.inner:
            x = r-self.r_plus
            f = np.polynomial.polynomial.polyval(x,self.horizon_coefficients)
            first = np.polynomial.polynomial.polyval(x,
                np.arange(1,len(self.horizon_coefficients))*self.horizon_coefficients[1:])
        else:
            f, first = self._solution(r)
        delta, d, p = r*r-2*r+a*a, 2*(r-1), 6*(r-1)-4j*a
        second = (20*a/3+10j*r-p*first)/delta
        third = (10j-(d+p)*second-6*first)/delta
        fourth = -((2*d+p)*third+14*second)/delta
        return np.array([f,first,second,third,fourth])

    def asymptotic_constant(self, radius=None):
        """Estimate C in dF/dw=5ir/3+2i log(r)+C+O(1/r).

        Remove inverse powers through r^-4; the remainder is O(log(r)/r^5).
        Compare extraction radii and solver tolerances before using C.
        """
        r = self.outer if radius is None else radius
        if r < 20:
            raise ValueError("Extract the asymptotic constant at r >= 20")
        value = self.radial_derivatives(r)[0]-5j*r/3-2j*np.log(r)
        for k in range(2,6):
            value += self.asymptotic_coefficients[k]/((k-1)*r**(k-1))
        return complex(value)

    def weyl_frequency_derivative(self, radius):
        """First-frequency coefficient of electric_weyl_transfer."""
        r, a = radius, self.chi
        jet = _connection_jet(r,a,0)
        derivative = _connection_jet(r,a,1)-jet
        return complex((np.dot(_fourth_derivative_weights(jet),
                                _hertz_derivatives(r,a,self.radial_derivatives(r)))
                        +np.dot(_weight_frequency_derivative(jet,derivative),
                                _hertz_derivatives(r,a,[1,0,0,0,0])))/24)

    def cpy_electric_coefficient(self):
        """N in F_CPY=(1+w N) F_horizon, accurate through first order in w.

        Uses the electric quadrupole curvature in CPY (2016), Eq. (A13b).
        This reference prescription is not a verification of their disputed
        binary horizon flux, or a treatment of the companion's magnetic tide.
        """
        source = np.conj(self.asymptotic_constant())+8*self.chi/3+17j/6
        target = 2j*math.log(2)-1j-2*self.chi*(math.pi**2/3-299/54)
        return complex(np.conj(target-source))

    def stf_electric_coefficient(self):
        """N for the local STF electric moment, including first-spin mixing.

        The external Cartesian curvature gives the constant 14 a/9 in
        dT/dw after the independent magnetic octupole is separated. The
        spin-independent phase retains the Schwarzschild CPY convention.
        This fixes first-frequency boundary data, not higher-order terms.
        """
        return complex(-self.asymptotic_constant()-10*self.chi/9
                       +1j*(23/6-2*math.log(2)))
