import unittest

import numpy as np

from boson_ep.scalar_balance import axial_metric_source, ingoing_kerr_inverse, scalar_currents
from boson_ep.scalar_torque import background_angular_operators, background_angular_outer_flux, background_energy_operators, metric_angular_operator, torque_entry_bounds, torque_operator


class ScalarTorqueTests(unittest.TestCase):
    def setUp(self):
        self.inverse = np.array([[-1.4, .4, 0, 0], [.4, .7, 0, .1],
                                 [0, 0, .3, 0], [0, .1, 0, .4]])
        rng = np.random.default_rng(241)
        h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        self.h = (h + h.T) / 2
        self.fi, self.ff = .4 + .2j, .3 - .1j
        self.mi, self.mf = 1, -1
        self.si = np.array([.12 - .08j, -.3 + .1j, 1j * self.mi * self.fi])
        self.sf = np.array([.07 + .03j, .2 - .13j, 1j * self.mf * self.ff])
        self.w = np.array([[.2, .013 + .021j], [.007 - .014j, .19 - .003j]])
        self.c = np.array([.7 + .2j, -.2 + .4j])
        self.mass = .22

    def overlaps(self, first=0):
        trace = np.einsum("ab,ab", self.inverse, self.h)
        tensor = self.inverse @ self.h @ self.inverse - .5 * trace * self.inverse
        return np.array([
            self.ff.conjugate() * self.fi * tensor[first, 0],
            self.ff.conjugate() * (tensor[first, 1:] @ self.si),
            self.fi * (self.sf.conj() @ tensor[1:, first]),
            self.sf.conj() @ tensor[1:, 1:] @ self.si
            - .5 * trace * self.mass**2 * self.ff.conjugate() * self.fi,
        ])

    def fields(self, phi):
        phase = np.exp(1j * np.array([self.mi, self.mf]) * phi)
        values = np.array([self.fi, self.ff]) * phase
        value = values @ self.c
        gradient = np.concatenate(([-1j * values @ self.w @ self.c],
                                   self.c @ (phase[:, None] * np.array([self.si, self.sf]))))
        metric = self.h * np.exp(-2j * phi)
        return value, gradient, 2 * metric.real, 2 * (-2j * metric).real

    def test_operator_matches_direct_real_metric_stress_integral(self):
        direct = []
        for phi in np.arange(16) * 2 * np.pi / 16:
            value, gradient, _, derivative = self.fields(phi)
            direct.append(axial_metric_source(value, gradient, self.inverse, self.mass, derivative))
        operator = torque_operator(self.overlaps(), self.w)
        np.testing.assert_allclose(operator, operator.conj().T, atol=0)
        self.assertAlmostEqual(np.vdot(self.c, operator @ self.c).real, np.mean(direct), places=14)

    def test_charge_metric_variation_matches_full_metric_difference(self):
        base = np.linalg.inv(self.inverse)
        determinant = np.linalg.det(base)
        derivative = np.zeros(2)
        step = 1e-5
        for phi in np.arange(16) * 2 * np.pi / 16:
            value, gradient, h, _ = self.fields(phi)
            values = []
            for sign in (-1, 1):
                metric = base + sign * step * h
                measure = np.sqrt(np.linalg.det(metric) / determinant)
                current = scalar_currents(value, gradient, np.linalg.inv(metric), self.mass)
                values.append(measure * current[2, :2])
            derivative += (values[1] - values[0]) / (2 * step * 16)
        for index in (0, 1):
            operator = metric_angular_operator(self.overlaps(index), self.w, (self.mi, self.mf),
                                               inward_flux=index == 1)
            expected = (-1)**index * derivative[index]
            self.assertAlmostEqual(np.vdot(self.c, operator @ self.c).real, expected, places=10)

    def test_time_derivative_terms_are_not_replaced_by_mode_frequencies(self):
        actual = torque_operator(self.overlaps(), self.w)
        frozen = torque_operator(self.overlaps(), np.diag(np.diag(self.w)))
        self.assertGreater(np.linalg.norm(actual - frozen), 1e-4)

    def test_triangle_entry_bounds_include_nonconjugate_couplings(self):
        rng = np.random.default_rng(736)
        bounds = np.array([4., .2, .3, .5])
        frequencies = np.array([.2, .19-.003j])
        forward, reverse = .07, .09
        entries = torque_entry_bounds(bounds, frequencies, forward, reverse)
        for _ in range(200):
            overlap = bounds*rng.random(4)*np.exp(2j*np.pi*rng.random(4))
            generator = np.diag(frequencies)
            generator[1,0] = forward*rng.random()*np.exp(2j*np.pi*rng.random())
            generator[0,1] = reverse*rng.random()*np.exp(2j*np.pi*rng.random())
            self.assertTrue(np.all(abs(torque_operator(overlap, generator)) <= entries*(1+1e-14)))

    def test_kerr_charge_and_flux_operators_match_full_field(self):
        chi, theta = .7, 1.1
        rp = 1 + np.sqrt(1-chi*chi)
        sigma = rp*rp + chi*chi*np.cos(theta)**2
        self.inverse = ingoing_kerr_inverse(rp, theta, chi)
        fields, spatial = np.array([self.fi, self.ff]), np.array([self.si, self.sf])
        frequencies = np.diag(self.w)
        weights = -sigma*self.inverse[0,0]*abs(fields)**2
        radial = sigma*self.inverse[0,1]*(fields.conj()*spatial[:,0]).imag
        energies = [sigma*scalar_currents(value, np.r_[-1j*w*value, s], self.inverse, self.mass)[1,0]
                    for value,w,s in zip(fields,frequencies,spatial,strict=True)]
        j, fj = background_angular_operators(weights, radial, abs(fields)**2, (self.mi,self.mf), chi, self.w)
        e, fe = background_energy_operators(weights, energies, frequencies, abs(fields)**2,
                                            (self.mi,self.mf), chi, self.w)
        direct = []
        for phi in np.arange(16)*2*np.pi/16:
            value, gradient, _, _ = self.fields(phi)
            currents = sigma*scalar_currents(value, gradient, self.inverse, self.mass)
            direct.append([currents[2,0],-currents[2,1],currents[1,0],-currents[1,1]])
        expected = np.mean(direct, axis=0)
        actual = [np.vdot(self.c, op @ self.c).real for op in (j,fj,e,fe)]
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-16)

    def test_outer_flux_has_outward_orientation_and_radial_current(self):
        chi, theta, radius = .7, 1.1, 4.
        self.inverse = ingoing_kerr_inverse(radius, theta, chi)
        sigma = radius**2+chi**2*np.cos(theta)**2
        fields = np.array([self.fi,self.ff])
        first = np.array([self.si[0],self.sf[0]])
        radial_flux = (radius**2-2*radius+chi**2)*(fields.conj()*first).imag
        operator = background_angular_outer_flux(abs(fields)**2, radial_flux, (self.mi,self.mf),
                                                 chi, radius, self.w)
        direct = []
        for phi in np.arange(16)*2*np.pi/16:
            value, gradient, _, _ = self.fields(phi)
            direct.append(sigma*scalar_currents(value,gradient,self.inverse,self.mass)[2,1])
        self.assertAlmostEqual(np.vdot(self.c,operator @ self.c).real, np.mean(direct), places=14)


if __name__ == "__main__":
    unittest.main()
