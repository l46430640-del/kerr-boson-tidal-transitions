import unittest

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

from boson_ep.orbital_feedback import circular_angular_momentum_15pn, eigenfrequency_drift, expectation_row, metric_energy_flux_operator, propagate_feedback
from boson_ep.scalar_balance import scalar_currents
from tests import test_scalar_torque


class OrbitalFeedbackTests(unittest.TestCase):
    def test_orbital_derivatives_at_fixed_spin_and_mass(self):
        omega,q,mass,spin = .003,.02,1.1,.61
        value = lambda w,m,s: circular_angular_momentum_15pn(w,q,s/m**2,primary_mass=m)["angular_momentum"]
        data = circular_angular_momentum_15pn(omega,q,spin/mass**2,primary_mass=mass)
        for index,key in enumerate(("frequency_derivative","mass_derivative_at_fixed_spin","spin_derivative_at_fixed_mass")):
            args = [omega,mass,spin]
            step = 1e-5*args[index]
            args[index] += step
            plus = value(*args)
            args[index] -= 2*step
            minus = value(*args)
            self.assertAlmostEqual((plus-minus)/(2*step)/data[key],1,places=7)

    def test_kidder_separation_and_local_frequency_expansions(self):
        omega,q,chi = 2e-5,.03,.7
        total = 1+q
        nu,x1 = q/total**2,1/total
        y = (total*omega)**(2/3)
        # Expand sqrt(r), Kidder's circular L(r), then Omega_bar/Omega.
        radius_1pn = -(3-nu)/3
        radius_spin = -chi*(2*x1*x1+3*nu)/3
        local_1pn = -nu/3
        local_spin = nu*chi/3
        expected = q/total**(1/3)*omega**(-1/3)*(1
            +(radius_1pn/2+2+local_1pn)*y
            +(radius_spin/2-chi*(3*x1*x1+2*nu)+local_spin)*y**1.5)
        actual = circular_angular_momentum_15pn(omega,q,chi)["angular_momentum"]
        self.assertAlmostEqual(actual/expected,1,places=14)

    def test_kerr_test_mass_limit(self):
        q,chi = 1e-10,.65
        for omega in (1e-6,1e-7):
            radius = (1/omega-chi)**(2/3)
            exact = (radius**2-2*chi*np.sqrt(radius)+chi**2)/(radius**.75*np.sqrt(radius**1.5-3*np.sqrt(radius)+2*chi))
            pn = circular_angular_momentum_15pn(omega,q,chi)["angular_momentum"]/q
            self.assertLess(abs(pn/exact-1),5*omega**(4/3))

    def test_gw_torque_and_mass_scaling(self):
        data = circular_angular_momentum_15pn(1e-5,.002,.6,primary_mass=1.2)
        self.assertAlmostEqual(data["gw_torque"],-data["frequency_derivative"]*data["gw_chirp"])
        self.assertGreater(data["gw_torque"],0)

    def test_frequency_response_in_initial_time_units(self):
        alpha,chi = .2,.7
        da,dc = np.array([.9-.01j,.8-.02j]),np.array([.003+.001j,.002-.003j])
        base = np.array([.19,.18-.001j])
        dm,dj = 2e-7,-3e-7
        actual = eigenfrequency_drift(alpha,chi,base,da,dc,dm,dj)
        shifted = (base+da*alpha*dm+dc*((chi+dj)/(1+dm)**2-chi))/(1+dm)-base
        np.testing.assert_allclose(actual,shifted,rtol=2e-5,atol=1e-15)

    def test_hermitian_expectation_convention(self):
        c = np.array([.4+.3j,.1-.7j])
        rho = np.outer(c,c.conj())
        operator = np.array([[.2,.3+.4j],[.3-.4j,-.1]])
        y = [rho[0,0].real,rho[1,1].real,rho[1,0].real,rho[1,0].imag]
        self.assertAlmostEqual(expectation_row(operator) @ y,np.trace(operator @ rho).real)

    @staticmethod
    def model(hamiltonian,torque=None,energy=None,angular=None):
        return {"hamiltonian":hamiltonian,
                "cloud_torque":np.zeros((2,2)) if torque is None else torque,
                "horizon_energy":np.zeros((2,2)) if energy is None else energy,
                "horizon_angular":np.zeros((2,2)) if angular is None else angular,
                "orbit":{"frequency_derivative":-1.,"gw_chirp":1.,
                         "mass_derivative_at_fixed_spin":0.,"spin_derivative_at_fixed_mass":0.}}

    def test_zero_feedback_matches_matrix_exponential(self):
        h = np.array([[0,.3+.02j],[.2-.01j,.4-.1j]])
        data = self.model(h)
        result = propagate_feedback(lambda x,e,j:data,(0,3),2,0)
        amplitude = expm(-3j*h) @ np.array([1.,0.])
        np.testing.assert_allclose(result["density"],np.outer(amplitude,amplitude.conj()),atol=2e-10)
        self.assertLess(result["norm_balance_residual"],1e-12)

    def test_torque_sign_and_state_dependent_chirp(self):
        gamma,fc = .2,.1
        for torque in (-1.,1.):
            data = self.model(np.diag([-1j*gamma,0]),torque=np.diag([torque,0]),energy=np.diag([.3,0]))
            result = propagate_feedback(lambda x,e,j:data,(0,2),2,fc)
            expected = brentq(lambda p: np.log(p)+fc*torque*(p-1)+4*gamma,.01,.99)
            self.assertAlmostEqual(result["initial_survival"],expected,places=9)
            self.assertAlmostEqual(result["absorbed_energy_per_cloud_fraction"],.3*(1-expected)/(2*gamma),places=9)

    def test_secular_absorption_matches_nonlinear_solution(self):
        gamma,beta = .05,.4
        def model(x,e,j):
            rate = gamma+beta*e
            return self.model(np.diag([-1j*rate,0]),energy=np.diag([2*rate,0]))
        result = propagate_feedback(model,(0,3),2,.01)
        expected = (gamma+beta)/(beta+gamma*np.exp(6*(gamma+beta)))
        self.assertAlmostEqual(result["initial_survival"],expected,places=9)
        self.assertAlmostEqual(result["absorbed_energy_per_cloud_fraction"],1-expected,places=9)

    def test_initial_exponential_and_density_samples(self):
        data = self.model(np.diag([-.1j,-2j]),energy=np.diag([.2,4]))
        result = propagate_feedback(lambda x,e,j:data,(-4,1),2,0,initial_span=.1,samples=[-4,-3.95,1])
        np.testing.assert_allclose(result["states"][:,0],np.exp(-.2*np.array([0,.05,5])),atol=1e-10)
        np.testing.assert_allclose(result["states"][:,0]+result["states"][:,6],1,atol=1e-12)

    def test_inspiral_stall_is_rejected(self):
        data = self.model(np.zeros((2,2)),torque=np.diag([-2.,0]))
        with self.assertRaisesRegex(ValueError,"monotonic"):
            propagate_feedback(lambda x,e,j:data,(0,1),2,1)

    def test_energy_flux_metric_variation_against_real_field(self):
        fixture = test_scalar_torque.ScalarTorqueTests()
        fixture.setUp()
        base = np.linalg.inv(fixture.inverse)
        step,direct = 1e-5,0.
        for phi in np.arange(16)*2*np.pi/16:
            value,gradient,h,_ = fixture.fields(phi)
            flux = []
            for sign in (-1,1):
                metric = base+sign*step*h
                currents = scalar_currents(value,gradient,np.linalg.inv(metric),fixture.mass)
                flux.append(-np.sqrt(np.linalg.det(metric)/np.linalg.det(base))*currents[1,1])
            direct += (flux[1]-flux[0])/(2*step*16)
        operator = metric_energy_flux_operator(fixture.overlaps(1),fixture.w)
        self.assertAlmostEqual(np.vdot(fixture.c,operator @ fixture.c).real,direct,places=11)


if __name__ == "__main__":
    unittest.main()
