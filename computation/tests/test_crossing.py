import math
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from boson_ep.crossing import density_generator, propagate_crossing, rotating_crossing_hamiltonian
from boson_ep.dynamics import _cf4_propagator
from boson_ep.projection import transform_amplitude_generator


class CrossingTests(unittest.TestCase):
    def test_rotating_frame_detuning_sign_follows_forward_harmonic(self):
        ei,ef,omega,gamma = .2,.18,.011,.003
        forward,reverse = .03+.01j,.04-.02j
        for t in (0,1,7):
            physical = np.array([[ei,reverse*np.exp(-2j*omega*t)],
                                  [forward*np.exp(2j*omega*t),ef-1j*gamma]])
            basis = np.diag([np.exp(1j*ei*t),np.exp(1j*(ei-2*omega)*t)])
            rate = np.diag([1j*ei,1j*(ei-2*omega)])@basis
            actual = transform_amplitude_generator(physical,basis,rate)
            expected = rotating_crossing_hamiltonian(ef-ei+2*omega,gamma,forward,reverse)
            np.testing.assert_allclose(actual,expected,atol=1e-16)

    def test_constant_nonsymmetric_hamiltonian_matches_exponential(self):
        for h in (np.array([[.2,.3+.1j],[.4-.05j,-.1-.2j]]),
                  np.array([[0,.2j],[-.3j,.1-.4j]])):
            actual = propagate_crossing(lambda x:h,(0,4),rtol=2e-11,atol=1e-14)
            state = expm(-4j*h)[:,0]
            expected = np.outer(state,state.conj())
            np.testing.assert_allclose(actual["density"],expected,rtol=1e-10,atol=1e-12)
            self.assertLess(actual["norm_balance_residual"],1e-12)
            self.assertLess(actual["purity_residual"],1e-12)

    def test_density_equation_matches_amplitude_equation(self):
        def h(t):
            return np.array([[0,.3+.02j*t],[.2-.01j*t,-t-.4j]])
        actual = propagate_crossing(h,(-6,6),rtol=2e-11,atol=1e-14)
        state = solve_ivp(lambda t,c:-1j*h(t)@c,(-6,6),[1+0j,0j],method="DOP853",
                          rtol=1e-12,atol=1e-14).y[:,-1]
        np.testing.assert_allclose(actual["density"],np.outer(state,state.conj()),rtol=1e-9,atol=1e-11)

    def test_linear_sweep_recovers_width_independent_lz_limit(self):
        z = .12
        for gamma in (.5,4.,20.):
            edge = 120*max(1,gamma)
            result = propagate_crossing(lambda x:np.array([[0,math.sqrt(z)],
                                                           [math.sqrt(z),-x-1j*gamma]]),
                                        (-edge,edge),rtol=2e-9,atol=1e-13)
            # A finite window misses the Lorentzian tails, O(z gamma/edge).
            self.assertLess(abs(result["initial_survival"]-math.exp(-2*math.pi*z)),.0021)
            self.assertLess(result["norm_balance_residual"],1e-10)
            self.assertLess(abs(result["mixing_gain"]),1e-12)

    def test_endpoint_extension_converges_towards_lz(self):
        z,gamma = .1,10.
        errors = []
        for factor in (20,40,80):
            result = propagate_crossing(lambda x:np.array([[0,math.sqrt(z)],
                                                           [math.sqrt(z),-x-1j*gamma]]),
                                        (-factor*gamma,factor*gamma))
            errors.append(abs(result["initial_survival"]-math.exp(-2*math.pi*z)))
        self.assertTrue(errors[0]>errors[1]>errors[2])
        self.assertLess(errors[2]/errors[0],.26)

    def test_general_basis_covariance_with_endpoint_projector(self):
        def h(t):
            return np.array([[0,.2],[.2,-t-.3j]],dtype=complex)
        def maps(t):
            off = np.array([[0,.08j*np.exp(.2j*t)],[.05*np.exp(-.2j*t),0]])
            rate = off*np.array([[0,.2j],[-.2j,0]])
            return np.eye(2)+off,rate
        original = propagate_crossing(h,(-4,4),rtol=2e-11,atol=1e-14)
        start = maps(-4)[0][:,0]
        shifted = propagate_crossing(lambda t:transform_amplitude_generator(h(t),*maps(t)),(-4,4),
                                      initial_density=np.outer(start,start.conj()),rtol=2e-11,atol=1e-14)
        end = maps(4)[0]
        np.testing.assert_allclose(shifted["density"],end@original["density"]@end.conj().T,atol=1e-10)
        inverse = np.linalg.inv(end)
        projector = inverse.conj().T@np.diag([1,0])@inverse
        self.assertAlmostEqual(np.trace(projector@shifted["density"]).real,
                               original["initial_survival"],places=10)

    def test_time_coordinate_conversion(self):
        h = np.array([[0,.3],[.3,-.2j]])
        time = propagate_crossing(lambda t:h,(0,6))
        scaled = propagate_crossing(lambda x:h,(0,2),dt_dx=lambda x:3)
        np.testing.assert_allclose(time["density"],scaled["density"],atol=1e-10)

    def test_initial_relaxation_below_external_coordinate_spacing(self):
        h = np.diag([0,-1e15j])
        result = propagate_crossing(lambda x:h,(-50,50),initial_density=np.diag([.5,.5]))
        self.assertAlmostEqual(result["initial_survival"],.5,places=12)
        self.assertLess(abs(result["final_population"]),1e-12)
        self.assertAlmostEqual(result["diagonal_loss"],.5,places=12)

    def test_exponential_relaxation_matches_direct_amplitude_propagation(self):
        def h(t):
            return np.array([[0,.3+.01*t],[.3+.01*t,-1000-100j]])
        accelerated = propagate_crossing(h,(0,1),rtol=2e-10,atol=1e-14,
                                          initial_exponential_span=.4,samples=[0,.1,.4,1])
        state = solve_ivp(lambda t,c:-1j*h(t)@c,(0,1),[1+0j,0j],method="DOP853",
                          rtol=1e-12,atol=1e-14).y[:,-1]
        np.testing.assert_allclose(accelerated["density"],np.outer(state,state.conj()),atol=2e-10)
        np.testing.assert_allclose(accelerated["density_samples"][0],np.diag([1,0]),atol=1e-14)
        self.assertLess(accelerated["initial_exponential_scaled_error"],.1)

    def test_commutator_free_factor_order_has_fourth_order_convergence(self):
        generator = lambda t,g:-1j*np.array([[t,.4],[.4,-.2j]])
        problem = SimpleNamespace(generator=generator)
        exact = solve_ivp(lambda t,c:generator(t,0)@c,(0,1),[1+0j,0j],method="DOP853",
                          rtol=1e-13,atol=1e-15).y[:,-1]
        errors = []
        for count in (4,8,16):
            state = np.array([1+0j,0j])
            for index in range(count):
                state = _cf4_propagator(problem,index/count,1/count,0)@state
            errors.append(np.linalg.norm(state-exact))
        for coarse,fine in zip(errors[:-1],errors[1:]):
            self.assertGreater(coarse/fine,14)
            self.assertLess(coarse/fine,18)

    def test_complex_product_cannot_be_replaced_by_absolute_square(self):
        h = np.array([[0,.3+.1j],[.3+.1j,-.4j]])
        actual = propagate_crossing(lambda x:h,(0,3))
        other = h.copy()
        other[0,1] = h[1,0].conjugate()
        hermitian = propagate_crossing(lambda x:other,(0,3))
        self.assertGreater(abs(actual["initial_survival"]-hermitian["initial_survival"]),.02)
        self.assertGreater(abs(actual["mixing_gain"]),.01)

    def test_invalid_inputs(self):
        h = lambda x:np.zeros((2,2))
        for interval in ((1,0),(0,float("inf"))):
            with self.assertRaises(ValueError):
                propagate_crossing(h,interval)
        with self.assertRaises(ValueError):
            density_generator(np.zeros((3,3)))
        with self.assertRaises(ValueError):
            propagate_crossing(h,(0,1),initial_density=np.diag([1,-.1]))
        with self.assertRaises(ValueError):
            propagate_crossing(h,(0,1),dt_dx=lambda x:-1)
        with self.assertRaises(ValueError):
            propagate_crossing(h,(0,1),rtol=float("nan"))


if __name__ == "__main__":
    unittest.main()
