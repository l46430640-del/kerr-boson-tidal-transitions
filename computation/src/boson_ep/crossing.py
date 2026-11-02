"""Dissipative two-level propagation with independent directed couplings."""

import math
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import _cf4_propagator


_DENSITY_BASIS = np.array([
    [[1,0],[0,0]], [[0,0],[0,1]], [[0,1],[1,0]], [[0,-1j],[1j,0]],
],dtype=complex)


def rotating_crossing_hamiltonian(detuning,decay,forward,reverse):
    """Rotating frame for the forward exp(+2 i orbital_phase) harmonic.

    Factoring exp(-i omega_i t) from the initial component and
    exp(-i omega_i t+2 i orbital_phase) from the final component gives
    detuning=2(Omega-Omega_res) on the final diagonal.
    """
    return np.array([[0,reverse],[forward,detuning-1j*decay]],dtype=complex)


def _density_matrix(values):
    return np.array([[values[0],values[2]-1j*values[3]],
                     [values[2]+1j*values[3],values[1]]],dtype=complex)


def density_generator(hamiltonian):
    """Real generator for populations, coherence and two norm-balance integrals.

    The coherence is rho[1,0]. The last two entries integrate the diagonal
    loss and off-diagonal norm source; neither is assumed positive.
    """
    h = np.asarray(hamiltonian,dtype=complex)
    if h.shape != (2,2) or not np.isfinite(h).all():
        raise ValueError("Require a finite two-level Hamiltonian")
    result = np.zeros((6,6))
    for column,rho in enumerate(_DENSITY_BASIS):
        derivative = -1j*(h@rho-rho@h.conj().T)
        result[:4,column] = (derivative[0,0].real,derivative[1,1].real,
                             derivative[1,0].real,derivative[1,0].imag)
        result[4,column] = -2*(h[0,0].imag*rho[0,0].real+h[1,1].imag*rho[1,1].real)
        result[5,column] = 2*(h[0,1]*rho[1,0]+h[1,0]*rho[0,1]).imag
    return result


def propagate_crossing(hamiltonian, interval, *, dt_dx=None, initial_density=None,
                       rtol=2e-9, atol=2e-13, max_step=math.inf, samples=None,
                       initial_exponential_span=0.):
    """Integrate d rho/dx=-i(dt/dx)(H rho-rho H^dagger) with implicit Radau.

    H is specified in one fixed reference basis. Occupations in another basis
    require transforming the endpoint projector as well as the density.
    No populations are clipped, and no Landau-Zener value replaces the ODE.
    """
    left,right = map(float,interval)
    if (not math.isfinite(left+right) or right <= left
            or not math.isfinite(rtol) or not math.isfinite(atol)
            or rtol <= 0 or atol <= 0 or math.isnan(max_step) or max_step <= 0):
        raise ValueError("Require an ordered finite interval and positive solver tolerances")
    span = float(initial_exponential_span)
    if not math.isfinite(span) or not 0<=span<right-left:
        raise ValueError("The initial exponential span must fit within the interval")
    initial = np.diag([1.,0.]).astype(complex) if initial_density is None else np.asarray(initial_density,dtype=complex)
    if (initial.shape != (2,2) or not np.isfinite(initial).all()
            or not np.allclose(initial,initial.conj().T,rtol=0,atol=1e-13)
            or np.min(np.linalg.eigvalsh(initial)) < -1e-13):
        raise ValueError("Initial density must be finite, Hermitian and positive semidefinite")
    initial_norm = np.trace(initial).real
    if initial_norm <= 0:
        raise ValueError("Initial density must have positive trace")
    y0 = np.array([initial[0,0].real,initial[1,1].real,
                   initial[1,0].real,initial[1,0].imag,0.,0.])

    def generator(x):
        scale = 1. if dt_dx is None else float(dt_dx(x))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("The evolution coordinate must increase with time")
        return scale*density_generator(hamiltonian(x))

    def exponential_start(duration,count):
        state = y0.copy()
        step = duration/count
        problem = SimpleNamespace(generator=lambda x,unused:generator(left+x))
        for index in range(count):
            state = _cf4_propagator(problem,index*step,step,0)@state
        return state

    count,start_error = 0,0.
    start = y0
    if span:
        coarse = exponential_start(span,1)
        for count in (2,4,8,16,32,64,128,256,512,1024):
            start = exponential_start(span,count)
            start_error = float(np.linalg.norm(start-coarse)/(atol+rtol*np.linalg.norm(start)))
            if start_error < .1:
                break
            coarse = start
        else:
            raise RuntimeError("Initial exponential relaxation did not converge")
    # Resolve a short initial relaxation even when the external coordinate
    # is far from zero and its floating-point spacing exceeds that layer.
    solution = solve_ivp(lambda s,y:generator(left+s)@y,(span,right-left),start,method="Radau",
                         jac=lambda s,y:generator(left+s),rtol=rtol,atol=atol,
                         max_step=max_step,dense_output=samples is not None)
    if not solution.success:
        raise RuntimeError(solution.message)
    final = solution.y[:,-1]
    rho = _density_matrix(final)
    balance = np.trace(rho).real+final[4]-final[5]-initial_norm
    result = {"density":rho,"initial_survival":float(final[0]/initial_norm),
              "final_population":float(final[1]/initial_norm),
              "diagonal_loss":float(final[4]/initial_norm),
              "mixing_gain":float(final[5]/initial_norm),
              "norm_balance_residual":float(abs(balance)/initial_norm),
              "purity_residual":float(abs(np.linalg.det(rho))/initial_norm**2),
              "minimum_density_eigenvalue":float(np.min(np.linalg.eigvalsh(rho))/initial_norm),
              "steps":len(solution.t)-1,"function_evaluations":solution.nfev,
              "initial_exponential_steps":count,"initial_exponential_scaled_error":start_error}
    if samples is not None:
        coordinates = np.asarray(samples,dtype=float)
        if coordinates.ndim != 1 or not np.isfinite(coordinates).all() or np.any(coordinates<left) or np.any(coordinates>right):
            raise ValueError("Requested samples must lie inside the evolution interval")
        result["x"] = coordinates
        values = [exponential_start(x-left,max(2,count)) if x-left<span else solution.sol(x-left)
                  for x in coordinates]
        result["density_samples"] = np.array([_density_matrix(v) for v in values])
    return result
