"""Circular orbital angular momentum and scalar-feedback evolution."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from .binary_tide import circular_local_tide_15pn
from .crossing import _density_matrix, density_generator


def _angular_momentum(omega, companion_mass, primary_mass, primary_spin):
    total = primary_mass+companion_mass
    x1 = primary_mass/total
    nu = primary_mass*companion_mass/total**2
    chi = primary_spin/primary_mass**2
    y = (total*omega)**(2/3)
    a = 1.5-nu/6
    b = -chi*(10*x1*x1/3+13*nu/6)
    leading = primary_mass*companion_mass/total**(1/3)*omega**(-1/3)
    return leading*(1+a*y+b*y**1.5)


def circular_angular_momentum_15pn(omega, q, chi, *, primary_mass=1.0):
    """L and derivatives in initial-primary units at fixed local drive Omega.

    q is the fixed companion mass in those units. The aligned-spin circular
    J-S relation is from Kidder (1995), Sec. IV, transformed with the local
    frequency relation already used in binary_tide. The secondary is nonspinning.
    """
    if not (omega > 0 and q > 0 and primary_mass > 0 and abs(chi) < 1):
        raise ValueError("Require positive masses/frequency and subextremal spin")
    spin = chi*primary_mass**2
    total = primary_mass+q
    x1, nu = primary_mass/total, primary_mass*q/total**2
    y = (total*omega)**(2/3)
    a, b = 1.5-nu/6, -chi*(10*x1*x1/3+13*nu/6)
    leading = primary_mass*q/total**(1/3)*omega**(-1/3)
    derivative_factor = 1-a*y-2*b*y**1.5
    derivative = -leading/(3*omega)*derivative_factor
    step = 1e-20
    mass_derivative = _angular_momentum(omega,q,primary_mass+1j*step,spin).imag/step
    spin_derivative = -leading*(10*x1*x1/3+13*nu/6)*y**1.5/primary_mass**2
    orbit = circular_local_tide_15pn(primary_mass*omega, q/primary_mass, chi)
    chirp = orbit["chirp_M2_local_15pn"]/primary_mass**2
    if derivative >= 0 or chirp <= 0:
        raise ValueError("The circular PN inspiral is outside its monotonic branch")
    return {"angular_momentum": float(_angular_momentum(omega,q,primary_mass,spin)),
            "frequency_derivative": derivative, "mass_derivative_at_fixed_spin": mass_derivative,
            "spin_derivative_at_fixed_mass": spin_derivative,
            "frequency_derivative_over_leading": derivative_factor,
            "gw_chirp": chirp, "gw_torque": -derivative*chirp}


def eigenfrequency_drift(alpha, chi, frequencies, alpha_derivative, chi_derivative,
                         mass_change, angular_momentum_change):
    """Linear frequency drift in initial-primary time units, including widths."""
    spin_change = angular_momentum_change-2*chi*mass_change
    return ((alpha*np.asarray(alpha_derivative)-np.asarray(frequencies))*mass_change
            + np.asarray(chi_derivative)*spin_change)


def expectation_row(operator):
    """Hermitian expectation against (pi,pf,Re rho_fi,Im rho_fi)."""
    matrix = np.asarray(operator)
    return np.array([matrix[0,0].real,matrix[1,1].real,2*matrix[0,1].real,-2*matrix[0,1].imag])


def metric_energy_flux_operator(overlap, temporal_generator):
    """First metric variation of the inward energy flux on a radial surface.

    The three surface overlaps contract S^{ra} with the field's temporal
    and spatial derivatives, with the same harmonic convention as scalar_torque.
    """
    a,b,c = np.asarray(overlap,dtype=complex)[:3]
    generator = np.asarray(temporal_generator,dtype=complex)
    initial,final = np.eye(2,dtype=complex)
    di,df = -1j*generator[0],-1j*generator[1]
    product = -.5*(2*a*np.outer(df.conj(),di)+b*np.outer(df.conj(),initial)
                    +c*np.outer(final,di))
    return product+product.conj().T


def propagate_feedback(model, interval, frequency_scale, cloud_fraction, *,
                       torque_feedback=True, initial_span=0., rtol=2e-9, atol=2e-14,
                       samples=None):
    """Evolve density and absorbed E,J, with a state-dependent circular chirp.

    model(x, absorbed_E/fc, absorbed_J/fc) returns H and physical scalar
    torque/flux operators per unit initial cloud mass. Eight real variables
    retain the two norm-balance integrals and the absorbed energy/angular
    momentum. The orbit must remain monotonically inspiralling.
    """
    left, right = interval
    if not (np.isfinite([left,right,cloud_fraction,frequency_scale,initial_span,rtol,atol]).all()
            and right > left and cloud_fraction >= 0 and frequency_scale > 0
            and 0 <= initial_span < right-left and rtol > 0 and atol > 0):
        raise ValueError("Require a finite ordered interval and nonnegative cloud mass")
    initial = np.zeros(8)
    initial[0] = 1

    def coefficients(x, state):
        data = model(x,state[6],state[7])
        matrix = np.zeros((8,8))
        matrix[:6,:6] = density_generator(data["hamiltonian"])
        matrix[6,:4] = expectation_row(data["horizon_energy"])
        matrix[7,:4] = expectation_row(data["horizon_angular"])
        orbit = data["orbit"]
        correction = 0.
        chirp_row = np.zeros(4)
        if torque_feedback:
            chirp_row = cloud_fraction/abs(orbit["frequency_derivative"])*(expectation_row(data["cloud_torque"])
                + orbit["mass_derivative_at_fixed_spin"]*matrix[6,:4]
                + orbit["spin_derivative_at_fixed_mass"]*matrix[7,:4])
            correction = chirp_row @ state[:4]
        chirp = orbit["gw_chirp"]+correction
        if not np.isfinite(chirp) or chirp <= 0:
            raise ValueError("Feedback stopped the monotonic inspiral")
        return frequency_scale/(2*chirp)*matrix, chirp_row/chirp, correction/orbit["gw_chirp"]

    def jacobian(s,state):
        x = left+s
        matrix, gradient, _ = coefficients(x,state)
        result = matrix.copy()
        result[:,:4] -= np.outer(matrix @ state,gradient)
        # Only the two absorbed charges enter the remaining model coefficients.
        # Inactive balance-integral columns must stay exactly zero; allowing an
        # adaptive finite-difference Jacobian to probe them overflows its steps.
        for column in (6,7):
            step = 1e-4*max(1.,abs(state[column]))
            plus, minus = state.copy(),state.copy()
            plus[column] += step
            minus[column] -= step
            result[:,column] = ((coefficients(x,plus)[0]-coefficients(x,minus)[0]) @ state)/(2*step)
        return result

    start_matrix, _, _ = coefficients(left, initial)
    start = expm(initial_span*start_matrix) @ initial if initial_span else initial
    solution = solve_ivp(lambda s,y: coefficients(left+s,y)[0] @ y,
                         (initial_span,right-left),start,method="Radau",rtol=rtol,atol=atol,
                         jac=jacobian,dense_output=True)
    if not solution.success:
        raise RuntimeError(solution.message)
    final = solution.y[:,-1]
    density = _density_matrix(final)
    result = {"initial_survival": float(final[0]), "final_population": float(final[1]),
              "absorbed_energy_per_cloud_fraction": float(final[6]),
              "absorbed_angular_momentum_per_cloud_fraction": float(final[7]),
              "norm_balance_residual": float(abs(final[0]+final[1]+final[4]-final[5]-1)),
              "minimum_density_eigenvalue": float(min(np.linalg.eigvalsh(density))),
              "purity_residual": float(abs(np.linalg.det(density))),
              "steps": len(solution.t)-1, "function_evaluations": solution.nfev,
              "density": density}
    if samples is not None:
        coordinates = np.asarray(samples)
        if (coordinates.ndim != 1 or not np.isfinite(coordinates).all()
                or np.any(coordinates < left) or np.any(coordinates > right)):
            raise ValueError("Samples outside the evolution interval")
        states = np.array([expm((x-left)*start_matrix) @ initial if x-left < initial_span
                           else solution.sol(x-left) for x in coordinates])
        result["coordinates"], result["states"] = coordinates, states
        result["relative_chirp_feedback"] = np.array([coefficients(x,y)[2]
                                                      for x,y in zip(coordinates,states,strict=True)])
    return result
