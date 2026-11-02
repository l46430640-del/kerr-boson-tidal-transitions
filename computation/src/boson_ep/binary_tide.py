"""Circular electric tide and frequency sweep in the local black-hole frame.

The 1PN and 1.5PN functions keep their stated truncation separate. The chirp
assumes gravitational radiation from a prescribed orbit without cloud feedback.
"""

import math


def circular_local_tide(omega_local_M, q):
    """Return 1PN coefficients at fixed local drive frequency, in primary M units.

    Frame and tidal moments: Poisson (2015), Sec. XI, and
    Poisson--Corrigan (2018), Sec. IX. The barycentric chirp coefficient is
    743/336+11 nu/4. Products are expanded consistently through first PN order.
    """
    if not (math.isfinite(omega_local_M) and omega_local_M > 0
            and math.isfinite(q) and q > 0):
        raise ValueError("Require finite positive frequency and mass ratio")
    total = 1+q
    x1, x2, nu = 1/total, q/total, q/total**2
    y = (total*omega_local_M)**(2/3)
    time_coefficient = (2*x1+3*x2)*x2/2
    electric_coefficient = 1+nu+x1/2
    chirp_coefficient = time_coefficient-743/336-3*nu/4
    separation = total**(1/3)*omega_local_M**(-2/3)
    epsilon = q/total*omega_local_M**2
    chirp = 96/5*q/total**(1/3)*omega_local_M**(11/3)
    return {
        "local_frequency_M": omega_local_M, "q": q, "pn_parameter": y,
        "X1": x1, "X2": x2, "nu": nu,
        "dt_dtau_1pn": 1+time_coefficient*y,
        "orbital_frequency_M_1pn": omega_local_M*(1+nu*y),
        "separation_M_newtonian": separation,
        "separation_M_1pn": separation*(1-(3+nu)*y/3),
        "epsilon_newtonian": epsilon,
        "electric_amplitude_correction": electric_coefficient*y,
        "epsilon_electric_1pn": epsilon*(1+electric_coefficient*y),
        "chirp_M2_newtonian": chirp,
        "local_chirp_correction": chirp_coefficient*y,
        "chirp_M2_local_1pn": chirp*(1+chirp_coefficient*y),
        "local_vs_barycentric_chirp_correction": (time_coefficient+2*nu)*y,
        "coupling_squared_over_chirp_correction": (2*electric_coefficient-chirp_coefficient)*y,
        "omitted_spin_and_tail_order": y**1.5,
    }


def circular_local_tide_15pn(omega_local_M, q, chi1, chi2=0.0):
    """Aligned-spin circular tide through relative 1.5PN at fixed local Omega.

    chi1 and chi2 are signed along the orbital angular momentum. The local
    frame is Poisson's horizon-locked frame (gamma_d=-1); the barycentric
    spin-orbit chirp is Kidder (1995). This is a weak-field series, with
    constant masses/spins and no horizon absorption or cloud feedback.
    """
    if not all(math.isfinite(chi) and abs(chi) <= 1 for chi in (chi1,chi2)):
        raise ValueError("Require finite dimensionless spins with absolute value <= 1")
    result = circular_local_tide(omega_local_M,q)
    x1,x2,nu,y = (result[key] for key in ("X1","X2","nu","pn_parameter"))
    spin_order = y**1.5
    bar_chi = x1*(1+x1)*chi1+3*nu*chi2
    frame_spin = nu*chi1+x2*x2*chi2
    beta = 113/12*(x1*x1*chi1+x2*x2*chi2)+25/4*nu*(chi1+chi2)
    electric_spin = (bar_chi-2*x2*chi2)*spin_order
    chirp_tail = 4*math.pi*spin_order
    chirp_spin = (-beta-5*frame_spin/3)*spin_order
    result.update({
        "chi1":chi1,"chi2":chi2,"bar_chi":bar_chi,"frame_spin_coefficient":frame_spin,
        "barycentric_spin_orbit_beta":beta,
        "orbital_frequency_M_15pn":omega_local_M*(1+nu*y-frame_spin*spin_order),
        "separation_M_15pn":result["separation_M_newtonian"]*(1-(3+nu)*y/3-bar_chi*spin_order/3),
        "electric_spin_correction":electric_spin,
        "electric_amplitude_correction_15pn":result["electric_amplitude_correction"]+electric_spin,
        "epsilon_electric_15pn":result["epsilon_newtonian"]*(1+result["electric_amplitude_correction"]+electric_spin),
        "local_chirp_tail_correction":chirp_tail,"local_chirp_spin_correction":chirp_spin,
        "local_chirp_correction_15pn":result["local_chirp_correction"]+chirp_tail+chirp_spin,
        "chirp_M2_local_15pn":result["chirp_M2_newtonian"]*(1+result["local_chirp_correction"]+chirp_tail+chirp_spin),
        "coupling_squared_over_chirp_correction_15pn":result["coupling_squared_over_chirp_correction"]+2*electric_spin-chirp_tail-chirp_spin,
        "omitted_relative_pn_order":y*y,
    })
    return result
