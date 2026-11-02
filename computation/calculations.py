"""Numerical functions for Kerr tidal couplings and circular resonance crossings."""

import cmath
import csv
from dataclasses import fields
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import sph_harm_y

from boson_ep.binary_tide import circular_local_tide_15pn
from boson_ep.crossing import propagate_crossing, rotating_crossing_hamiltonian
from boson_ep.models import KerrModeResult, KerrModeSettings, RelativisticTideSettings, State
from boson_ep.relativity import angular_mode_values, build_radial_ode_evaluator
from boson_ep.relativistic_tides import _hydrogenic_radial, _qbs_decay_rate

ROOT = Path(__file__).resolve().parent


def _state(label: str) -> State:
    n_value = int(label[0])
    l_value = int(label[1])
    m_value = int(label[2:])
    return State(n_value, l_value, m_value)

def _number(row: dict[str, Any], name: str, default: float = math.nan) -> float:
    value = row.get(name)
    return default if value is None else float(value)

def _mode_from_row(row: dict[str, Any]) -> KerrModeResult:
    return KerrModeResult(
        alpha=float(row["alpha"]),
        chi=float(row["chi"]),
        state=_state(str(row["state"])),
        frequency_M=complex(_number(row, "omega_real_M"), _number(row, "gamma_M")),
        separation_constant=complex(
            _number(row, "separation_real"), _number(row, "separation_imag")
        ),
        angular_l_values=np.asarray(row["angular_l_values"], dtype=int),
        angular_coefficients=np.asarray(
            [complex(*item) for item in row["angular_coefficients"]], dtype=complex
        ),
        radial_coefficients=np.asarray(
            [complex(*item) for item in row["radial_coefficients"]], dtype=complex
        ),
        bilinear_norm=complex(
            _number(row, "bilinear_norm_real"),
            _number(row, "bilinear_norm_imag"),
        ),
        r_99_M=_number(row, "r_99_M"),
        cf_residual=_number(row, "cf_residual", math.inf),
        radial_residual=_number(row, "radial_residual", math.inf),
        angular_residual=_number(row, "angular_residual", math.inf),
        converged=bool(row["converged"]),
        message=str(row.get("message", "")),
        selected_truncation=row.get("selected_truncation"),
        convergence_history=tuple(row.get("convergence_history", ())),
        radial_ode_residual=(
            None
            if row.get("radial_ode_residual") is None
            else float(row["radial_ode_residual"])
        ),
    )

def _load_settings(path: Path) -> RelativisticTideSettings:
    payload = json.loads(path.read_text(encoding="utf-8"))["settings"]
    mode_payload = dict(payload.pop("mode"))
    mode_names = {item.name for item in fields(KerrModeSettings)}
    tide_names = {item.name for item in fields(RelativisticTideSettings)}
    mode = KerrModeSettings(
        **{key: value for key, value in mode_payload.items() if key in mode_names}
    )
    return RelativisticTideSettings(
        mode=mode,
        **{key: value for key, value in payload.items() if key in tide_names},
    )

def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))

def load_mode_pair(point):
    row = next(row for row in read_csv(assets.ATLAS / "kernels.csv") if row["point_id"] == point)
    modes = []
    for path in sorted((assets.ATLAS / "cache/modes").glob("*.json")):
        payload = json.loads(path.read_text())["payload"]
        if (abs(float(payload["alpha"]) - float(row["alpha"])) < 1.0e-12
                and abs(float(payload["chi"]) - float(row["chi"])) < 1.0e-10
                and payload["state"] in (row["initial"], row["final"])):
            modes.append(atlas._mode_from_row(payload))
    if len(modes) != 2:
        raise ValueError(f"Expected two modes for {point}, found {len(modes)}")
    modes.sort(key=lambda mode: mode.state.label != row["initial"])
    return row, modes

def factors(initial, final, settings):
    alpha, chi = initial.alpha, initial.chi
    horizon = 1 + math.sqrt(1 - chi**2)
    decays = [_qbs_decay_rate(alpha, mode.frequency_M) for mode in (initial, final)]
    outer = horizon + settings.mode.outer_decay_lengths / min(decays)
    lower = horizon + settings.horizon_cutoffs[0]
    nodes, weights = np.polynomial.legendre.leggauss(settings.radial_nodes)
    radius = lower + (outer - lower) * (nodes + 1) / 2
    weights = weights * (outer - lower) / 2
    radial = [build_radial_ode_evaluator(
        alpha, chi, mode.state, mode.frequency_M, mode.separation_constant,
        mode.radial_coefficients, outer_decay_lengths=(outer - horizon) * 1.001 * decay,
    )(radius)[0] for mode, decay in zip((initial, final), decays)]
    norms_r = [math.sqrt(float(np.sum(weights * radius**2 * abs(value)**2))) for value in radial]
    overlap_r = np.sum(weights * radius**4 * radial[0] * radial[1])

    # Use the same hydrogenic integration domain and orders as the production reference.
    nodes_h, weights_h = np.polynomial.legendre.leggauss(180)
    outer_h = 35 * max(initial.state.n, final.state.n)**2 / alpha**2
    radius_h = outer_h * (nodes_h + 1) / 2
    overlap_h = np.sum(weights_h * outer_h / 2 * radius_h**4
                       * _hydrogenic_radial(alpha, initial.state, radius_h)
                       * _hydrogenic_radial(alpha, final.state, radius_h))
    u, w = np.polynomial.legendre.leggauss(settings.angular_nodes)
    theta = np.arccos(u)
    angular = [angular_mode_values(theta, mode.state, mode.separation_constant,
                                  mode.angular_l_values, mode.angular_coefficients)[0]
               for mode in (initial, final)]
    norms_a = [math.sqrt(float(2 * math.pi * np.sum(w * abs(value)**2))) for value in angular]
    overlap_a = np.sum(w * (1 - u**2) * angular[0] * angular[1])
    uh, wh = np.polynomial.legendre.leggauss(120)
    theta_h = np.arccos(uh)
    overlap_ah = np.sum(wh * (1 - uh**2)
                        * sph_harm_y(initial.state.l, initial.state.m, theta_h, 0.0)
                        * sph_harm_y(final.state.l, final.state.m, theta_h, 0.0))
    return complex(-3 * alpha * math.pi / 4 * overlap_h * overlap_ah), {
        "frequency": cmath.sqrt(initial.frequency_M * final.frequency_M) / alpha,
        "radial": overlap_r / (math.prod(norms_r) * overlap_h),
        "angular": overlap_a / (math.prod(norms_a) * overlap_ah),
        "normalization": 2 * alpha * math.prod(norms_r) * math.prod(norms_a),
    }

def kernel_functions(record):
    functions,checks = {},{}
    samples = record["samples"]
    for key in ("forward","reverse"):
        selected = [next(r for r in samples if r["frequency_ratio"]==1),samples[-1]]
        x = [0,*[r["frequency_ratio"] for r in selected]]
        y = [complex(*record["static"][key]),*[complex(*r[key]) for r in selected]]
        coefficients = np.polynomial.polynomial.polyfit(x,y,2)
        functions[key] = lambda factor,c=coefficients:np.polynomial.polynomial.polyval(factor,c)
        checks[key] = max(abs(functions[key](r["frequency_ratio"])/complex(*r[key])-1)
                          for r in samples if r["frequency_ratio"] not in x)
    if max(checks.values()) > 1e-7:
        raise AssertionError(f"Kernel frequency interpolation requires more samples: {checks}")
    return functions,checks

def physical_problem(record,q,variant):
    if variant not in ("matched","modulus","hydrogenic","frozen_modulus"):
        raise ValueError("Unknown crossing comparison")
    omega = record["Omega_res_M"]
    wi,wf = map(lambda z:complex(*z),record["mode_frequencies_M"])
    gamma = wi.imag-wf.imag
    orbit0 = circular_local_tide_15pn(omega,q,record["chi"])
    scale = max(gamma,math.sqrt(2*orbit0["chirp_M2_local_15pn"]))
    functions,checks = kernel_functions(record)
    row = next(r for r in assets._read_csv(assets.ATLAS/"kernels.csv") if r["point_id"]==record["point_id"])
    hydrogenic = complex(float(row["hydrogenic_kernel_M_real"]),float(row["hydrogenic_kernel_M_imag"]))

    def couplings(factor,orbit):
        if variant=="hydrogenic":
            return (orbit["epsilon_electric_15pn"]*hydrogenic,)*2
        values = [orbit["epsilon_newtonian"]*(functions[key](factor)
                  +orbit["electric_amplitude_correction_15pn"]*complex(*record["static"][key]))
                  for key in ("forward","reverse")]
        if variant in ("modulus","frozen_modulus"):
            values[1] = values[0].conjugate()
        return values

    def orbit_at(x):
        current = omega+scale*x/2
        return current,circular_local_tide_15pn(current,q,record["chi"])

    def hamiltonian(x):
        current,orbit = orbit_at(x)
        forward,reverse = couplings(1,orbit0) if variant=="frozen_modulus" else couplings(current/omega,orbit)
        return rotating_crossing_hamiltonian(scale*x,gamma,forward,reverse)

    def dt_dx(x):
        orbit = orbit0 if variant=="frozen_modulus" else orbit_at(x)[1]
        return scale/(2*orbit["chirp_M2_local_15pn"])

    forward,reverse = couplings(1,orbit0)
    return hamiltonian,dt_dx,scale,{"gamma_M":gamma,"Gamma_over_sqrt_sweep":gamma/math.sqrt(2*orbit0["chirp_M2_local_15pn"]),
                                   "Gamma_over_gap":gamma/(2*omega),"eta_forward_over_Gamma":abs(forward)/gamma,
                                   "lz_reference_survival":math.exp(-2*math.pi*abs(forward)**2/(2*orbit0["chirp_M2_local_15pn"])),
                                   "kernel_interpolation_checks":checks}

def quartic_coefficients(higher):
    selected = [higher["samples"][i] for i in (0, 2, 3)]
    coefficients, checks = {}, {}
    for key in ("forward", "reverse"):
        coefficients[key] = np.polynomial.polynomial.polyfit(
            [s["frequency_ratio"] for s in selected], [complex(*s[key]) for s in selected], 2)
        probe = higher["samples"][1]
        checks[key] = abs(np.polynomial.polynomial.polyval(probe["frequency_ratio"], coefficients[key])
                          / complex(*probe[key]) - 1)
    if max(checks.values()) > 1e-6:
        raise AssertionError("Electric l=4 interpolation failed")
    return coefficients, checks

def problem(record, coefficients, kh4, q, variant):
    if variant not in ("matched", "hydrogenic", "modulus"):
        raise ValueError("Unknown crossing comparison")
    h0, dt_dx, scale, metadata = physical_problem(
        record, q, "hydrogenic" if variant == "hydrogenic" else "matched")
    omega = record["Omega_res_M"]

    def h(x):
        matrix = h0(x).copy()
        current = omega + scale*x/2
        orbit = circular_local_tide_15pn(current, q, record["chi"])
        for key, i, j in (("forward", 1, 0), ("reverse", 0, 1)):
            value = kh4 if variant == "hydrogenic" else np.polynomial.polynomial.polyval(
                current/omega, coefficients[key])
            matrix[i, j] += orbit["epsilon_newtonian"]*value/orbit["separation_M_newtonian"]**2
        if variant == "modulus":
            matrix[0, 1] = matrix[1, 0].conjugate()
        return matrix

    chirp = circular_local_tide_15pn(omega, q, record["chi"])["chirp_M2_local_15pn"]
    metadata["lz_reference_survival"] = math.exp(-2*math.pi*abs(h(0)[1, 0])**2/(2*chirp))
    metadata["eta_forward_over_Gamma"] = abs(h(0)[1, 0])/metadata["gamma_M"]
    return h, dt_dx, scale, metadata

def upper_endpoint(record, q):
    return brentq(lambda f: record["r99_M"]/circular_local_tide_15pn(
        f*record["Omega_res_M"], q, record["chi"])["separation_M_15pn"] - .1, 1, 2)

def run_case(record, coefficients, kh4, q, variant, lower, upper, *, refine=False, inspect=False):
    h, dt_dx, scale, metadata = problem(record, coefficients, kh4, q, variant)
    omega = record["Omega_res_M"]
    left, right = [2*omega*(f-1)/scale for f in (lower, upper)]
    samples = None
    if inspect:
        samples = np.unique(np.concatenate((
            np.sinh(np.linspace(np.arcsinh(left), np.arcsinh(right), 1201)),
            np.linspace(max(left, -3), min(right, 3), 481), [left, right, 0])))
        samples = np.clip(samples, left, right)
    result = propagate_crossing(h, (left, right), dt_dx=dt_dx,
        rtol=2e-10 if refine else 2e-8, atol=2e-15 if refine else 2e-13, samples=samples,
        initial_exponential_span=min(.001*(right-left), 40/(metadata["gamma_M"]*dt_dx(left))))
    rho = result.pop("density")
    result.update({"variant": variant, "q": q, "frequency_bounds": [lower, upper], **metadata,
                   "density_real": rho.real.tolist(), "density_imag": rho.imag.tolist()})
    if result["norm_balance_residual"] > 1e-8 or result["minimum_density_eigenvalue"] < -1e-9:
        raise AssertionError("Crossing failed density balance or positivity")
    if inspect:
        torque = []
        for x, density in zip(result.pop("x"), result.pop("density_samples"), strict=True):
            current = omega + scale*x/2
            orbit = circular_local_tide_15pn(current, q, record["chi"])
            flow = abs(2*(h(x)[0, 1]*density[1, 0]).imag)
            orbital_l = q/(1+q)**(1/3)*current**(-1/3)
            gw_torque = orbital_l*orbit["chirp_M2_local_15pn"]/(3*current)
            torque.append(2/record["alpha"]*flow/gw_torque)
        result["sampled_torque_per_cloud_fraction"] = max(torque)
        result["torque_sample_count"] = len(torque)
        result["torque_sampling_relative_change"] = abs(max(torque)/max(torque[::2])-1)
        if result["torque_sampling_relative_change"] > .01:
            raise AssertionError("The sampled transfer torque requires a finer grid")
        result["sampled_cloud_fraction_for_ten_percent_torque"] = .1/max(torque)
    return result

def feedback_bound(record, coefficients4, q, lower, upper):
    omega, alpha = record["Omega_res_M"], record["alpha"]
    lo, hi = lower*omega, upper*omega
    gamma = complex(*record["mode_frequencies_M"][0]).imag - complex(*record["mode_frequencies_M"][1]).imag
    functions, _ = kernel_functions(record)
    coefficients = {key: np.polynomial.polynomial.polyfit([0, .5, 1], [f(x) for x in (0, .5, 1)], 2)
                    for key, f in functions.items()}
    high = circular_local_tide_15pn(hi, q, record["chi"])
    pn = abs(high["electric_amplitude_correction"]) + abs(high["electric_spin_correction"])
    chirp_min = 1 - abs(high["local_chirp_correction"]) - abs(
        high["local_chirp_tail_correction"] + high["local_chirp_spin_correction"])
    if chirp_min <= 0 or gamma <= 0 or not 0 < lo < hi:
        raise ValueError("Positive decay, chirp and frequency bounds are required")
    powers, inv_b2 = upper**np.arange(3), high["separation_M_newtonian"]**-2
    reverse = sum(abs(coefficients["reverse"])*powers) + pn*abs(complex(*record["static"]["reverse"]))
    reverse += inv_b2*sum(abs(coefficients4["reverse"])*powers)
    difference = sum(abs(coefficients["reverse"]-coefficients["forward"].conj())*powers)
    difference += pn*abs(complex(*record["static"]["reverse"])-complex(*record["static"]["forward"]).conjugate())
    difference += inv_b2*sum(abs(coefficients4["reverse"]-coefficients4["forward"].conj())*powers)
    b, a = q/(1+q), 96/5*q/(1+q)**(1/3)
    log_norm = b*b*difference*difference/(2*gamma*a*chirp_min)*3/4*(hi**(4/3)-lo**(4/3))
    ratio = 6*(1+q)**(1/3)/(alpha*q)*b*reverse/(a*chirp_min)*lo**(-1/3)*math.exp(log_norm)
    # Polynomial coefficient bounds apply continuously; samples check their implementation.
    for current in np.geomspace(lo, hi, 501):
        orbit = circular_local_tide_15pn(current, q, record["chi"])
        values = {key: functions[key](current/omega)
            + orbit["electric_amplitude_correction_15pn"]*complex(*record["static"][key])
            + np.polynomial.polynomial.polyval(current/omega, coefficients4[key])
              / orbit["separation_M_newtonian"]**2 for key in functions}
        if (abs(values["reverse"]) > reverse*(1+1e-12)
                or abs(values["reverse"]-values["forward"].conjugate()) > difference*(1+1e-10)
                or orbit["chirp_M2_local_15pn"] < a*current**(11/3)*chirp_min*(1-1e-12)):
            raise AssertionError("Continuous coefficient bound failed its sample check")
    return {"reverse_kernel_bound": float(reverse), "asymmetry_kernel_bound": float(difference),
            "chirp_fraction_lower_bound": chirp_min, "norm_log_bound": log_norm,
            "torque_ratio_bound_per_cloud_fraction": float(ratio),
            "cloud_fraction_for_ten_percent_bound": float(.1/ratio)}

def assess(record, higher, previous):
    coefficients, checks = quartic_coefficients(higher)
    kh4, q0 = complex(*previous["hydrogenic_l4_kernel"]), previous["q"]
    evaluations = []

    def evaluate(q, refine=False, inspect=False):
        upper = upper_endpoint(record, q)
        runs = [run_case(record, coefficients, kh4, q, variant, .05, upper,
                         refine=refine, inspect=inspect and variant == "matched")
                for variant in ("matched", "hydrogenic")]
        value = {"q": q, "upper_frequency_ratio": upper, "runs": runs,
                 "depletion_difference": runs[0]["initial_survival"]-runs[1]["initial_survival"]}
        evaluations.append({"q": q, "depletion_difference": value["depletion_difference"]})
        return value

    original = evaluate(q0, refine=True)
    old_runs = {r["variant"]: r for r in previous["runs"]}
    preservation = [abs(new["initial_survival"]-old_runs[name]["initial_survival"])
                    for new, name in zip(original["runs"], ("add_l4", "hydrogenic_add_l4"), strict=True)]
    if max(preservation) > 5e-10:
        raise AssertionError("The archived E4 crossing is not reproduced")
    optimum = minimize_scalar(lambda logq: -evaluate(math.exp(logq))["depletion_difference"],
        bounds=(math.log(q0/2), math.log(q0*2)), method="bounded", options={"xatol": 2e-4, "maxiter": 24})
    q = math.exp(optimum.x)
    if not optimum.success or not .55*q0 < q < 1.9*q0:
        raise AssertionError("The bounded maximum search needs review")
    coarse, final = evaluate(q), evaluate(q, refine=True, inspect=True)
    errors = [abs(c["initial_survival"]-f["initial_survival"])
              for c, f in zip(coarse["runs"], final["runs"], strict=True)]
    if max(errors) > 2e-7:
        raise AssertionError("Crossing tolerance refinement failed")
    neighbors = [evaluate(q*factor) for factor in (.98, 1.02)]
    if any(n["depletion_difference"] >= final["depletion_difference"] for n in neighbors):
        raise AssertionError("Neighboring crossings do not bracket the maximum")
    upper = final["upper_frequency_ratio"]
    endpoint = run_case(record, coefficients, kh4, q, "matched", .1, upper, refine=True)
    short_upper = min(1.2, upper-.05)
    short = run_case(record, coefficients, kh4, q, "matched", .05, short_upper, refine=True)
    conjugate = run_case(record, coefficients, kh4, q, "modulus", .05, upper, refine=True)
    h, _, scale, metadata = problem(record, coefficients, kh4, q, "matched")
    omega = record["Omega_res_M"]
    orbit0 = circular_local_tide_15pn(omega, q, record["chi"])
    variations = [{"frequency_ratio": f, "coupling_ratio": abs(h(2*omega*(f-1)/scale)[1, 0]/h(0)[1, 0]),
                   "chirp_ratio": circular_local_tide_15pn(f*omega, q, record["chi"])["chirp_M2_local_15pn"]
                                  / orbit0["chirp_M2_local_15pn"]}
                  for f in (1-metadata["Gamma_over_gap"], 1+metadata["Gamma_over_gap"])]
    bound = feedback_bound(record, coefficients, q, .05, upper)
    if final["runs"][0]["sampled_torque_per_cloud_fraction"] > bound["torque_ratio_bound_per_cloud_fraction"]:
        raise AssertionError("Sampled transfer exceeds the continuous estimate")
    return {"point_id": record["point_id"], "alpha": record["alpha"],
            "original_q_result": original, "archive_survival_displacements": preservation,
            "quartic_interpolation_checks": checks, "local_maximum": final,
            "optimization_evaluations": evaluations, "neighbor_checks": neighbors,
            "solver_refinement_survival_displacements": errors,
            "later_start": endpoint, "earlier_stop": short, "conjugate_reverse": conjugate,
            "one_width_variations": variations, "continuous_feedback_bound": bound}

def complete_kernel(record, q, *, hydrogenic=False):
    orbit = circular_local_tide_15pn(record["Omega_M"],q,record["chi"])
    if hydrogenic:
        base = complex(*record["hydrogenic_quadrupole"])
        return (base*(1+orbit["electric_amplitude_correction_15pn"])
                +complex(*record["hydrogenic_l4"])/orbit["separation_M_newtonian"]**2)
    return (complex(*record["matched"])
            +orbit["electric_amplitude_correction_15pn"]*complex(*record["static"])
            +complex(*record["electric_l4"])/orbit["separation_M_newtonian"]**2)

atlas = SimpleNamespace(_mode_from_row=_mode_from_row)
assets = SimpleNamespace(ATLAS=ROOT/"data/atlas", _read_csv=read_csv)
