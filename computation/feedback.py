"""Physical scalar currents, tidal torque and secular absorption response."""

import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.integrate import simpson

from boson_ep.binary_tidal_metric import CircularBinaryTide
from boson_ep.binary_tide import circular_local_tide_15pn
from boson_ep.crossing import propagate_crossing
from boson_ep.higher_tide import ElectricHexadecapoleTide
from boson_ep.models import CFSettings
from boson_ep.orbital_feedback import circular_angular_momentum_15pn, eigenfrequency_drift, metric_energy_flux_operator, propagate_feedback
from boson_ep.relativity import solve_quasibound_cf
from boson_ep.scalar_balance import ingoing_radial_provider
from boson_ep.scalar_torque import advanced_tide_to_ingoing, background_angular_operators, background_angular_outer_flux, background_energy_operators, metric_angular_operator, stress_overlaps, torque_entry_bounds, torque_operator
from boson_ep.tidal_metric import irg_metric_coefficient
from calculations import feedback_bound, kernel_functions, load_mode_pair, problem, quartic_coefficients

ROOT = Path(__file__).resolve().parent


def read_json(name):
    return json.loads((ROOT/"data"/name).read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def inputs(index):
    return [read_json(name)[index] for name in (
        "directed_couplings.json", "electric_l4.json", "crossing_seeds.json",
        "crossings.json", "scalar_currents.json")]


def complex_array(values):
    array = np.asarray(values)
    return array[..., 0] + 1j * array[..., 1]

def interpolants(index):
    upper = 1.4 if index == 0 else 1.2
    archive = {'samples': read_json('stress_overlaps.json')}
    point = ('a0.200000_21+1_21-1', 'a0.400000_32+2_32+0')[index]
    sample = lambda f, fine: next((s for s in archive['samples'] if s['point_id'] == point and s['frequency_ratio'] == f and (s['refined'] == fine)))
    samples = [sample(f, False) for f in (0.05, 0.6, upper)]
    independent = sample(1.0, True)
    fit = {}
    errors = {}
    for field in ('overlaps', 'boundary_overlaps'):
        offsets = (None,) if field == 'overlaps' else ('0.001', '0.0005', 'outer')
        for offset in offsets:
            for key in ('matched', 'static', 'quartic'):

                def value(row):
                    source = row[field] if offset is None else row[field][offset]
                    return complex_array(source[key])
                coefficients = np.polynomial.polynomial.polyfit([s['frequency_ratio'] for s in samples], [value(s) for s in samples], 2)
                label = (field, offset, key)
                fit[label] = coefficients
                reference = value(sample(1.0, False)) if offset == 'outer' else value(independent)
                errors[str(label)] = float(np.linalg.norm(np.polynomial.polynomial.polyval(1.0, coefficients) - reference) / max(np.linalg.norm(reference), 1e-30))
    if max((v for k, v in errors.items() if 'boundary' not in k)) > 2e-07:
        raise ValueError('Stress overlap interpolation needs additional frequency samples')
    return (fit, sample(1.0, False), errors)

def physical_torque_bound(record, quartic, fits, q, lower, upper, initial_energy):
    """Continuous triangle bound within the interpolated scalar-field model."""
    omega = record['Omega_res_M']
    lo, hi = (lower * omega, upper * omega)
    orbit = circular_local_tide_15pn(hi, q, record['chi'])
    powers = upper ** np.arange(3)
    pn = abs(orbit['electric_amplitude_correction']) + abs(orbit['electric_spin_correction'])
    invb2 = orbit['separation_M_newtonian'] ** (-2)
    functions, _ = kernel_functions(record)
    directed = {}
    for key in ('forward', 'reverse'):
        c = np.polynomial.polynomial.polyfit([0, 0.5, 1], [functions[key](x) for x in (0, 0.5, 1)], 2)
        directed[key] = float(np.dot(abs(c), powers) + pn * abs(complex(*record['static'][key])) + invb2 * np.dot(abs(quartic[key]), powers))
    component = lambda key: np.einsum('k,kj->j', powers, abs(fits['overlaps', None, key]))
    a, b, c, d = component('matched') + pn * component('static') + invb2 * component('quartic')
    wi, wf = [complex(*w) for w in record['mode_frequencies_M']]
    gamma = wi.imag - wf.imag
    forward = orbit['epsilon_newtonian'] * directed['forward']
    reverse = orbit['epsilon_newtonian'] * directed['reverse']
    r = forward / gamma
    entries = torque_entry_bounds([a, b, c, d], [wi, wf], forward, reverse)
    k00, k11, k10 = (entries[0, 0], entries[1, 1], entries[1, 0])
    existing = feedback_bound(record, quartic, q, lower, upper)
    norm = np.exp(existing['norm_log_bound'])
    coefficient = norm / initial_energy * (k00 + k11 * min(1, r * r) + 2 * k10 * min(0.5, r))
    b_orbit, chirp = (q / (1 + q), 96 / 5 * q / (1 + q) ** (1 / 3))
    epsilon_over_gw = 3 * b_orbit * (1 + q) ** (1 / 3) / (q * chirp * existing['chirp_fraction_lower_bound']) * lo ** (-1 / 3)
    ratio = coefficient * epsilon_over_gw
    return {'scope': 'All frequencies and all evolved states in the retained interpolated two-mode model, with ci=1, cf=0 initially.', 'gw_reference': 'Leading circular angular momentum differentiated along the archived local 1.5PN chirp; PN corrections to orbital angular momentum remain a separate check.', 'torque_ratio_bound_per_cloud_fraction': float(ratio), 'cloud_fraction_for_ten_percent_bound': float(0.1 / ratio), 'cloud_fraction_for_one_percent_bound': float(0.01 / ratio), 'final_amplitude_over_sqrt_norm_bound': float(r), 'modal_norm_bound': float(norm), 'unit_epsilon_stress_overlap_bounds': [float(v) for v in (a, b, c, d)], 'directed_kernel_bounds': directed, 'chirp_fraction_lower_bound': existing['chirp_fraction_lower_bound']}

def assess(index, record, higher, earlier, final, normalization):
    fits, reference, interpolation = interpolants(index)
    _, modes = load_mode_pair(record['point_id'])
    ms = [m.state.m for m in modes]
    frequencies = np.array([m.frequency_M for m in modes])
    frequencies -= 1j * frequencies[0].imag
    q = final['local_maximum']['q']
    lower, upper = (0.05, final['local_maximum']['upper_frequency_ratio'])
    omega = record['Omega_res_M']
    coefficients, _ = quartic_coefficients(higher)
    kh4 = complex(*earlier['hydrogenic_l4_kernel'])
    h, dt_dx, scale, metadata = problem(record, coefficients, kh4, q, 'matched')
    weights = reference['time_weights']
    radial = reference['radial_current']
    horizon = reference['horizon_amplitudes']
    initial_energy = normalization['modes'][0]['charges']['energy']
    energies = [m['charges']['energy'] for m in normalization['modes']]
    j0, fh0 = background_angular_operators(weights, radial, horizon, ms, record['chi'], np.diag(frequencies))
    _, eh0 = background_energy_operators(weights, energies, frequencies, horizon, ms, record['chi'], np.diag(frequencies))
    numerical_threshold_j, numerical_threshold_e = (fh0[0, 0], eh0[0, 0])
    fh0[0, 0] = 0
    eh0[0, 0] = 0
    outer0 = background_angular_outer_flux(reference['outer_amplitudes'], reference['outer_radial_flux'], ms, record['chi'], reference['outer_radius'], np.diag(frequencies))

    def operators(current):
        x = 2 * (current - omega) / scale
        orbit = circular_local_tide_15pn(current, q, record['chi'])
        hrot = h(x)
        temporal = hrot + np.diag([frequencies[0], frequencies[0] - 2 * current])

        def combined(field, offset=None):
            value = {key: np.polynomial.polynomial.polyval(current / omega, fits[field, offset, key]) for key in ('matched', 'static', 'quartic')}
            return orbit['epsilon_newtonian'] * (value['matched'] + orbit['electric_amplitude_correction_15pn'] * value['static'] + value['quartic'] / orbit['separation_M_newtonian'] ** 2)
        overlap = combined('overlaps')
        torque = torque_operator(overlap, temporal)
        charge, flux = background_angular_operators(weights, radial, horizon, ms, record['chi'], temporal)
        flux[0, 0] -= numerical_threshold_j
        _, energy_flux = background_energy_operators(weights, energies, frequencies, horizon, ms, record['chi'], temporal)
        energy_flux[0, 0] -= numerical_threshold_e
        correction = charge - j0 + metric_angular_operator(overlap, temporal, ms)
        caps = [metric_angular_operator(combined('boundary_overlaps', offset), temporal, ms, inward_flux=True) for offset in ('0.001', '0.0005')]
        flux += 2 * caps[1] - caps[0]
        outer_flux = background_angular_outer_flux(reference['outer_amplitudes'], reference['outer_radial_flux'], ms, record['chi'], reference['outer_radius'], temporal)
        outer_flux += metric_angular_operator(combined('boundary_overlaps', 'outer'), temporal, ms)
        return (torque, correction, flux, hrot, orbit, max((np.linalg.norm(c) for c in caps)), energy_flux, outer_flux)
    left, right = [2 * omega * (factor - 1) / scale for factor in (lower, upper)]
    midpoint = 2400
    us = np.r_[np.linspace(np.arcsinh(left), 0, midpoint + 1)[:-1], np.linspace(0, np.arcsinh(right), midpoint + 1)]
    xs = np.sinh(us)
    xs[0], xs[-1] = (left, right)
    evolution = propagate_crossing(h, (left, right), dt_dx=dt_dx, rtol=2e-10, atol=2e-15, samples=xs, initial_exponential_span=min(0.001 * (right - left), 40 / (metadata['gamma_M'] * dt_dx(left))))
    values = []
    for x, rho in zip(xs, evolution['density_samples'], strict=True):
        current = omega + scale * x / 2
        torque, correction, horizon_flux, hrot, orbit, cap_norm, energy_flux, outer_flux = operators(current)
        charge = j0 + correction
        step = current * 0.0001
        derivative = (operators(current + step)[1] - operators(current - step)[1]) / (2 * step)
        derivative *= orbit['chirp_M2_local_15pn']
        rate = 2 * frequencies[0].imag * charge + 1j * (hrot.conj().T @ charge - charge @ hrot) + derivative
        hfree = np.diag(np.diag(hrot))
        baseline = 2 * frequencies[0].imag * j0 + 1j * (hfree.conj().T @ j0 - j0 @ hfree) + fh0 + outer0
        defect = rate + horizon_flux + outer_flux - torque - baseline
        expect = lambda matrix: float(np.trace(matrix @ rho).real / initial_energy)
        orbital_l = q / (1 + q) ** (1 / 3) * current ** (-1 / 3)
        gw = orbital_l * orbit['chirp_M2_local_15pn'] / (3 * current)
        values.append([current / omega, expect(torque), expect(defect), expect(horizon_flux), abs(expect(torque)) / gw, abs(expect(defect)) / gw, cap_norm / initial_energy / gw, expect(correction), float(np.trace(rho).real), expect(energy_flux), expect(charge), expect(rate), expect(baseline), expect(outer_flux)])
    values = np.array(values)
    peak = np.argmax(values[:, 4])
    time_weights = np.array([dt_dx(x) for x in xs]) * np.cosh(us)

    def integrate(data, stride=1):
        return float(simpson(data[:midpoint + 1:stride], x=us[:midpoint + 1:stride]) + simpson(data[midpoint::stride], x=us[midpoint::stride]))
    integral = lambda column: integrate(values[:, column] * time_weights)
    absolute_integral = lambda column: integrate(abs(values[:, column]) * time_weights)
    prior_survival = final['local_maximum']['runs'][0]['initial_survival']
    if abs(evolution['initial_survival'] - prior_survival) > 1e-09:
        raise ValueError('The prescribed crossing no longer matches its archived result')
    result = {'point_id': record['point_id'], 'q': q, 'frequency_bounds': [lower, upper], 'initial_survival': evolution['initial_survival'], 'interpolation_errors': interpolation, 'sample_count': len(xs), 'physical_torque_peak_per_cloud_fraction': values[peak, 4], 'peak_frequency_ratio': values[peak, 0], 'cloud_fraction_for_ten_percent_sampled_torque': 0.1 / values[peak, 4], 'integrated_cloud_torque_per_cloud_fraction': integral(1), 'integrated_horizon_angular_flux_per_cloud_fraction': integral(3), 'integrated_kerr_horizon_energy_flux_per_cloud_fraction': integral(9), 'cloud_angular_momentum_change_per_cloud_fraction': values[-1, 10] - values[0, 10], 'charge_integral_endpoint_error_relative_to_torque': abs(integral(11) - (values[-1, 10] - values[0, 10])) / absolute_integral(1), 'isolated_baseline_residual_integral_per_cloud_fraction': integral(12), 'absolute_outer_angular_flux_over_absolute_torque_integral': absolute_integral(13) / absolute_integral(1), 'integrated_balance_defect_per_cloud_fraction': integral(2), 'absolute_defect_over_absolute_torque_integral': absolute_integral(2) / absolute_integral(1), 'max_defect_over_gw_per_cloud_fraction': max(values[:, 5]), 'max_horizon_metric_cap_operator_over_gw_per_cloud_fraction': max(values[:, 6]), 'max_absolute_metric_and_envelope_charge_correction_per_cloud_fraction': max(abs(values[:, 7])), 'baseline_residual_subtracted': True, 'saturated_initial_flux_set_to_zero': True, 'baseline_residual_reason': 'Finite precision isolated-mode flux residuals were verified separately and are removed to resolve the tidal contribution.', 'scope': 'Physical stress torque and current-balance defect of a slowly varying two-mode reconstruction on T=v-r slices; no orbit feedback has been evolved.'}
    result['continuous_physical_torque_bound'] = physical_torque_bound(record, coefficients, fits, q, lower, upper, initial_energy)
    if result['physical_torque_peak_per_cloud_fraction'] > result['continuous_physical_torque_bound']['torque_ratio_bound_per_cloud_fraction']:
        raise ValueError('The sampled physical torque exceeds its continuous bound')
    bound = result['continuous_physical_torque_bound']
    if max(values[:, 8]) > bound['modal_norm_bound'] * (1 + 2e-09) or max(evolution['density_samples'][:, 1, 1].real) > bound['final_amplitude_over_sqrt_norm_bound'] ** 2 * bound['modal_norm_bound'] * (1 + 2e-09):
        raise ValueError('The modal norm or final amplitude exceeds its continuous bound')
    selection = np.unique(np.r_[np.arange(0, len(xs), 2), len(xs) - 1])
    result['sampling_refinement'] = {'peak_torque_relative': abs(max(values[selection, 4]) / values[peak, 4] - 1), 'integrated_torque_relative': abs(integrate(values[:, 1] * time_weights, stride=2) / integral(1) - 1), 'integrated_defect_absolute_per_cloud_fraction': abs(integrate(values[:, 2] * time_weights, stride=2) - integral(2))}
    if result['sampling_refinement']['peak_torque_relative'] > 0.0001 or result['sampling_refinement']['integrated_torque_relative'] > 1e-05 or result['charge_integral_endpoint_error_relative_to_torque'] > 1e-05:
        raise ValueError('Physical torque sampling needs refinement')
    result['samples'] = [dict(zip(('frequency_ratio', 'cloud_torque_per_fc', 'balance_defect_per_fc', 'horizon_flux_per_fc', 'torque_over_gw_per_fc', 'defect_over_gw_per_fc', 'cap_norm_over_gw_per_fc', 'charge_correction_per_fc', 'modal_norm', 'kerr_horizon_energy_flux_per_fc', 'angular_charge_per_fc', 'angular_charge_rate_per_fc', 'isolated_baseline_residual_per_fc', 'outer_angular_flux_per_fc'), row.tolist(), strict=True)) for row in values[::max(1, len(values) // 16)]]
    return result

def derivatives(mode):
    settings = CFSettings(truncation=mode.selected_truncation, angular_lmax=14)
    result = {'state': mode.state.label, 'frequency': [mode.frequency_M.real, mode.frequency_M.imag], 'samples': {}, 'derivatives': {}, 'refinement': {}}
    for parameter, step in (('alpha', 0.0001), ('chi', 0.0001)):
        evaluations = []
        slopes = []
        for width in (step, step / 2):
            values = []
            for sign in (-1, 1):
                alpha = mode.alpha + sign * width if parameter == 'alpha' else mode.alpha
                chi = mode.chi + sign * width if parameter == 'chi' else mode.chi
                solved = solve_quasibound_cf(alpha, chi, mode.state, settings, mode.frequency_M)
                if not solved.converged:
                    raise ValueError(solved.message)
                values.append(solved.frequency_M)
                evaluations.append({'displacement': sign * width, 'frequency': [solved.frequency_M.real, solved.frequency_M.imag], 'cf_residual': solved.residual})
            slopes.append((values[1] - values[0]) / (2 * width))
        extrapolated = (4 * slopes[1] - slopes[0]) / 3
        result['samples'][parameter] = evaluations
        result['derivatives'][parameter] = [extrapolated.real, extrapolated.imag]
        result['refinement'][parameter] = {'real_absolute': abs(slopes[1].real - slopes[0].real), 'imag_absolute': abs(slopes[1].imag - slopes[0].imag)}
        if abs(slopes[1].real - slopes[0].real) > 3e-08 or abs(slopes[1].imag - slopes[0].imag) > 0.0001 * abs(extrapolated.imag):
            raise ValueError('Spectral derivative failed step refinement')
    return result

def calculate(index, frequency_ratio, refine=False):
    start = time.monotonic()
    records = read_json('directed_couplings.json')
    record = records[index]
    _, modes = load_mode_pair(record['point_id'])
    omega = record['Omega_res_M'] * frequency_ratio
    lengths = 32 if refine else 24
    outer = max((ingoing_radial_provider(m, lengths).outer_radius for m in modes)) * 1.01
    matched = CircularBinaryTide(record['chi'], omega, outer)
    quartic = ElectricHexadecapoleTide(record['chi'], omega, outer)
    metrics = {'matched': lambda r, t: advanced_tide_to_ingoing(matched.metric(r, t), r, omega), 'static': lambda r, t: advanced_tide_to_ingoing(irg_metric_coefficient(r, t, record['chi'], 0), r, omega), 'quartic': lambda r, t: advanced_tide_to_ingoing(quartic.metric(r, t), r, omega)}
    result = stress_overlaps(modes, metrics, radial_nodes=28 if refine else 20, angular_nodes=24 if refine else 16, outer_decay_lengths=lengths)
    temporal = np.diag([m.frequency_M for m in modes])
    result['leading_torque_coefficients'] = {key: [torque_operator(value, temporal)[1, 0].real, torque_operator(value, temporal)[1, 0].imag] for key, value in result['overlaps'].items()}
    result['overlaps'] = {key: [[z.real, z.imag] for z in value] for key, value in result['overlaps'].items()}
    result['boundary_overlaps'] = {offset: {key: [[z.real, z.imag] for z in value] for key, value in fields.items()} for offset, fields in result['boundary_overlaps'].items()}
    for key in ('time_weights', 'radial_current'):
        result[key] = result[key].tolist()
    result.update({'point_id': record['point_id'], 'frequency_ratio': frequency_ratio, 'refined': refine, 'seconds': time.monotonic() - start, 'scope': 'Stress overlaps; not yet a closed orbital-feedback calculation.'})
    name = f'point{index}-f{frequency_ratio:g}' + ('-fine' if refine else '')
    write_json(ROOT / 'output' / (name + '.json'), result)
    print(json.dumps({'point_id': result['point_id'], 'frequency_ratio': frequency_ratio, 'refined': refine, 'seconds': result['seconds']}), flush=True)
    return result

def make_model(index, fc, variant, absorption):
    record, higher, earlier, final, normalization = inputs(index)
    fits, reference, _ = interpolants(index)
    spectrum = {'pairs': read_json('spectral_response.json')}
    derivatives = spectrum['pairs'][index]
    da = np.array([complex(*m['derivatives']['alpha']) for m in derivatives['modes']])
    dc = np.array([complex(*m['derivatives']['chi']) for m in derivatives['modes']])
    _, modes = load_mode_pair(record['point_id'])
    ms = [m.state.m for m in modes]
    frequencies = np.array([m.frequency_M for m in modes])
    frequencies -= 1j * frequencies[0].imag
    q = final['local_maximum']['q']
    omega = record['Omega_res_M']
    coeff4, _ = quartic_coefficients(higher)
    h0, dt_dx, scale, metadata = problem(record, coeff4, complex(*earlier['hydrogenic_l4_kernel']), q, variant)
    energies = [m['charges']['energy'] for m in normalization['modes']]
    numbers = np.array([m['charges']['number'] for m in normalization['modes']])
    number = numbers[0]
    _, jflux0 = background_angular_operators(reference['time_weights'], reference['radial_current'], reference['horizon_amplitudes'], ms, record['chi'], np.diag(frequencies))
    _, eflux0 = background_energy_operators(reference['time_weights'], energies, frequencies, reference['horizon_amplitudes'], ms, record['chi'], np.diag(frequencies))

    def model(x, absorbed_energy, absorbed_angular):
        current = omega + scale * x / 2
        dm, dj = (fc * absorbed_energy, fc * absorbed_angular) if absorption else (0.0, 0.0)
        drift = eigenfrequency_drift(record['alpha'], record['chi'], frequencies, da, dc, dm, dj)
        h = h0(x).copy()
        h += np.diag(drift)
        temporal = h0(x) + np.diag([frequencies[0], frequencies[0] - 2 * current])
        orbit = circular_angular_momentum_15pn(current, q, record['chi'])
        mass_derivative = orbit['mass_derivative_at_fixed_spin']
        spin_derivative = orbit['spin_derivative_at_fixed_mass']
        orbit['mass_derivative_at_fixed_spin'] = 0.0
        orbit['spin_derivative_at_fixed_mass'] = 0.0
        tide = circular_local_tide_15pn(current, q, record['chi'])

        def overlap(field, offset=None):
            values = {key: np.polynomial.polynomial.polyval(current / omega, fits[field, offset, key]) for key in ('matched', 'static', 'quartic')}
            return tide['epsilon_newtonian'] * (values['matched'] + tide['electric_amplitude_correction_15pn'] * values['static'] + values['quartic'] / tide['separation_M_newtonian'] ** 2)
        if variant == 'matched':
            torque = torque_operator(overlap('overlaps'), temporal)
            _, jf = background_angular_operators(reference['time_weights'], reference['radial_current'], reference['horizon_amplitudes'], ms, record['chi'], temporal)
            _, ef = background_energy_operators(reference['time_weights'], energies, frequencies, reference['horizon_amplitudes'], ms, record['chi'], temporal)
            jf[0, 0] -= jflux0[0, 0]
            ef[0, 0] -= eflux0[0, 0]
            boundary = [overlap('boundary_overlaps', offset) for offset in ('0.001', '0.0005')]
            cap = [metric_angular_operator(value, temporal, ms, inward_flux=True) for value in boundary]
            jf += 2 * cap[1] - cap[0]
            ecap = [metric_energy_flux_operator(value, temporal) for value in boundary]
            ef += 2 * ecap[1] - ecap[0]
            jf += np.diag(-2 * drift.imag * np.array(ms) * numbers)
            ef += np.diag(-2 * drift.imag * np.array(energies))
        else:
            torque = np.zeros((2, 2), complex)
            torque[1, 0] = 1j * (ms[0] - ms[1]) * number * h[1, 0]
            torque[0, 1] = torque[1, 0].conjugate()
            jf = np.diag(-2 * (frequencies + drift).imag * np.array(ms) * numbers)
            ef = np.diag(-2 * (frequencies + drift).imag * np.array(energies))
            ecap = [np.zeros((2, 2))]
        return {'hamiltonian': h, 'cloud_torque': torque / energies[0], 'horizon_energy': ef / energies[0], 'horizon_angular': jf / energies[0], 'orbit': orbit, 'frequency_drift': drift, 'composite_parameter_torque_correction_bound': abs(current * mass_derivative) + abs(spin_derivative), 'energy_cap_norm_per_fc': max((np.linalg.norm(c) for c in ecap)) / energies[0]}
    lower, upper = (0.05, final['local_maximum']['upper_frequency_ratio'])
    interval = tuple((2 * omega * (v - 1) / scale for v in (lower, upper)))
    span = min(0.001 * (interval[1] - interval[0]), 40 / (metadata['gamma_M'] * dt_dx(interval[0])))
    return (model, interval, scale, span, record, final)

def run(index, fc, variant, absorption=True, torque_feedback=True, refine=False):
    start = time.monotonic()
    model, interval, scale, span, record, final = make_model(index, fc, variant, absorption)
    left, right = interval
    count = 800 if refine else 400
    us = np.r_[np.linspace(np.arcsinh(left), 0, count + 1)[:-1], np.linspace(0, np.arcsinh(right), count + 1)]
    xs = np.sinh(us)
    xs[0], xs[-1] = (left, right)
    result = propagate_feedback(model, interval, scale, fc, torque_feedback=torque_feedback, initial_span=span * (0.5 if refine else 1), rtol=2e-10 if refine else 2e-09, atol=2e-15 if refine else 2e-14, samples=xs)
    states = result.pop('states')
    data = [model(x, y[6], y[7]) for x, y in zip(xs, states, strict=True)]
    shifts = np.array([d['frequency_drift'] for d in data])
    chirps = np.array([d['orbit']['gw_chirp'] for d in data]) * (1 + result['relative_chirp_feedback'])
    weights = scale / (2 * chirps) * np.cosh(us)
    integral = lambda y: float(simpson(y[:count + 1], x=us[:count + 1]) + simpson(y[count:], x=us[count:]))
    gamma = -complex(*record['mode_frequencies_M'][1]).imag
    result.update({'point_id': record['point_id'], 'cloud_fraction': fc, 'variant': variant, 'absorption_changes_background': absorption, 'torque_feedback': torque_feedback, 'refined': refine, 'scope': 'Leading tidal torque feedback on the circular 1.5PN orbit, with linear complex-frequency and isolated-flux response to absorption. Orbital background, mode profiles and tidal couplings are fixed at their initial values. BH absorption is an internal redistribution, not net central accretion.', 'mass_change': fc * result['absorbed_energy_per_cloud_fraction'], 'angular_momentum_change': fc * result['absorbed_angular_momentum_per_cloud_fraction'], 'max_resonance_detuning_shift_over_Gamma': float(max(abs((shifts[:, 0] - shifts[:, 1]).real)) / gamma), 'max_initial_width_change': float(max(abs(shifts[:, 0].imag))), 'integrated_initial_decay_exponent': integral(-2 * shifts[:, 0].imag * weights), 'max_initial_growth_rate': float(max(shifts[:, 0].imag)), 'max_final_width_change_over_Gamma': float(max(abs(shifts[:, 1].imag)) / gamma), 'max_relative_chirp_feedback': float(max(abs(result.pop('relative_chirp_feedback')))), 'minimum_pn_orbital_derivative_factor': min((d['orbit']['frequency_derivative_over_leading'] for d in data)), 'composite_parameter_correction_relative_to_torque_bound': max((d['composite_parameter_torque_correction_bound'] for d in data)), 'max_modal_norm': float(max(states[:, 0] + states[:, 1])), 'horizon_energy_cap_integral_bound_per_fc': integral(np.array([d['energy_cap_norm_per_fc'] for d in data]) * (states[:, 0] + states[:, 1]) * weights), 'survival_change_from_prescribed': result['initial_survival'] - next((r['initial_survival'] for r in final['local_maximum']['runs'] if r['variant'] == variant)), 'seconds': time.monotonic() - start})
    density = result.pop('density')
    result['density_real'], result['density_imag'] = (density.real.tolist(), density.imag.tolist())
    result.pop('coordinates')
    if result['norm_balance_residual'] > 1e-08 or result['minimum_density_eigenvalue'] < -1e-08:
        raise ValueError('Coupled density balance or positivity failed')
    return result