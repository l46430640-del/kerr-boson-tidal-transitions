"""Recalculate the scalar-current and orbital-feedback examples from public inputs."""

import math

import numpy as np

import feedback as f
from boson_ep.scalar_balance import mode_current_integrals


def close(actual, expected, *, rtol=2e-8, atol=1e-14):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def scalar_currents():
    results = []
    for pair in f.read_json("scalar_currents.json"):
        _, modes = f.load_mode_pair(pair["point_id"])
        rows = []
        for index, (mode, expected) in enumerate(zip(modes, pair["modes"], strict=True)):
            row = mode_current_integrals(mode, radial_nodes=48, angular_nodes=36, outer_decay_lengths=40)
            for quantity in ("charges", "horizon_flux", "outer_flux", "analytic_horizon_flux"):
                for key, value in row[quantity].items():
                    close(value, expected[quantity][key], atol=1e-14 if quantity == "charges" else 2e-18)
            if index == 1:
                for key, residual in row["balance_residual"].items():
                    flux = row["horizon_flux"][key]
                    if (abs(residual/flux) if flux else abs(residual)) > 2e-8:
                        raise AssertionError("Damped scalar current does not balance")
            rows.append(row)
        results.append({"point_id": pair["point_id"], "modes": rows})
    return results


def stress_overlaps():
    results = []
    points = [r["point_id"] for r in f.read_json("directed_couplings.json")]
    for expected in f.read_json("stress_overlaps.json"):
        index = points.index(expected["point_id"])
        row = f.calculate(index, expected["frequency_ratio"], expected["refined"])
        for key in ("time_weights", "radial_current", "horizon_amplitudes", "outer_amplitudes", "outer_radial_flux"):
            close(row[key], expected[key], rtol=2e-7, atol=1e-24)
        for key in row["overlaps"]:
            close(row["overlaps"][key], expected["overlaps"][key], rtol=2e-7, atol=1e-24)
        for offset, fields in row["boundary_overlaps"].items():
            for key, values in fields.items():
                close(values, expected["boundary_overlaps"][offset][key], rtol=2e-7, atol=1e-24)
        row.pop("seconds", None)
        row["scope"] = "Physical stress and boundary overlaps at the specified drive frequency."
        results.append(row)
    return results


def spectral_response():
    results = []
    currents = f.read_json("scalar_currents.json")
    overlaps = f.read_json("stress_overlaps.json")
    for index, expected in enumerate(f.read_json("spectral_response.json")):
        _, modes = f.load_mode_pair(expected["point_id"])
        rows = [f.derivatives(mode) for mode in modes]
        for row, saved in zip(rows, expected["modes"], strict=True):
            for key, (real, imag) in row["derivatives"].items():
                close(real, saved["derivatives"][key][0], atol=3e-8)
                close(imag, saved["derivatives"][key][1], rtol=1e-4, atol=1e-18)
        reference = next(r for r in overlaps if r["point_id"] == expected["point_id"]
                         and r["frequency_ratio"] == 1 and not r["refined"])
        number = currents[index]["modes"][0]["charges"]["number"]
        chi, m, wr = modes[0].chi, modes[0].state.m, modes[0].frequency_M.real
        rp = 1+math.sqrt(1-chi*chi)
        checks = {}
        for key, (real, imag) in rows[0]["derivatives"].items():
            threshold = 0 if key == "alpha" else -2*chi/math.sqrt(1-chi*chi)*wr-m
            independent = -reference["horizon_amplitudes"][0]/(2*number)*(2*rp*real+threshold)
            checks[key] = abs(independent/imag-1)
            if checks[key] > 1e-4:
                raise AssertionError("Width response disagrees with the horizon balance")
        results.append({"point_id": expected["point_id"], "modes": rows,
                        "horizon_width_relative_errors": checks})
    return results


def scalar_torque():
    results = []
    for index, expected in enumerate(f.read_json("scalar_torque.json")):
        inputs = f.inputs(index)
        row = f.assess(index, *inputs)
        for key in ("initial_survival", "physical_torque_peak_per_cloud_fraction",
                    "absolute_defect_over_absolute_torque_integral",
                    "integrated_cloud_torque_per_cloud_fraction", "integrated_horizon_angular_flux_per_cloud_fraction",
                    "integrated_kerr_horizon_energy_flux_per_cloud_fraction"):
            close(row[key], expected[key], rtol=2e-7, atol=1e-12)
        original = row["continuous_physical_torque_bound"]["torque_ratio_bound_per_cloud_fraction"]
        close(original, expected["continuous_physical_torque_bound"]["torque_ratio_bound_per_cloud_fraction"])
        record = inputs[0]
        q, upper = row["q"], row["frequency_bounds"][1]
        y = ((1+q)*record["Omega_res_M"]*upper)**(2/3)
        nu, x1 = q/(1+q)**2, 1/(1+q)
        a = 1.5-nu/6
        b = -record["chi"]*(10*x1*x1/3+13*nu/6)
        lower = 1-a*y-2*abs(b)*y**1.5
        if lower <= 0:
            raise AssertionError("No positive bound on the PN orbital derivative")
        row["pn_continuous_torque_bound_per_cloud_fraction"] = original/lower
        row["pn_derivative_lower_bound"] = lower
        if original/lower > (1.63e5, 1.32e6)[index]:
            raise AssertionError("Physical torque exceeds the quoted continuous bound")
        results.append(row)
    return results


def orbital_feedback():
    results = []
    for index, expected in enumerate(f.read_json("orbital_feedback.json")):
        rows = {}
        for name, saved in expected["runs"].items():
            row = f.run(index, saved["cloud_fraction"], saved["variant"],
                        absorption=saved["absorption_changes_background"],
                        torque_feedback=saved["torque_feedback"], refine=saved["refined"])
            close(row["initial_survival"], saved["initial_survival"], rtol=0, atol=2e-10)
            for key in ("mass_change", "angular_momentum_change", "max_relative_chirp_feedback",
                        "integrated_initial_decay_exponent", "max_initial_width_change"):
                close(row[key], saved[key], rtol=2e-6, atol=1e-20)
            row.pop("seconds", None)
            rows[name] = row
            print(f"feedback {index} {name}: {row['initial_survival']:.12g}", flush=True)
        for variant in ("matched", "hydrogenic"):
            close(rows[variant+"_zero"]["survival_change_from_prescribed"], 0, rtol=0, atol=1e-9)
            close(rows[variant+"_coupled"]["initial_survival"], rows[variant+"_refined"]["initial_survival"],
                  rtol=0, atol=1e-9)
        gap = rows["matched_refined"]["initial_survival"]-rows["hydrogenic_refined"]["initial_survival"]
        close(gap, expected["coupled_survival_difference"], rtol=0, atol=4e-10)
        results.append({"point_id": expected["point_id"], "runs": rows, "coupled_survival_difference": gap})
    return results
