"""Reproduce numerical data with bounded, single-threaded calculations."""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import calculations as c
from boson_ep.binary_tidal_metric import CircularBinaryTide
from boson_ep.higher_tide import ElectricHexadecapoleTide
from boson_ep.models import TransitionConfig
from boson_ep.multipoles import PointCompanionMultipoles
from boson_ep.relativistic_tides import compute_transition_kernel, tidal_kernel_from_modes_M


def read_json(name):
    return json.loads((ROOT/"data"/name).read_text(encoding="utf-8"))


def verify_files():
    manifest = json.loads((ROOT/"manifest.json").read_text(encoding="utf-8"))
    entries = (json.dumps(manifest["files"], indent=2, sort_keys=True, allow_nan=False)+"\n").encode("utf-8")
    if hashlib.sha256(entries).hexdigest() != manifest["content_id"]:
        raise AssertionError("Manifest content identifier does not match")
    for name, entry in manifest["files"].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise AssertionError(f"Missing or unsafe file: {name}")
        if path.stat().st_size != entry["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise AssertionError(f"Checksum mismatch: {name}")
    import boson_ep
    if not Path(boson_ep.__file__).resolve().is_relative_to(ROOT.resolve()):
        raise AssertionError("The numerical package was imported from outside the distribution")
    return {"content_id": manifest["content_id"], "files": len(manifest["files"])}


def accuracy():
    coefficients = {r["point_id"]: r for r in read_json("accuracy_coefficients.json")}
    rows = c.read_csv(ROOT/"data/accuracy.csv")
    maximum, kept = 0.0, 0
    for row in rows:
        record, q = coefficients[row["point_id"]], float(row["q"])
        correction = abs(c.complete_kernel(record, q)/c.complete_kernel(record, q, hydrogenic=True))-1
        orbit = c.circular_local_tide_15pn(record["Omega_M"], q, record["chi"])
        rho = record["r99_M"]/orbit["separation_M_15pn"]
        maximum = max(maximum, abs(correction-float(row["completed_amplitude_change"])),
                      abs(rho-float(row["rho_15pn"])))
        kept += abs(correction) > .1 and rho <= .07
    if maximum > 1e-12 or len(rows) != 1095 or kept != 1094:
        raise AssertionError("Tidal comparison does not reproduce")
    return {"samples": len(rows), "retained_samples": kept, "maximum_absolute_difference": maximum}


def components():
    settings = c._load_settings(ROOT/"data/settings.json")
    expected = c.read_csv(ROOT/"data/components.csv")
    hybrids = {(r["point_id"], int(r["mask"])): float(r["kernel_abs"])
               for r in c.read_csv(ROOT/"data/hybrids.csv")}
    maximum = 0.0
    for row in expected:
        _, modes = c.load_mode_pair(row["point_id"])
        base, factors = c.factors(*modes, settings)
        values = {}
        for mask in range(16):
            value = base*math.prod(v for bit, v in enumerate(factors.values()) if mask & (1 << bit))
            maximum = max(maximum, abs(abs(value)/hybrids[row["point_id"], mask]-1))
            values[mask] = math.log(abs(value/base))
        for bit, name in enumerate(factors):
            share = sum(math.factorial(mask.bit_count())*math.factorial(3-mask.bit_count())/24
                        * (values[mask | (1 << bit)]-values[mask])
                        for mask in range(16) if not mask & (1 << bit))
            maximum = max(maximum, abs(share-float(row["shapley_"+name])))
    if maximum > 1e-10:
        raise AssertionError("Component reconstruction does not reproduce")
    return {"samples": len(expected), "subsets": len(hybrids), "maximum_difference": maximum}


def multipoles():
    results = []
    for sample in read_json("multipoles.json"):
        row, modes = c.load_mode_pair(sample["point_id"])
        config = TransitionConfig(float(row["alpha"]), modes[0].state, modes[1].state,
            q=sample["q"], chi=modes[0].chi, settings=c._load_settings(ROOT/"data/settings.json"))
        evaluator = PointCompanionMultipoles(config, modes)
        errors = [abs(evaluator.kernel(r["ell"], sample["separation_M"], strategy="quad")
                      /complex(*r["kernel"])-1) for r in sample["multipoles"]]
        if max(errors) > 2e-6:
            raise AssertionError("Multipole reconstruction does not reproduce")
        results.append({"point_id": sample["point_id"], "maximum_relative_difference": max(errors)})
    return results


def crossings(optimize=False):
    result = []
    higher = {r["point_id"]: r for r in read_json("electric_l4.json")}
    expected = {r["point_id"]: r for r in read_json("crossings.json")}
    previous = {r["point_id"]: r for r in read_json("crossing_seeds.json")}
    for record in read_json("directed_couplings.json"):
        target = expected[record["point_id"]]
        coefficients, _ = c.quartic_coefficients(higher[record["point_id"]])
        kh4 = complex(*previous[record["point_id"]]["hydrogenic_l4_kernel"])
        q = target["local_maximum"]["q"]
        upper = c.upper_endpoint(record, q)
        runs = [c.run_case(record, coefficients, kh4, q, variant, .05, upper, refine=True)
                for variant in ("matched", "hydrogenic")]
        errors = [abs(r["initial_survival"]-saved["initial_survival"])
                  for r, saved in zip(runs, target["local_maximum"]["runs"], strict=True)]
        if max(errors) > 2e-8:
            raise AssertionError("Finite-interval survival does not reproduce")
        bound = c.feedback_bound(record, coefficients, q, .05, upper)
        old_bound = target["continuous_feedback_bound"]["cloud_fraction_for_ten_percent_bound"]
        if abs(bound["cloud_fraction_for_ten_percent_bound"]/old_bound-1) > 1e-10:
            raise AssertionError("Population-transfer diagnostic does not reproduce")
        sample = {"point_id": record["point_id"], "q": q, "upper_frequency_ratio": upper,
                  "runs": runs, "transfer_diagnostic": bound, "survival_displacements": errors}
        if optimize:
            sample["optimized"] = c.assess(record, higher[record["point_id"]], previous[record["point_id"]])
        result.append(sample)
        print(json.dumps({"point": record["point_id"], "survival_displacements": errors}), flush=True)
    return result


def kernel_point(point):
    row, modes = c.load_mode_pair(point)
    omega, chi = float(row["omega_res_M"]), float(row["chi"])
    if omega <= 0:
        raise ValueError("This command requires a prograde resonance")
    settings = c._load_settings(ROOT/"data/settings.json")
    config = TransitionConfig(float(row["alpha"]), modes[0].state, modes[1].state, chi=chi, settings=settings)
    original = compute_transition_kernel(config, modes)
    value = original.covariant_kernel_M
    expected = complex(float(row["covariant_kernel_M_real"]), float(row["covariant_kernel_M_imag"]))
    if abs(value/expected-1) > 2e-8:
        raise AssertionError("Quadrupolar kernel does not reproduce")
    outer = 1+math.sqrt(1-chi*chi)+settings.mode.outer_decay_lengths/min(
        c._qbs_decay_rate(m.alpha, m.frequency_M) for m in modes)
    output = {"point_id": point, "quadrupole": [value.real, value.imag],
              "relative_difference": abs(value/expected-1)}
    for name, filename, tide in (
        ("electric_quadrupole_magnetic_octupole", "directed_couplings.json", CircularBinaryTide(chi, omega, 1.01*outer)),
        ("electric_l4", "electric_l4.json", ElectricHexadecapoleTide(chi, omega, 1.01*outer))):
        forward = tidal_kernel_from_modes_M(*modes, omega, settings, metric=tide.metric_bl)
        reverse = tidal_kernel_from_modes_M(*reversed(modes), omega, settings,
            metric=lambda r, t: tide.metric_bl(r, t).conj(), harmonic_m=2)
        output[name] = {"forward": [forward.real, forward.imag], "reverse": [reverse.real, reverse.imag]}
        saved = next((r for r in read_json(filename) if r["point_id"] == point), None)
        if saved is not None:
            reference = next(r for r in saved["samples"] if r["frequency_ratio"] == 1)
            errors = {key: abs(value/complex(*reference[key])-1)
                      for key, value in (("forward", forward), ("reverse", reverse))}
            if max(errors.values()) > 2e-8:
                raise AssertionError(f"Directed {name} coupling does not reproduce")
            output[name]["relative_differences"] = errors
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", action="store_true")
    parser.add_argument("--multipoles", action="store_true")
    parser.add_argument("--crossings", action="store_true")
    parser.add_argument("--optimize", action="store_true", help="Repeat the bounded local maximum searches")
    parser.add_argument("--kernel-point", help="Recompute a saved-mode coupling, for example a0.200000_21+1_21-1")
    parser.add_argument("--tests", action="store_true")
    parser.add_argument("--scalar-currents", action="store_true")
    parser.add_argument("--stress-overlaps", action="store_true")
    parser.add_argument("--spectral-response", action="store_true")
    parser.add_argument("--torque", action="store_true")
    parser.add_argument("--feedback", action="store_true")
    args = parser.parse_args()
    report = {"integrity": verify_files(), "accuracy": accuracy()}
    if args.components:
        report["components"] = components()
    if args.multipoles:
        report["multipoles"] = multipoles()
    if args.crossings or args.optimize:
        report["crossings"] = crossings(args.optimize)
    if args.kernel_point:
        report["kernel"] = kernel_point(args.kernel_point)
    if any((args.scalar_currents, args.stress_overlaps, args.spectral_response, args.torque, args.feedback)):
        import feedback_checks as fc
        for enabled, name, calculate in (
            (args.scalar_currents, "scalar_currents", fc.scalar_currents),
            (args.stress_overlaps, "stress_overlaps", fc.stress_overlaps),
            (args.spectral_response, "spectral_response", fc.spectral_response),
            (args.torque, "scalar_torque", fc.scalar_torque),
            (args.feedback, "orbital_feedback", fc.orbital_feedback),
        ):
            if enabled:
                report[name] = calculate()
    if args.tests:
        log = io.StringIO()
        suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"), top_level_dir=str(ROOT))
        tests = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        if not tests.wasSuccessful():
            raise AssertionError(log.getvalue())
        report["tests"] = {"run": tests.testsRun, "failures": len(tests.failures), "errors": len(tests.errors)}
    output = ROOT/"output"
    output.mkdir(exist_ok=True)
    (output/"reproduction.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "checks": list(report)}, indent=2))


if __name__ == "__main__":
    main()
