"""Build the covariant adiabatic Kerr tidal-transition atlas.

The scan is single-process and restartable.  Pass ``--quick`` for a smoke
test; the full grid is the default and can take several hours.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boson_ep.models import (
    KerrModeSettings,
    RelativisticTideSettings,
    State,
    TransitionConfig,
)
from boson_ep.relativistic_tides import (
    TRANSITION_CHANNELS,
    compute_relativistic_transition,
    gauge_audit_transition,
)
from boson_ep.relativity import saturation_spin_cf, solve_kerr_mode


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "relativistic_tides"


def _canonical_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _code_hash() -> str:
    root = ROOT
    paths = list((root / "src" / "boson_ep").glob("*.py"))
    paths += [
        root / "scripts" / "certify_pre_atlas.py",
        Path(__file__).resolve(),
        root / "results/relativistic_tides/benchmarks/schwarzschild_kernel_reference.csv",
    ]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _state_from_label(label: str) -> State:
    if len(label) < 4 or label[2] not in "+-":
        raise ValueError(f"invalid state label in frozen config: {label}")
    return State(int(label[0]), int(label[1]), int(label[2:]))


def _load_frozen_configuration(
    config_path: Path, certificate_path: Path
) -> tuple[dict[str, object], RelativisticTideSettings]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    if certificate.get("status") != "pre_atlas_certified":
        raise SystemExit("formal atlas blocked: certificate is not pre_atlas_certified")
    embedded_hash = config.get("config_sha256")
    unhashed = dict(config)
    unhashed.pop("config_sha256", None)
    actual_hash = _canonical_hash(unhashed)
    if not embedded_hash or embedded_hash != actual_hash:
        raise SystemExit("formal atlas blocked: frozen config hash is invalid")
    if certificate.get("config_sha256") != embedded_hash:
        raise SystemExit("formal atlas blocked: certificate/config hashes differ")
    if certificate.get("code_sha256") != _code_hash():
        raise SystemExit("formal atlas blocked: certified source code has changed")
    settings_payload = dict(config["settings"])
    mode_payload = dict(settings_payload.pop("mode"))
    allowed_mode = {item.name for item in fields(KerrModeSettings)}
    allowed_tide = {item.name for item in fields(RelativisticTideSettings)}
    mode = KerrModeSettings(**{
        key: value for key, value in mode_payload.items() if key in allowed_mode
    })
    if "horizon_cutoffs" in settings_payload:
        settings_payload["horizon_cutoffs"] = tuple(settings_payload["horizon_cutoffs"])
    tide = RelativisticTideSettings(
        mode=mode,
        **{key: value for key, value in settings_payload.items() if key in allowed_tide and key != "mode"},
    )
    return config, tide


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    temporary.replace(path)


def _relative(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-300)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="run one low-cost point")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", type=Path, help="certified frozen atlas config")
    parser.add_argument("--certificate", type=Path, help="pre-atlas certificate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = OUTPUT / "smoke" if args.quick else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    if args.quick:
        alphas = (0.10,)
        channels = TRANSITION_CHANNELS[:1]
        mode_settings = KerrModeSettings(
            truncation=150, angular_lmax=10, series_terms=150, angular_nodes=40
        )
        tide_settings = RelativisticTideSettings(
            mode=mode_settings, angular_nodes=10, radial_nodes=32,
            horizon_cutoffs=(3.0e-4, 1.0e-4, 3.0e-5)
        )
        truncations = (100, 150)
        q_values = np.geomspace(1.0e-4, 1.0, 5)
        config_hash = "quick"
    else:
        if args.config is None or args.certificate is None:
            raise SystemExit(
                "formal atlas blocked: --config and --certificate are mandatory"
            )
        frozen, tide_settings = _load_frozen_configuration(
            args.config.resolve(), args.certificate.resolve()
        )
        alphas = tuple(float(value) for value in frozen["alphas"])
        channels = tuple(
            (_state_from_label(pair[0]), _state_from_label(pair[1]))
            for pair in frozen["channels"]
        )
        mode_settings = tide_settings.mode
        truncations = (150, 250, 400)
        q_values = np.asarray(frozen["q_values"], dtype=float)
        config_hash = str(frozen["config_sha256"])
    cache = output / "cache" / config_hash[:16]
    cache.mkdir(parents=True, exist_ok=True)
    atlas_rows: list[dict[str, object]] = []
    mode_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []
    phenomenology_rows: list[dict[str, object]] = []

    for alpha in alphas:
        for initial, final in channels:
            key = f"a{alpha:.6f}_{initial.label}_{final.label}"
            cache_path = cache / f"{key}.json"
            if cache_path.exists() and not args.overwrite:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                atlas_rows.append(payload["transition"])
                mode_rows.extend(payload.get("modes", []))
                audit_rows.append(payload["audit"])
                phenomenology_rows.extend(payload.get("phenomenology", []))
                convergence_rows.extend(payload.get("convergence", []))
                continue
            try:
                chi = saturation_spin_cf(alpha, initial, mode_settings)
            except RuntimeError as error:
                atlas_rows.append(
                    {
                        "alpha": alpha,
                        "initial": initial.label,
                        "final": final.label,
                        "status": "mode_not_converged",
                        "message": str(error),
                    }
                )
                continue
            modes = [
                solve_kerr_mode(alpha, chi, state, mode_settings)
                for state in (initial, final)
            ]
            local_modes = [mode.to_dict() for mode in modes]
            mode_rows.extend(local_modes)
            transition = compute_relativistic_transition(
                TransitionConfig(
                    alpha=alpha,
                    initial=initial,
                    final=final,
                    q=float(q_values[0]),
                    chi=chi,
                    settings=tide_settings,
                )
            )
            transition_row = transition.to_dict()
            atlas_rows.append(transition_row)
            audit = gauge_audit_transition(
                TransitionConfig(
                    alpha=alpha,
                    initial=initial,
                    final=final,
                    chi=chi,
                    settings=tide_settings,
                )
            ).to_dict()
            audit_rows.append(audit)
            local_phenomenology: list[dict[str, object]] = []
            if transition.status not in ("mode_not_converged", "no_prograde_resonance"):
                for q_value in q_values:
                    epsilon = q_value / (1.0 + q_value) * transition.omega_res_M**2
                    eta = epsilon * transition.covariant_kernel_M
                    chirp = (
                        96.0 / 5.0 * q_value / (1.0 + q_value) ** (1.0 / 3.0)
                        * transition.omega_res_M ** (11.0 / 3.0)
                    )
                    z_value = abs(eta) ** 2 / (
                        abs(initial.m - final.m) * chirp
                    )
                    local_phenomenology.append(
                        {
                            "alpha": alpha,
                            "chi": chi,
                            "initial": initial.label,
                            "final": final.label,
                            "q": q_value,
                            "eta_abs_M": abs(eta),
                            "landau_zener_z": z_value,
                            "depletion_probability": -math.expm1(-2.0 * math.pi * z_value),
                        }
                    )
            phenomenology_rows.extend(local_phenomenology)
            local_convergence: list[dict[str, object]] = []
            for state in (initial, final):
                peers = []
                for truncation in truncations:
                    result = solve_kerr_mode(
                        alpha,
                        chi,
                        state,
                        KerrModeSettings(
                            truncation=truncation,
                            angular_lmax=mode_settings.angular_lmax,
                            series_terms=truncation,
                            angular_nodes=mode_settings.angular_nodes,
                        ),
                    )
                    peers.append(result)
                reference = peers[-1]
                for truncation, result in zip(truncations, peers, strict=True):
                    local_convergence.append(
                        {
                            **result.to_dict(),
                            "truncation": truncation,
                            "real_frequency_relative_to_max_N": _relative(
                                result.frequency_M.real, reference.frequency_M.real
                            ),
                            "gamma_relative_to_max_N": _relative(
                                result.frequency_M.imag, reference.frequency_M.imag
                            ),
                            "norm_error": abs(result.bilinear_norm - 1.0),
                        }
                    )
            convergence_rows.extend(local_convergence)
            _atomic_json(
                cache_path,
                {
                    "transition": transition_row,
                    "modes": local_modes,
                    "audit": audit,
                    "phenomenology": local_phenomenology,
                    "convergence": local_convergence,
                },
            )

    _write_rows(output / "mode_catalog.csv", mode_rows)
    _write_rows(output / "transition_atlas.csv", atlas_rows)
    _write_rows(output / "gauge_audit.csv", audit_rows)
    _write_rows(output / "convergence.csv", convergence_rows)
    _write_rows(output / "phenomenology.csv", phenomenology_rows)
    for name, rows in (
        ("mode_catalog", mode_rows),
        ("transition_atlas", atlas_rows),
        ("gauge_audit", audit_rows),
        ("convergence", convergence_rows),
        ("phenomenology", phenomenology_rows),
    ):
        (output / f"{name}.json").write_text(
            json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8"
        )

    valid = [row for row in atlas_rows if "covariant_kernel_M_abs" in row]
    if valid:
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        for channel in sorted({(row["initial"], row["final"]) for row in valid}):
            selected = [row for row in valid if (row["initial"], row["final"]) == channel]
            axis.plot(
                [row["alpha"] for row in selected],
                [row["covariant_kernel_M_abs"] / row["hydrogenic_kernel_M_abs"] for row in selected],
                marker="o",
                label=f"{channel[0]} to {channel[1]}",
            )
        axis.axhline(1.0, color="black", linewidth=0.8)
        axis.set_xlabel(r"$\alpha$")
        axis.set_ylabel(r"$|K_{\rm cov}|/|K_{\rm H}|$")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output / "kernel_corrections.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        axis.scatter(
            [row["omega_res_M"] for row in valid],
            [row["r_99_over_b"] for row in valid],
            c=[row["alpha"] for row in valid],
        )
        axis.axvline(1.0e-2, color="black", linestyle="--")
        axis.axhline(0.10, color="black", linestyle="--")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$M\Omega_{\rm res}$")
        axis.set_ylabel(r"$r_{99}/b$")
        figure.tight_layout()
        figure.savefig(output / "tidal_validity.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        for channel in sorted({(row["initial"], row["final"]) for row in valid}):
            selected = [row for row in valid if (row["initial"], row["final"]) == channel]
            axis.plot(
                [row["alpha"] for row in selected],
                [row["omega_res_M"] for row in selected],
                marker="o",
                label=f"{channel[0]} to {channel[1]}",
            )
        axis.axhline(1.0e-2, color="black", linestyle="--", linewidth=0.8)
        axis.set_yscale("log")
        axis.set_xlabel(r"$\alpha$")
        axis.set_ylabel(r"$M\Omega_{\rm res}$")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output / "resonance_frequencies.png", dpi=180)
        plt.close(figure)

    if phenomenology_rows:
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        for channel in sorted({(row["initial"], row["final"]) for row in phenomenology_rows}):
            selected = [
                row for row in phenomenology_rows
                if (row["initial"], row["final"]) == channel
                and row["alpha"] == min(
                    item["alpha"] for item in phenomenology_rows
                    if (item["initial"], item["final"]) == channel
                )
            ]
            axis.plot(
                [row["q"] for row in selected],
                [row["depletion_probability"] for row in selected],
                label=f"{channel[0]} to {channel[1]}",
            )
        axis.set_xscale("log")
        axis.set_xlabel(r"$q$")
        axis.set_ylabel("Landau-Zener depletion")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output / "lz_depletion.png", dpi=180)
        plt.close(figure)

    if convergence_rows:
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        for state in sorted({row["state"] for row in convergence_rows}):
            selected = [row for row in convergence_rows if row["state"] == state]
            axis.plot(
                [row["truncation"] for row in selected],
                [max(row["radial_residual"], 1.0e-18) for row in selected],
                marker="o",
                label=state,
            )
        axis.set_yscale("log")
        axis.set_xlabel("Leaver truncation N")
        axis.set_ylabel("radial ODE residual")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output / "mode_convergence.png", dpi=180)
        plt.close(figure)

    summary = {
        "quick": args.quick,
        "config_sha256": None if args.quick else config_hash,
        "points": len(atlas_rows),
        "status_counts": {
            status: sum(row.get("status") == status for row in atlas_rows)
            for status in sorted({str(row.get("status")) for row in atlas_rows})
        },
        "gauge_audits_passed": sum(row["status"] == "ok" for row in audit_rows),
        "scan_gate": (
            "quick_smoke_not_certified" if args.quick else "pre_atlas_certified"
        ),
        "local_gauge_audit_note": (
            "legacy pointwise diagnostics are not the compact-support pilot certificate"
        ),
    }
    (output / "scan_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
