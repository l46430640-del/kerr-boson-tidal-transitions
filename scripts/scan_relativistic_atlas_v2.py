"""Certified, restartable v2 Kerr tidal-transition atlas.

The v2 runner keeps modes, q-independent kernels and q-dependent binary
phenomenology in separate stages.  Every cache object is content addressed;
legacy atlas caches are intentionally unreadable here.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from boson_ep.certification import direct_ward_audit, operator_form_audit
from boson_ep.models import (
    GaugeVectorSpec,
    KernelErrorBudgetResult,
    KerrModeResult,
    KerrModeSettings,
    PhenomenologyResult,
    RelativisticTideSettings,
    SaturationResult,
    State,
    TransitionConfig,
    TransitionKernelResult,
    WardAuditResult,
)
from boson_ep.relativistic_tides import (
    TRANSITION_CHANNELS,
    compute_kernel_error_budget,
    compute_transition_kernel,
    evaluate_transition_at_q,
)
from boson_ep.relativity import (
    solve_saturation_mode_cf,
    solve_kerr_mode_adaptive,
    trace_saturation_branch_cf,
)


SCHEMA_VERSION = 2
ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = ROOT / "results" / "relativistic_tides" / "v2"
REFERENCE = ROOT / "results/relativistic_tides/benchmarks/schwarzschild_kernel_reference.csv"
EXPECTED_COUNTS = {
    "saturation": 45,
    "modes": 120,
    "kernels": 75,
    "audits": 75,
    "error_budgets": 75,
    "phenomenology": 2475,
}
PILOT_IDS = {
    "a0.100000_21+1_21-1",
    "a0.250000_32+2_30+0",
    "a0.450000_43+3_41+1",
}


def _strict_value(value: Any) -> Any:
    """Convert NumPy/scientific values into RFC-compliant JSON values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, complex):
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            return None
        return [value.real, value.imag]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _strict_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_strict_value(item) for item in value]
    return value


def canonical_json(payload: Any) -> str:
    return json.dumps(
        _strict_value(payload), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    )


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def source_fingerprint() -> str:
    """Hash exactly the source set shared with the v2 certifier."""
    paths = list((ROOT / "src" / "boson_ep").glob("*.py"))
    paths += [
        ROOT / "scripts" / "certify_pre_atlas_v2.py",
        Path(__file__).resolve(),
        REFERENCE,
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"v2 fingerprint source is missing: {missing[0]}")
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def dependency_fingerprint() -> tuple[str, dict[str, str]]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    # This record is intentionally identical to the v2 certifier.  Matplotlib
    # only consumes completed rows and is not part of the numerical identity.
    for name in ("scipy", "sympy"):
        versions[name] = importlib.metadata.version(name)
    return canonical_hash(versions), versions


def run_fingerprint(config_sha256: str) -> tuple[str, dict[str, str]]:
    dependency_sha, versions = dependency_fingerprint()
    digest = canonical_hash({
        "schema_version": SCHEMA_VERSION,
        "code_sha256": source_fingerprint(),
        "config_sha256": config_sha256,
        "dependency_sha256": dependency_sha,
    })
    return digest, versions


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_strict_value(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    safe_rows: list[dict[str, Any]] = []
    for source in _strict_value(rows):
        safe_rows.append({
            key: (
                json.dumps(value, sort_keys=True, allow_nan=False)
                if isinstance(value, (dict, list))
                else value
            )
            for key, value in source.items()
        })
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(safe_rows)
    temporary.replace(path)


class StagedCache:
    def __init__(
        self, directory: Path, code_sha256: str, config_sha256: str,
        dependency_sha256: str,
    ) -> None:
        self.directory = directory
        self.code_sha256 = code_sha256
        self.config_sha256 = config_sha256
        self.dependency_sha256 = dependency_sha256
        directory.mkdir(parents=True, exist_ok=True)

    def _path(self, stage: str, point_id: str) -> Path:
        safe = point_id.replace("/", "_").replace("+", "p")
        return self.directory / stage / f"{safe}.json"

    def load(self, stage: str, point_id: str) -> dict[str, Any] | None:
        path = self._path(stage, point_id)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        expected = {
            "schema_version": SCHEMA_VERSION,
            "code_sha256": self.code_sha256,
            "config_sha256": self.config_sha256,
            "dependency_sha256": self.dependency_sha256,
            "point_id": point_id,
            "stage": stage,
        }
        if any(envelope.get(key) != value for key, value in expected.items()):
            return None
        if not isinstance(envelope.get("stage_status"), str):
            return None
        payload = envelope.get("payload")
        if envelope.get("payload_sha256") != canonical_hash(payload):
            return None
        return payload

    def store(
        self, stage: str, point_id: str, stage_status: str,
        payload: dict[str, Any],
    ) -> None:
        strict = _strict_value(payload)
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "code_sha256": self.code_sha256,
            "config_sha256": self.config_sha256,
            "dependency_sha256": self.dependency_sha256,
            "point_id": point_id,
            "stage": stage,
            "stage_status": stage_status,
            "payload_sha256": canonical_hash(strict),
            "payload": strict,
        }
        _atomic_json(self._path(stage, point_id), envelope)


def _state(label: str) -> State:
    return State(int(label[0]), int(label[1]), int(label[2:]))


def _complex(row: dict[str, Any], prefix: str) -> complex | None:
    real, imag = row.get(prefix + "_real"), row.get(prefix + "_imag")
    return None if real is None or imag is None else complex(real, imag)


def _number(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key)
    return default if value is None else float(value)


def _mode_from_row(row: dict[str, Any]) -> KerrModeResult:
    return KerrModeResult(
        alpha=float(row["alpha"]), chi=float(row["chi"]), state=_state(row["state"]),
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
        converged=bool(row["converged"]), message=str(row.get("message", "")),
        selected_truncation=row.get("selected_truncation"),
        convergence_history=tuple(row.get("convergence_history", ())),
        radial_ode_residual=(
            None
            if row.get("radial_ode_residual") is None
            else float(row["radial_ode_residual"])
        ),
    )


def _saturation_from_row(row: dict[str, Any]) -> SaturationResult:
    return SaturationResult(
        float(row["alpha"]), _state(row["state"]), _number(row, "chi"),
        complex(_number(row, "omega_real_M", 0.0), _number(row, "gamma_M", 0.0)),
        _number(row, "cf_residual", math.inf),
        _number(row, "saturation_residual", math.inf), int(row["truncation"]),
        bool(row["converged"]), str(row.get("message", "")),
    )


def _kernel_from_row(row: dict[str, Any]) -> TransitionKernelResult:
    return TransitionKernelResult(
        row["point_id"], float(row["alpha"]), _number(row, "chi"),
        _state(row["initial"]), _state(row["final"]), row["kernel_status"],
        row.get("failure_stage"), row.get("omega_res_M"), row.get("r_99_M"),
        _complex(row, "hydrogenic_kernel_M"),
        _complex(row, "semirelativistic_kernel_M"),
        _complex(row, "covariant_kernel_M"), str(row.get("message", "")),
    )


def _budget_from_row(row: dict[str, Any]) -> KernelErrorBudgetResult:
    return KernelErrorBudgetResult(
        row["point_id"], row["status"], tuple(row.get("sources", ())),
        row.get("rss_relative"), row.get("worst_relative"),
        row.get("worst_source"), row.get("systematic_error_abs"),
        bool(row.get("complete", False)),
    )


def _ward_from_row(row: dict[str, Any]) -> WardAuditResult:
    return WardAuditResult(
        float(row["alpha"]), _number(row, "chi"), _state(row["initial"]),
        _state(row["final"]), row["status"],
        complex(_number(row, "physical_kernel_real", 0.0),
                _number(row, "physical_kernel_imag", 0.0)),
        _number(row, "maximum_invariance_residual", math.inf),
        _number(row, "maximum_pure_gauge_residual", math.inf),
        _number(row, "maximum_linearity_residual", math.inf),
        _number(row, "off_resonance_ratio", 0.0), tuple(row.get("rows", ())),
    )


def _load_inputs(config_path: Path, certificate_path: Path):
    frozen = json.loads(config_path.read_text(encoding="utf-8"))
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    if certificate.get("status") != "pre_atlas_certified":
        raise SystemExit("v2 atlas blocked: certificate is not pre_atlas_certified")
    if certificate.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit("v2 atlas blocked: certificate schema is not v2")
    unhashed = dict(frozen)
    embedded = unhashed.pop("config_sha256", None)
    actual = canonical_hash(unhashed)
    if embedded != actual or certificate.get("config_sha256") != embedded:
        raise SystemExit("v2 atlas blocked: frozen configuration hash mismatch")
    code_sha = source_fingerprint()
    if certificate.get("code_sha256") != code_sha:
        raise SystemExit("v2 atlas blocked: certified source fingerprint mismatch")
    if certificate.get("implementation_failures") != 0 or certificate.get("failures"):
        raise SystemExit("v2 atlas blocked: certification contains implementation failures")
    settings_payload = dict(frozen["settings"])
    mode_payload = dict(settings_payload.pop("mode"))
    if "adaptive_truncations" in mode_payload:
        mode_payload["adaptive_truncations"] = tuple(mode_payload["adaptive_truncations"])
    mode_names = {item.name for item in fields(KerrModeSettings)}
    tide_names = {item.name for item in fields(RelativisticTideSettings)}
    mode = KerrModeSettings(**{key: value for key, value in mode_payload.items()
                               if key in mode_names})
    if "horizon_cutoffs" in settings_payload:
        settings_payload["horizon_cutoffs"] = tuple(settings_payload["horizon_cutoffs"])
    tide = RelativisticTideSettings(
        mode=mode, **{key: value for key, value in settings_payload.items()
                      if key in tide_names and key != "mode"}
    )
    alphas = tuple(float(value) for value in frozen["alphas"])
    channels = tuple((_state(pair[0]), _state(pair[1])) for pair in frozen["channels"])
    q_values = tuple(float(value) for value in frozen["q_values"])
    if (
        len(alphas) != 15 or len(set(alphas)) != 15
        or len(channels) != 5 or len(set(channels)) != 5
        or len(q_values) != 33 or len(set(q_values)) != 33
    ):
        raise SystemExit("v2 atlas blocked: formal grid must be 15 x 5 x 33")
    return frozen, certificate, embedded, tide, alphas, channels, q_values


def _gauge_specs(chi: float, r_99: float) -> tuple[GaugeVectorSpec, ...]:
    r_plus = 1.0 + math.sqrt(1.0 - chi**2)
    supports = (
        ("near_horizon", r_plus + 2.0, r_plus + 10.0),
        ("cloud_core", max(r_plus + 0.1, 0.2 * r_99), max(r_plus + 0.2, 0.7 * r_99)),
    )
    rows = []
    for kind in ("temporal", "radial", "polar", "axial"):
        for name, inner, outer in supports:
            if outer <= inner:
                outer = inner + 1.0
            for amplitude in (0.1, 1.0):
                rows.append(GaugeVectorSpec(kind, inner, outer, amplitude,
                                            support_name=name))
    return tuple(rows)


def _failure_row(point_id: str, stage: str, message: str) -> dict[str, Any]:
    return {"point_id": point_id, "status": f"{stage}_failed",
            "failure_stage": stage, "message": message}


def _savefig_atomic(fig, path: Path) -> None:
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    fig.savefig(temporary, dpi=180)
    temporary.replace(path)


def _finite_number(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def _branch_difference(row: dict[str, Any]) -> float:
    marker = "direction_difference="
    message = str(row.get("message", ""))
    if marker not in message:
        return math.inf
    try:
        return float(message.split(marker, 1)[1].split(";", 1)[0])
    except ValueError:
        return math.inf


def _mode_row_passes(row: dict[str, Any]) -> bool:
    return bool(
        row.get("converged")
        and _finite_number(row.get("cf_residual"))
        and float(row["cf_residual"]) < 1.0e-10
        and _finite_number(row.get("radial_residual"))
        and float(row["radial_residual"]) < 1.0e-8
        and _finite_number(row.get("angular_residual"))
        and float(row["angular_residual"]) < 1.0e-8
        and _finite_number(row.get("radial_ode_residual"))
        and float(row["radial_ode_residual"]) < 1.0e-6
        and _finite_number(row.get("bilinear_norm_error"))
        and float(row["bilinear_norm_error"]) < 1.0e-5
        and _finite_number(row.get("r_99_M"))
        and float(row["r_99_M"]) > 0.0
        and row.get("selected_truncation") is not None
        and int(row["selected_truncation"]) <= 1200
        and row.get("convergence_history")
        and bool(row["convergence_history"][-1].get("passed"))
    )


def _audit_row_passes(row: dict[str, Any], *, pilot: bool) -> bool:
    if row.get("status") == "not_applicable":
        return True
    return bool(
        row.get("status") == "ok"
        and _finite_number(row.get("maximum_invariance_residual"))
        and float(row["maximum_invariance_residual"]) < 1.0e-8
        and _finite_number(row.get("maximum_pure_gauge_residual"))
        and float(row["maximum_pure_gauge_residual"]) < 1.0e-8
        and _finite_number(row.get("maximum_linearity_residual"))
        and float(row["maximum_linearity_residual"]) < 1.0e-8
        and _finite_number(row.get("operator_form_residual"))
        and float(row["operator_form_residual"]) < 1.0e-10
        and len(row.get("rows", ())) == 16
        and len(row.get("operator_form_rows", ())) == 12
        and (
            not pilot
            or (
                _finite_number(row.get("off_resonance_ratio"))
                and float(row["off_resonance_ratio"]) > 100.0
            )
        )
    )


def _budget_row_passes(row: dict[str, Any], *, applicable: bool) -> bool:
    if not applicable:
        return row.get("status") == "not_applicable" and bool(row.get("complete"))
    expected = {
        "mode_N", "angular_lmax", "radial_nodes", "angular_nodes",
        "horizon_cutoff", "counterterm", "outer_range",
        "radial_strategy", "delta_box_form",
    }
    sources = row.get("sources", ())
    names = {source.get("source") for source in sources}
    return bool(
        row.get("status") == "ok"
        and row.get("complete")
        and len(sources) == len(expected)
        and names == expected
        and all(
            source.get("status") == "ok"
            and _finite_number(source.get("absolute_shift"))
            and _finite_number(source.get("relative_shift"))
            for source in sources
        )
        and _finite_number(row.get("rss_relative"))
        and _finite_number(row.get("worst_relative"))
        and _finite_number(row.get("systematic_error_abs"))
    )


def _plots(output: Path, kernels: list[dict[str, Any]],
           budgets: list[dict[str, Any]], phenomena: list[dict[str, Any]]) -> None:
    publication = [row for row in phenomena if row.get("publication_valid")]
    publication_ids = {row["point_id"] for row in publication}
    kernel_ok = [row for row in kernels if row.get("kernel_status") == "ok"
                 and row["point_id"] in publication_ids]
    if kernel_ok:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for channel in sorted({(row["initial"], row["final"]) for row in kernel_ok}):
            selected = sorted((row for row in kernel_ok
                               if (row["initial"], row["final"]) == channel),
                              key=lambda row: row["alpha"])
            ax.plot([row["alpha"] for row in selected],
                    [row["covariant_kernel_M_abs"] / row["hydrogenic_kernel_M_abs"]
                     for row in selected], marker="o", label=f"{channel[0]} to {channel[1]}")
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set(xlabel=r"$\alpha$", ylabel=r"$|K_{cov}|/|K_H|$")
        ax.legend(fontsize=8)
        fig.tight_layout(); _savefig_atomic(fig, output / "kernel_corrections.png"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for channel in sorted({(row["initial"], row["final"]) for row in kernel_ok}):
            selected = sorted((row for row in kernel_ok
                               if (row["initial"], row["final"]) == channel),
                              key=lambda row: row["alpha"])
            ax.plot([row["alpha"] for row in selected],
                    [row["omega_res_M"] for row in selected], marker="o",
                    label=f"{channel[0]} to {channel[1]}")
        ax.axhline(1.0e-2, color="black", linestyle="--", linewidth=0.8)
        ax.set_yscale("log"); ax.set(xlabel=r"$\alpha$", ylabel=r"$M\Omega_{res}$")
        ax.legend(fontsize=8)
        fig.tight_layout(); _savefig_atomic(fig, output / "resonance_frequencies.png"); plt.close(fig)
    if publication:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.scatter([row["q"] for row in publication],
                   [row["r_99_over_b"] for row in publication],
                   c=[row["alpha"] for row in publication], s=12)
        ax.set_xscale("log"); ax.set(xlabel=r"$q$", ylabel=r"$r_{99}/b$")
        fig.tight_layout(); _savefig_atomic(fig, output / "tidal_validity.png"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for channel in sorted({(row["initial"], row["final"]) for row in publication}):
            pool = [row for row in publication if (row["initial"], row["final"]) == channel]
            alpha = min(row["alpha"] for row in pool)
            selected = sorted((row for row in pool if row["alpha"] == alpha), key=lambda row: row["q"])
            ax.plot([row["q"] for row in selected],
                    [row["depletion_covariant"] for row in selected], label=f"{channel[0]} to {channel[1]}")
        ax.set_xscale("log"); ax.set(xlabel=r"$q$", ylabel="Covariant depletion")
        ax.legend(fontsize=8)
        fig.tight_layout(); _savefig_atomic(fig, output / "lz_depletion.png"); plt.close(fig)
    complete = [row for row in budgets if row.get("complete")
                and row["point_id"] in publication_ids]
    if complete:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.scatter([row["point_id"] for row in complete],
                   [row["worst_relative"] for row in complete], s=14)
        ax.set_yscale("log"); ax.tick_params(axis="x", labelrotation=90, labelsize=5)
        ax.set_ylabel("Measured worst-case relative error")
        fig.tight_layout(); _savefig_atomic(fig, output / "error_budget.png"); plt.close(fig)


def _write_report(output: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Kerr 潮汐跃迁图谱 v2 报告", "",
        f"- Atlas gate: `{summary['atlas_gate_status']}`",
        f"- Input certificate: `{summary['input_certificate_status']}`",
        f"- Implementation failures: `{summary['implementation_failures']}`",
        f"- Run fingerprint: `{summary['run_fingerprint']}`", "",
        "## 输出基数", "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in summary["counts"].items())
    lines += ["", "## 解释", "",
              "`tidal_expansion_invalid`、`adiabatic_tide_invalid` 和 `no_valid_q` 是物理/有效域状态，"
              "不计作实现失败。只有求根、模态、Ward、误差预算、缓存或输出完整性失败才阻止认证。", ""]
    science = summary.get("scientific_assessment", {})
    if science:
        lines += [
            "## 物理门槛", "",
            f"- 有发表有效 q 的 kernel 点：`{science['publication_kernel_points']}`",
            f"- 无有效 q 的点：`{science['no_valid_q_points']}`",
            f"- 最大有效域 kernel 修正：`{science['maximum_kernel_correction']:.6e}`",
            f"- 通过 10% 且 3 sigma_worst 门槛的点：`{science['significant_kernel_points']}`",
            f"- 最大 depletion 绝对变化：`{science['maximum_depletion_change']:.6e}`",
            "",
            ("实现认证通过后存在超过误差预算的显著相对论修正。"
             if science["significant_kernel_points"]
             else "实现认证与物理显著性分离：当前有效域内没有点通过主论文显著性门槛。"),
            "",
        ]
    temporary = output / "relativistic_tides_report.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output / "relativistic_tides_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen, certificate, config_sha, settings, alphas, channels, q_values = _load_inputs(
        args.config.resolve(), args.certificate.resolve()
    )
    code_sha = source_fingerprint()
    dependency_sha, versions = dependency_fingerprint()
    fingerprint, _ = run_fingerprint(config_sha)
    if certificate.get("run_fingerprint") != fingerprint:
        raise SystemExit("v2 atlas blocked: certificate/run fingerprints differ")
    output = V2_ROOT / fingerprint
    output.mkdir(parents=True, exist_ok=True)
    cache = StagedCache(output / "cache", code_sha, config_sha, dependency_sha)

    initial_states = tuple(dict.fromkeys(initial for initial, _ in channels))
    saturation: dict[tuple[float, State], SaturationResult] = {}
    saturation_rows: list[dict[str, Any]] = []
    for state in initial_states:
        missing = []
        for alpha in alphas:
            point_id = f"a{alpha:.6f}_{state.label}"
            row = None if args.overwrite else cache.load("saturation", point_id)
            if row is not None:
                result = _saturation_from_row(row)
                saturation[(alpha, state)] = result; saturation_rows.append(row)
            else:
                missing.append(alpha)
        if missing:
            try:
                traced = trace_saturation_branch_cf(tuple(missing), state, settings.mode)
            except RuntimeError as branch_error:
                traced = []
                for alpha in missing:
                    try:
                        traced.append(solve_saturation_mode_cf(alpha, state, settings.mode))
                    except RuntimeError as point_error:
                        traced.append(SaturationResult(
                            alpha, state, math.nan, 0j, math.inf, math.inf,
                            settings.mode.truncation, False,
                            f"bidirectional branch: {branch_error}; point: {point_error}",
                        ))
            for result in traced:
                point_id = f"a{result.alpha:.6f}_{state.label}"
                row = result.to_dict()
                cache.store("saturation", point_id, "ok" if result.converged else "failed", row)
                saturation[(result.alpha, state)] = result; saturation_rows.append(row)
    saturation_rows.sort(key=lambda row: (row["alpha"], row["state"]))

    requirements: dict[tuple[float, State, State], float] = {}
    for alpha in alphas:
        for initial, final in channels:
            sat = saturation.get((alpha, initial))
            chi = sat.chi if sat is not None and sat.converged else math.nan
            requirements[(alpha, initial, initial)] = chi
            requirements[(alpha, initial, final)] = chi
    modes: dict[tuple[float, State, State], KerrModeResult] = {}
    mode_rows: list[dict[str, Any]] = []
    previous_modes: dict[tuple[State, State], tuple[float, complex]] = {}
    for (alpha, branch, state), chi in requirements.items():
        point_id = f"a{alpha:.6f}_{branch.label}_{state.label}"
        row = None if args.overwrite else cache.load("modes", point_id)
        if row is not None and row.get("angular_coefficients") is not None:
            result = _mode_from_row(row)
        elif math.isfinite(chi):
            continuation = previous_modes.get((branch, state))
            if continuation is None:
                sat = saturation.get((alpha, branch))
                seed = sat.frequency_M if sat is not None and sat.converged else None
            else:
                prior_alpha, seed = continuation
                current_hydrogenic = alpha * (
                    1.0 - alpha**2 / (2.0 * state.n**2)
                )
                prior_hydrogenic = prior_alpha * (
                    1.0 - prior_alpha**2 / (2.0 * state.n**2)
                )
                seed += current_hydrogenic - prior_hydrogenic
            result = solve_kerr_mode_adaptive(
                alpha, chi, state, settings.mode, seed=seed
            )
            row = result.to_dict(include_coefficients=True)
            cache.store("modes", point_id, "ok" if result.converged else "failed", row)
        else:
            row = _failure_row(point_id, "saturation", "no converged saturation spin")
            cache.store("modes", point_id, "failed", row)
            mode_rows.append(row)
            continue
        modes[(alpha, branch, state)] = result
        if result.converged:
            previous_modes[(branch, state)] = (alpha, result.frequency_M)
        else:
            previous_modes.pop((branch, state), None)
        mode_row = {
            "point_id": point_id,
            "branch": branch.label,
            **result.to_dict(),
            "bilinear_norm_error": abs(result.bilinear_norm - 1.0),
        }
        mode_rows.append(mode_row)
    mode_rows.sort(key=lambda row: (row.get("alpha", -1), row.get("state", ""), row.get("point_id", "")))

    kernels: dict[str, TransitionKernelResult] = {}
    kernel_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    phenomenon_rows: list[dict[str, Any]] = []
    pilot_ids = PILOT_IDS
    for alpha in alphas:
        for initial, final in channels:
            point_id = f"a{alpha:.6f}_{initial.label}_{final.label}"
            pair = (modes.get((alpha, initial, initial)), modes.get((alpha, initial, final)))
            config = TransitionConfig(alpha, initial, final, chi=(pair[0].chi if pair[0] else None), settings=settings)
            cached = None if args.overwrite else cache.load("kernels", point_id)
            if cached is not None:
                kernel = _kernel_from_row(cached)
            elif all(pair):
                kernel = compute_transition_kernel(config, pair)
                cache.store("kernels", point_id, kernel.kernel_status, kernel.to_dict())
            else:
                kernel = TransitionKernelResult(point_id, alpha, math.nan, initial, final,
                    "mode_not_converged", "modes", None, None, None, None, None,
                    "one or both resolved modes are unavailable")
                cache.store("kernels", point_id, "failed", kernel.to_dict())
            kernels[point_id] = kernel; kernel_rows.append(kernel.to_dict())

            cached = None if args.overwrite else cache.load("error_budgets", point_id)
            if cached is not None:
                budget = _budget_from_row(cached)
            elif kernel.kernel_status == "ok" and all(pair):
                budget = compute_kernel_error_budget(config, pair)
                cache.store("error_budgets", point_id, budget.status, budget.to_dict())
            else:
                budget = KernelErrorBudgetResult(point_id, "not_applicable", (), None, None, None, None, True)
                cache.store("error_budgets", point_id, budget.status, budget.to_dict())
            budget_rows.append(budget.to_dict())

            cached = None if args.overwrite else cache.load("audits", point_id)
            if cached is not None:
                audit = _ward_from_row(cached)
                audit_row = cached
            elif kernel.kernel_status == "ok" and all(pair):
                audit = direct_ward_audit(
                    config, pair, _gauge_specs(kernel.chi, float(kernel.r_99_M)),
                    include_off_resonance=point_id in pilot_ids,
                    physical_kernel=complex(kernel.covariant_kernel_M),
                )
                operator_residual, operator_rows = operator_form_audit(
                    config, resolved_modes=pair
                )
                combined_rows = list(audit.rows)
                if combined_rows:
                    combined_rows[0] = {
                        **combined_rows[0],
                        "operator_form_maximum_residual": operator_residual,
                        "operator_form_rows": operator_rows,
                    }
                audit = replace(
                    audit,
                    status=(
                        "ok" if audit.status == "ok" and operator_residual < 1.0e-10
                        else "gauge_audit_failed"
                    ),
                    rows=tuple(combined_rows),
                )
                audit_row = {
                    "point_id": point_id,
                    **audit.to_dict(),
                    "operator_form_residual": operator_residual,
                    "operator_form_rows": operator_rows,
                }
                cache.store("audits", point_id, audit.status, audit_row)
            else:
                audit = WardAuditResult(alpha, kernel.chi, initial, final,
                    "not_applicable", 0j, 0.0, 0.0, 0.0, 0.0, ())
                audit_row = {
                    "point_id": point_id,
                    **audit.to_dict(),
                    "operator_form_residual": None,
                    "operator_form_rows": [],
                }
                cache.store("audits", point_id, audit.status, audit_row)
            audit_rows.append(audit_row)

            for q in q_values:
                q_id = f"{point_id}_q{q:.12e}"
                cached_q = None if args.overwrite else cache.load("phenomenology", q_id)
                if cached_q is None:
                    result = evaluate_transition_at_q(
                        kernel, q, budget,
                        tidal_radius_limit=settings.tidal_radius_limit,
                        adiabatic_frequency_limit=settings.adiabatic_frequency_limit,
                    )
                    cached_q = result.to_dict()
                    if result.eta_semirelativistic_M is not None and result.eta_hydrogenic_M:
                        cached_q["power_ratio_semirelativistic_to_hydrogenic"] = (
                            abs(result.eta_semirelativistic_M / result.eta_hydrogenic_M) ** 2
                        )
                    cache.store("phenomenology", q_id, result.status, cached_q)
                phenomenon_rows.append(cached_q)

    kernel_rows.sort(key=lambda row: (row["alpha"], row["initial"], row["final"]))
    budget_rows.sort(key=lambda row: row["point_id"])
    audit_rows.sort(key=lambda row: (row["alpha"], row["initial"], row["final"]))
    phenomenon_rows.sort(key=lambda row: (row["point_id"], row["q"]))
    datasets = {
        "saturation": saturation_rows, "modes": mode_rows, "kernels": kernel_rows,
        "audits": audit_rows, "error_budgets": budget_rows,
        "phenomenology": phenomenon_rows,
    }
    for name, rows in datasets.items():
        _atomic_json(output / f"{name}.json", rows)
        _atomic_csv(output / f"{name}.csv", rows)

    counts = {name: len(rows) for name, rows in datasets.items()}
    expected_point_ids = {
        f"a{alpha:.6f}_{initial.label}_{final.label}"
        for alpha in alphas for initial, final in channels
    }
    expected_saturation_ids = {
        (round(alpha, 12), state.label)
        for alpha in alphas for state in initial_states
    }
    # Build the canonical 120 branch/state ids explicitly.
    expected_mode_ids = set()
    for alpha in alphas:
        for branch in initial_states:
            states = {branch}
            states.update(final for initial, final in channels if initial == branch)
            expected_mode_ids.update(
                f"a{alpha:.6f}_{branch.label}_{state.label}" for state in states
            )
    expected_q_ids = {
        (point_id, float(q)) for point_id in expected_point_ids for q in q_values
    }
    actual_id_sets = {
        "saturation": {
            (round(float(row["alpha"]), 12), str(row["state"]))
            for row in saturation_rows if "alpha" in row and "state" in row
        },
        "modes": {row.get("point_id") for row in mode_rows},
        "kernels": {row.get("point_id") for row in kernel_rows},
        "audits": {row.get("point_id") for row in audit_rows},
        "error_budgets": {row.get("point_id") for row in budget_rows},
        "phenomenology": {
            (row.get("point_id"), float(row["q"]))
            for row in phenomenon_rows if row.get("q") is not None
        },
    }
    expected_id_sets = {
        "saturation": expected_saturation_ids,
        "modes": expected_mode_ids,
        "kernels": expected_point_ids,
        "audits": expected_point_ids,
        "error_budgets": expected_point_ids,
        "phenomenology": expected_q_ids,
    }
    count_failures = sum(
        counts.get(name) != expected
        or actual_id_sets[name] != expected_id_sets[name]
        or len(actual_id_sets[name]) != len(datasets[name])
        for name, expected in EXPECTED_COUNTS.items()
    )
    saturation_failures = sum(
        not (
            row.get("converged")
            and _finite_number(row.get("chi"))
            and _finite_number(row.get("cf_residual"))
            and float(row["cf_residual"]) < 1.0e-10
            and _finite_number(row.get("saturation_residual"))
            and float(row["saturation_residual"]) < 1.0e-12
            and _branch_difference(row) < 1.0e-10
        )
        for row in saturation_rows
    )
    mode_failures = sum(not _mode_row_passes(row) for row in mode_rows)
    kernel_by_id = {row["point_id"]: row for row in kernel_rows}
    kernel_failures = sum(
        row.get("kernel_status") not in {"ok", "no_prograde_resonance"}
        or (
            row.get("kernel_status") == "ok"
            and not all(
                _finite_number(row.get(key))
                for key in (
                    "omega_res_M", "r_99_M", "hydrogenic_kernel_M_abs",
                    "semirelativistic_kernel_M_abs", "covariant_kernel_M_abs",
                )
            )
        )
        for row in kernel_rows
    )
    audit_failures = sum(
        not _audit_row_passes(row, pilot=row.get("point_id") in pilot_ids)
        for row in audit_rows
    )
    budget_failures = sum(
        not _budget_row_passes(
            row,
            applicable=kernel_by_id.get(row.get("point_id"), {}).get("kernel_status") == "ok",
        )
        for row in budget_rows
    )
    phenomenology_failures = sum(
        row.get("status") not in {
            "ok", "tidal_expansion_invalid", "adiabatic_tide_invalid",
            "no_prograde_resonance",
        }
        or (
            kernel_by_id.get(row.get("point_id"), {}).get("kernel_status") == "ok"
            and not all(
                _finite_number(row.get(key))
                for key in (
                    "q", "separation_M", "r_99_over_b",
                    "eta_hydrogenic_M_abs", "eta_semirelativistic_M_abs",
                    "eta_covariant_M_abs", "z_hydrogenic",
                    "z_semirelativistic", "z_covariant",
                    "depletion_hydrogenic", "depletion_semirelativistic",
                    "depletion_covariant", "power_ratio_covariant_to_hydrogenic",
                    "eta_covariant_abs_lower", "eta_covariant_abs_upper",
                    "z_covariant_lower", "z_covariant_upper",
                    "power_ratio_lower", "power_ratio_upper",
                )
            )
        )
        for row in phenomenon_rows
    )
    implementation_failures = (
        count_failures + saturation_failures + mode_failures + kernel_failures
        + audit_failures + budget_failures + phenomenology_failures
    )
    dataset_sha256 = canonical_hash(datasets)
    valid_point_ids = {
        row["point_id"] for row in phenomenon_rows if row.get("publication_valid")
    }
    no_valid_q = sorted(expected_point_ids - valid_point_ids)
    budget_by_id = {row["point_id"]: row for row in budget_rows}
    publication_kernels = [
        row for row in kernel_rows
        if row["point_id"] in valid_point_ids and row.get("kernel_status") == "ok"
    ]
    corrections = {
        row["point_id"]: abs(
            row["covariant_kernel_M_abs"] / row["hydrogenic_kernel_M_abs"] - 1.0
        )
        for row in publication_kernels
        if row.get("hydrogenic_kernel_M_abs") not in (None, 0.0)
    }
    significant_points = sorted(
        point_id for point_id, correction in corrections.items()
        if correction >= 0.10
        and correction >= 3.0 * float(
            budget_by_id.get(point_id, {}).get("worst_relative") or math.inf
        )
    )
    depletion_changes = [
        abs(float(row["depletion_covariant"]) - float(row["depletion_hydrogenic"]))
        for row in phenomenon_rows
        if row.get("publication_valid")
        and _finite_number(row.get("depletion_covariant"))
        and _finite_number(row.get("depletion_hydrogenic"))
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "atlas_certified" if implementation_failures == 0 else "atlas_failed",
        "atlas_gate_status": "atlas_certified" if implementation_failures == 0 else "implementation_failed",
        "input_certificate_status": certificate["status"],
        "implementation_failures": implementation_failures,
        "failure_counts": {
            "count": count_failures, "saturation": saturation_failures,
            "modes": mode_failures, "kernels": kernel_failures,
            "audits": audit_failures, "error_budgets": budget_failures,
            "phenomenology": phenomenology_failures,
        },
        "counts": counts, "expected_counts": EXPECTED_COUNTS,
        "code_sha256": code_sha, "config_sha256": config_sha,
        "dependency_sha256": dependency_sha, "dependencies": versions,
        "run_fingerprint": fingerprint,
        "dataset_sha256": dataset_sha256,
        "publication_valid_rows": sum(bool(row.get("publication_valid")) for row in phenomenon_rows),
        "points_with_no_valid_q": no_valid_q,
        "scientific_assessment": {
            "publication_kernel_points": len(valid_point_ids),
            "no_valid_q_points": len(no_valid_q),
            "maximum_kernel_correction": max(corrections.values(), default=0.0),
            "significant_kernel_points": significant_points,
            "maximum_depletion_change": max(depletion_changes, default=0.0),
        },
    }
    _atomic_json(output / "atlas_summary.json", summary)
    _write_report(output, summary)
    if not args.no_plots:
        _plots(output, kernel_rows, budget_rows, phenomenon_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
