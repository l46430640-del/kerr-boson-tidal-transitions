"""Validate higher tides and an independent Kerr-kernel calculation.

This runner consumes the read-only certified v2 atlas, writes a schema-v3
cache, and reports success only when every numerical and physical check passes.
Run with ``--resume`` to reuse only fingerprint-matched v3 stages.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
from dataclasses import fields
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import sph_harm_y

from boson_ep.models import (
    ContourKernelSettings,
    KerrModeResult,
    KerrModeSettings,
    MultipoleTideSettings,
    RelativisticTideSettings,
    State,
    TransitionConfig,
)
from boson_ep.kernel_validation import (
    _eta_row,
    _resonance_frequency,
    _separation,
    compute_high_order_tide_budget,
    multipole_kernel_result,
    validate_kerr_kernel,
)
from boson_ep.relativistic_tides import compute_transition_kernel


SCHEMA_VERSION = 3
ROOT = Path(__file__).resolve().parents[1]
V2_CERTIFICATE = ROOT / "results/relativistic_tides/v2/certification/pre_atlas_certificate.json"
V2_CONFIG = ROOT / "results/relativistic_tides/v2/certification/frozen_atlas_config.json"
V2_ROOT = ROOT / "results/relativistic_tides/v2"
OUTPUT_ROOT = ROOT / "results/relativistic_tides/kernel_validation"
REFERENCE = ROOT / "results/relativistic_tides/benchmarks/schwarzschild_kernel_reference.csv"
SIGNIFICANT_POINTS = (
    (0.200, State(2, 1, 1), State(2, 1, -1)),
    (0.225, State(2, 1, 1), State(2, 1, -1)),
    (0.325, State(3, 2, 2), State(3, 2, 0)),
    (0.350, State(3, 2, 2), State(3, 2, 0)),
    (0.375, State(3, 2, 2), State(3, 2, 0)),
    (0.400, State(3, 2, 2), State(3, 2, 0)),
    (0.425, State(3, 2, 2), State(3, 2, 0)),
    (0.450, State(3, 2, 2), State(3, 2, 0)),
)
INDEPENDENT_POINTS = (
    (0.200, State(2, 1, 1), State(2, 1, -1)),
    (0.250, State(3, 2, 2), State(3, 0, 0)),
    (0.350, State(3, 2, 2), State(3, 2, 0)),
    (0.400, State(3, 2, 2), State(3, 2, 0)),
    (0.450, State(4, 3, 3), State(4, 3, 1)),
)
Q_VALUES = tuple(float(value) for value in np.geomspace(1.0e-4, 1.0, 257))
VALIDITY_LEVELS = (0.05, 0.07, 0.10)


def _strict(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            return None
        return [value.real, value.imag]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_strict(item) for item in value]
    return value


def _canonical(payload: Any) -> str:
    return json.dumps(
        _strict(payload), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    )


def _hash(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_strict(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    strict_rows = _strict(list(rows))
    for row in strict_rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        if keys:
            writer.writeheader()
            for row in strict_rows:
                writer.writerow({
                    key: (
                        json.dumps(value, sort_keys=True, allow_nan=False)
                        if isinstance(value, (dict, list)) else value
                    ) for key, value in row.items()
                })
    temporary.replace(path)


def _source_hash() -> str:
    paths = [
        ROOT / "src/boson_ep/models.py",
        ROOT / "src/boson_ep/relativity.py",
        ROOT / "src/boson_ep/relativistic_tides.py",
        ROOT / "src/boson_ep/kernel_validation.py",
        ROOT / "src/boson_ep/__init__.py",
        Path(__file__).resolve(),
        REFERENCE,
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0"); digest.update(path.read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def _dependencies() -> tuple[str, dict[str, str]]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "sympy": importlib.metadata.version("sympy"),
    }
    return _hash(versions), versions


class KernelCache:
    def __init__(self, root: Path, code: str, config: str, dependencies: str):
        self.root = root
        self.code = code
        self.config = config
        self.dependencies = dependencies

    def _path(self, stage: str, point_id: str) -> Path:
        safe = point_id.replace("+", "p").replace("/", "_")
        return self.root / stage / f"{safe}.json"

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
            "code_sha256": self.code,
            "config_sha256": self.config,
            "dependency_sha256": self.dependencies,
            "stage": stage,
            "point_id": point_id,
        }
        if any(envelope.get(key) != value for key, value in expected.items()):
            return None
        payload = envelope.get("payload")
        if envelope.get("payload_sha256") != _hash(payload):
            return None
        return payload

    def store(self, stage: str, point_id: str, status: str, payload: dict[str, Any]) -> None:
        strict = _strict(payload)
        _atomic_json(self._path(stage, point_id), {
            "schema_version": SCHEMA_VERSION,
            "code_sha256": self.code,
            "config_sha256": self.config,
            "dependency_sha256": self.dependencies,
            "stage": stage,
            "point_id": point_id,
            "stage_status": status,
            "payload_sha256": _hash(strict),
            "payload": strict,
        })


def _point_id(alpha: float, initial: State, final: State) -> str:
    return f"a{alpha:.6f}_{initial.label}_{final.label}"


def _mode_filename(alpha: float, branch: State, state: State) -> str:
    point = f"a{alpha:.6f}_{branch.label}_{state.label}".replace("+", "p")
    return point + ".json"


def _state_from_label(label: str) -> State:
    if len(label) < 4 or label[2] not in "+-":
        raise ValueError(f"invalid state label: {label}")
    return State(int(label[0]), int(label[1]), int(label[2:]))


def _mode_from_payload(payload: Mapping[str, Any]) -> KerrModeResult:
    def number(key: str, default: float = math.nan) -> float:
        value = payload.get(key)
        return default if value is None else float(value)

    return KerrModeResult(
        alpha=float(payload["alpha"]), chi=float(payload["chi"]),
        state=_state_from_label(str(payload["state"])),
        frequency_M=complex(number("omega_real_M"), number("gamma_M")),
        separation_constant=complex(
            number("separation_real"), number("separation_imag")
        ),
        angular_l_values=np.asarray(payload["angular_l_values"], dtype=int),
        angular_coefficients=np.asarray(
            [complex(*item) for item in payload["angular_coefficients"]],
            dtype=complex,
        ),
        radial_coefficients=np.asarray(
            [complex(*item) for item in payload["radial_coefficients"]],
            dtype=complex,
        ),
        bilinear_norm=complex(
            number("bilinear_norm_real"), number("bilinear_norm_imag")
        ),
        r_99_M=number("r_99_M"), cf_residual=number("cf_residual", math.inf),
        radial_residual=number("radial_residual", math.inf),
        angular_residual=number("angular_residual", math.inf),
        converged=bool(payload["converged"]), message=str(payload.get("message", "")),
        selected_truncation=payload.get("selected_truncation"),
        convergence_history=tuple(payload.get("convergence_history", ())),
        radial_ode_residual=(
            None if payload.get("radial_ode_residual") is None
            else float(payload["radial_ode_residual"])
        ),
    )


def _load_v2_mode(
    atlas: Path, summary: Mapping[str, Any], alpha: float, branch: State, state: State,
) -> KerrModeResult:
    path = atlas / "cache/modes" / _mode_filename(alpha, branch, state)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 2,
        "code_sha256": summary["code_sha256"],
        "config_sha256": summary["config_sha256"],
        "dependency_sha256": summary["dependency_sha256"],
        "stage": "modes",
        "stage_status": "ok",
    }
    if any(envelope.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"uncertified v2 mode envelope: {path.name}")
    if envelope.get("payload_sha256") != _hash(envelope.get("payload")):
        raise RuntimeError(f"corrupt v2 mode payload: {path.name}")
    mode = _mode_from_payload(envelope["payload"])
    if not mode.converged:
        raise RuntimeError(f"v2 mode is not converged: {path.name}")
    return mode


def _settings(payload: Mapping[str, Any]) -> RelativisticTideSettings:
    tide_payload = dict(payload)
    mode_payload = dict(tide_payload.pop("mode"))
    mode_payload["adaptive_truncations"] = tuple(mode_payload["adaptive_truncations"])
    mode_names = {item.name for item in fields(KerrModeSettings)}
    tide_names = {item.name for item in fields(RelativisticTideSettings)}
    mode = KerrModeSettings(**{
        key: value for key, value in mode_payload.items() if key in mode_names
    })
    tide_payload["horizon_cutoffs"] = tuple(tide_payload["horizon_cutoffs"])
    return RelativisticTideSettings(mode=mode, **{
        key: value for key, value in tide_payload.items()
        if key in tide_names and key != "mode"
    })


def _potential_reconstruction() -> dict[str, Any]:
    random = np.random.default_rng(20260715)
    residuals = []
    for _ in range(24):
        rho = float(random.uniform(1.0e-4, 0.2))
        theta = float(random.uniform(0.05, math.pi - 0.05))
        phi = float(random.uniform(-math.pi, math.pi))
        direct = 1.0 / math.sqrt(
            1.0 + rho * rho - 2.0 * rho * math.sin(theta) * math.cos(phi)
        )
        reconstructed = 0.0j
        for ell in range(21):
            harmonic_sum = sum(
                sph_harm_y(ell, m, theta, phi)
                * np.conjugate(sph_harm_y(ell, m, math.pi / 2.0, 0.0))
                for m in range(-ell, ell + 1)
            )
            reconstructed += 4.0 * math.pi / (2 * ell + 1) * rho**ell * harmonic_sum
        residuals.append(abs(reconstructed - direct) / direct)
    maximum = float(max(residuals))
    return {"samples": len(residuals), "maximum_relative_residual": maximum,
            "passed": maximum < 1.0e-10}


def _refinements(
    config: TransitionConfig, pair: tuple[KerrModeResult, KerrModeResult],
    multipole_settings: MultipoleTideSettings,
) -> list[dict[str, Any]]:
    kernel = compute_transition_kernel(config, pair)
    omega = _resonance_frequency(pair)
    r99 = max(pair[0].r_99_M, pair[1].r_99_M)
    rows = []
    for orbital_model in ("newtonian", "1pn"):
        boundaries: dict[str, float | None] = {}
        for level in VALIDITY_LEVELS:
            function = lambda log_q: (
                r99 / _separation(omega, math.exp(log_q), orbital_model) - level
            )
            lo, hi = math.log(1.0e-4), math.log(1.0)
            if function(lo) * function(hi) <= 0.0:
                boundaries[f"q_r99_over_b_{level:.2f}"] = math.exp(
                    brentq(function, lo, hi, xtol=1.0e-13, rtol=1.0e-13)
                )
            else:
                boundaries[f"q_r99_over_b_{level:.2f}"] = None

        def objective(log_q: float) -> float:
            row = _eta_row(
                config, pair, math.exp(log_q), orbital_model,
                multipole_settings, complex(kernel.covariant_kernel_M),
                complex(kernel.hydrogenic_kernel_M),
            )
            return -row.depletion_change if row.r_99_over_b <= 0.10 else 1.0

        optimum = minimize_scalar(
            objective, bounds=(math.log(1.0e-4), math.log(1.0)),
            method="bounded", options={"xatol": 1.0e-10},
        )
        q_peak = math.exp(float(optimum.x))
        peak = _eta_row(
            config, pair, q_peak, orbital_model, multipole_settings,
            complex(kernel.covariant_kernel_M), complex(kernel.hydrogenic_kernel_M),
        )
        rows.append({
            "point_id": _point_id(config.alpha, config.initial, config.final),
            "orbital_model": orbital_model,
            "q_depletion_extremum": q_peak,
            "depletion_change_extremum": peak.depletion_change,
            "r_99_over_b_at_extremum": peak.r_99_over_b,
            "optimizer_success": bool(optimum.success),
            **boundaries,
        })
    return rows


def _peak_rss() -> int | None:
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = ctypes.c_void_p
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = (
            ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong,
        )
        get_memory.restype = ctypes.c_int
        handle = get_process()
        if get_memory(
            handle, ctypes.byref(counters), counters.cb
        ):
            return int(counters.PeakWorkingSetSize)
        return None
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, OSError):
        return None


def _plots(
    output: Path, multipoles: list[dict[str, Any]],
    independent: list[dict[str, Any]], theory: list[dict[str, Any]],
) -> None:
    even = [row for row in multipoles if int(row["ell"]) % 2 == 0]
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    for point in sorted({row["point_id"] for row in even}):
        local = sorted((row for row in even if row["point_id"] == point), key=lambda row: row["ell"])
        scan = min(
            (row for row in theory
             if row.get("row_type") == "q_scan"
             and row["point_id"] == point
             and row["orbital_model"] == "newtonian"),
            key=lambda row: abs(math.log10(float(row["q"])) + 3.0),
        )
        q_value = float(scan["q"]); separation = float(scan["separation_M"])
        contributions = [
            q_value / separation ** (int(row["ell"]) + 1)
            * float(row["kernel_M_abs"])
            for row in local
        ]
        reference = max(contributions[0], 1.0e-300)
        axis.semilogy([row["ell"] for row in local],
                      [value / reference for value in contributions],
                      marker="o", linewidth=1.0, label=point)
    axis.set(xlabel="multipole ell", ylabel="|eta_ell| / |eta_2| at q=1e-3",
             title="Exact Newtonian multipole convergence")
    axis.grid(alpha=0.25); axis.legend(fontsize=6, ncol=2)
    figure.tight_layout(); figure.savefig(output / "multipole_convergence.png", dpi=180); plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    x = np.arange(len(independent))
    axis.semilogy(x, [row["amplitude_relative_difference"] for row in independent], "o", label="amplitude")
    axis.semilogy(x, [row["phase_difference"] for row in independent], "s", label="phase")
    axis.axhline(5.0e-3, color="black", linestyle="--", linewidth=1.0)
    axis.set_xticks(x, [row["point_id"] for row in independent], rotation=30, ha="right", fontsize=7)
    axis.set(ylabel="relative/radian difference", title="Independent Kerr-kernel cross-check")
    axis.grid(alpha=0.25); axis.legend(); figure.tight_layout()
    figure.savefig(output / "kernel_crosscheck.png", dpi=180); plt.close(figure)

    robust = [row for row in theory if row.get("row_type") == "q_scan"]
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    for point in sorted({row["point_id"] for row in robust}):
        local = [
            row for row in robust
            if row["point_id"] == point
            and row["orbital_model"] == "newtonian"
            and row["passes_strict_domain"]
        ]
        axis.semilogx([row["q"] for row in local], [row["depletion_change"] for row in local], label=point)
    axis.axhline(0.10, color="black", linestyle="--", linewidth=1.0)
    axis.set(xlabel="q", ylabel="|Delta depletion|", title="Robust effect regions")
    axis.grid(alpha=0.25); axis.legend(fontsize=6, ncol=2); figure.tight_layout()
    figure.savefig(output / "robust_effect_regions.png", dpi=180); plt.close(figure)


def _report(output: Path, certificate: Mapping[str, Any]) -> None:
    gate = certificate["validation_gate"]
    lines = [
        "# Higher-Tide and Independent Kerr-Kernel Validation", "",
        f"- 状态：`{certificate['status']}`",
        f"- 高阶潮汐失败：`{certificate['high_order_tide_failures']}`",
        f"- 独立 kernel 失败：`{certificate['independent_kernel_failures']}`",
        f"- 墙钟时间：`{certificate['wall_time_seconds']:.1f} s`", "",
        "## 数值结论", "",
        f"- 点伴星势重构最大残差：`{gate['potential_reconstruction_residual']:.3e}`",
        f"- 奇数 electric multipole 最大相对幅值：`{gate['maximum_odd_relative_amplitude']:.3e}`",
        f"- `ell=8/10` 最大尾项：`{gate['maximum_l8_l10_relative_difference']:.3e}`",
        f"- Gauss/quad 最大差异：`{gate['maximum_gauss_quad_difference']:.3e}`",
        f"- 独立 kernel 最大幅值差：`{gate['maximum_independent_amplitude_difference']:.3e}`",
        f"- 独立 kernel 最大相位差：`{gate['maximum_independent_phase_difference']:.3e} rad`", "",
        f"- Schwarzschild `alpha=0.15, 211->31-1` control difference: `{float(certificate['schwarzschild_control']['absolute_difference']):.3e}`", "",
        "## Physical thresholds", "",
        f"- `r99/b<=0.07` 且保留 10% kernel 修正：`{gate['ten_percent_effect_in_strict_domain']}`",
        f"- 修正超过三倍高阶潮汐包络：`{gate['effect_exceeds_three_sigma_proxy']}`",
        f"- `alpha=0.35, 322->320` 精确多极修正：`{gate['alpha035_multipole_shift']:.3%}`",
        f"- Newtonian 与 1PN 均有 10% depletion 改变：`{gate['depletion_abstract_gate']}`", "",
        "The validated quantities use local adiabatic Kerr quadrupolar tides; covariant higher multipoles are represented by the reported conservative envelope.",
    ]
    temporary = output / "kernel_validation_report.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output / "kernel_validation_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quick", action="store_true", help="one-point smoke test; never certifies")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); started = time.perf_counter()
    certificate_v2 = json.loads(V2_CERTIFICATE.read_text(encoding="utf-8"))
    frozen = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    if certificate_v2.get("status") != "pre_atlas_certified" or certificate_v2.get("implementation_failures") != 0:
        raise SystemExit("kernel validation blocked: v2 pre-atlas certificate is not clean")
    atlas = V2_ROOT / str(certificate_v2["run_fingerprint"])
    summary = json.loads((atlas / "atlas_summary.json").read_text(encoding="utf-8"))
    if summary.get("atlas_gate_status") != "atlas_certified" or summary.get("implementation_failures") != 0:
        raise SystemExit("kernel validation blocked: v2 atlas is not certified")
    if summary.get("run_fingerprint") != certificate_v2.get("run_fingerprint"):
        raise SystemExit("kernel validation blocked: v2 fingerprint mismatch")
    settings = _settings(frozen["settings"])
    points = SIGNIFICANT_POINTS[:1] if args.quick else SIGNIFICANT_POINTS
    independent_points = INDEPENDENT_POINTS[:1] if args.quick else INDEPENDENT_POINTS
    validation_config = {
        "schema_version": SCHEMA_VERSION,
        "v2_run_fingerprint": summary["run_fingerprint"],
        "v2_dataset_sha256": summary["dataset_sha256"],
        "significant_points": [(a, i.label, f.label) for a, i, f in points],
        "independent_points": [(a, i.label, f.label) for a, i, f in independent_points],
        "q_values": Q_VALUES,
        "validity_levels": VALIDITY_LEVELS,
        "multipole": MultipoleTideSettings().__dict__,
        "contour": ContourKernelSettings().__dict__,
    }
    config_sha = _hash(validation_config); code_sha = _source_hash()
    dependency_sha, versions = _dependencies()
    fingerprint = _hash({
        "schema_version": SCHEMA_VERSION, "code_sha256": code_sha,
        "config_sha256": config_sha, "dependency_sha256": dependency_sha,
        "v2_dataset_sha256": summary["dataset_sha256"],
    })
    output = OUTPUT_ROOT / fingerprint; output.mkdir(parents=True, exist_ok=True)
    cache = KernelCache(output / "cache", code_sha, config_sha, dependency_sha)
    multipole_settings = MultipoleTideSettings()

    reconstruction = _potential_reconstruction()
    all_modes: dict[str, tuple[KerrModeResult, KerrModeResult]] = {}
    multipole_rows: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []
    refinement_rows: list[dict[str, Any]] = []
    budget_summaries: list[dict[str, Any]] = []
    for alpha, initial, final in points:
        point_id = _point_id(alpha, initial, final)
        pair = (
            _load_v2_mode(atlas, summary, alpha, initial, initial),
            _load_v2_mode(atlas, summary, alpha, initial, final),
        )
        all_modes[point_id] = pair
        payload = None if args.overwrite or not args.resume else cache.load("high_order", point_id)
        if payload is None:
            config = TransitionConfig(alpha, initial, final, q=1.0e-3,
                                      chi=pair[0].chi, settings=settings)
            budget = compute_high_order_tide_budget(
                config, pair, Q_VALUES, multipole_settings
            )
            local_multipoles = []
            for ell in range(2, 11):
                row = multipole_kernel_result(config, pair, ell, multipole_settings).to_dict()
                row["point_id"] = point_id; local_multipoles.append(row)
            local_refinements = _refinements(config, pair, multipole_settings)
            payload = {
                "point_id": point_id, "status": budget.status,
                "budget": budget.to_dict(include_rows=False),
                "multipoles": local_multipoles,
                "q_rows": [row.to_dict() for row in budget.rows],
                "refinements": local_refinements,
            }
            cache.store("high_order", point_id, budget.status, payload)
        multipole_rows.extend(payload["multipoles"])
        q_rows.extend(payload["q_rows"])
        refinement_rows.extend(payload["refinements"])
        budget_summaries.append(payload["budget"])

    independent_rows: list[dict[str, Any]] = []
    for alpha, initial, final in independent_points:
        point_id = _point_id(alpha, initial, final)
        pair = all_modes.get(point_id) or (
            _load_v2_mode(atlas, summary, alpha, initial, initial),
            _load_v2_mode(atlas, summary, alpha, initial, final),
        )
        payload = None if args.overwrite or not args.resume else cache.load("independent", point_id)
        if payload is None:
            config = TransitionConfig(alpha, initial, final, q=1.0e-3,
                                      chi=pair[0].chi, settings=settings)
            result = validate_kerr_kernel(config, pair)
            payload = result.to_dict(); cache.store("independent", point_id, result.status, payload)
        independent_rows.append(payload)

    benchmark_rows = list(csv.DictReader(
        (V2_CERTIFICATE.parent / "schwarzschild_benchmark.csv").open(encoding="utf-8")
    ))
    schwarzschild = next(row for row in benchmark_rows if abs(float(row["alpha"]) - 0.15) < 1.0e-12)
    reference_pass = bool(str(schwarzschild["passed"]).lower() == "true" and
                          float(schwarzschild["absolute_difference"]) < max(
                              0.02, 2.0 * float(schwarzschild["sigma_digitization"])))
    schwarzschild_row = {**schwarzschild, "status": "ok" if reference_pass else "kernel_crosscheck_failed"}

    maximum_odd = 0.0; maximum_gauss_quad = 0.0
    for point_id in {row["point_id"] for row in multipole_rows}:
        local = [row for row in multipole_rows if row["point_id"] == point_id]
        quadrupole = max(float(next(row for row in local if int(row["ell"]) == 2)["kernel_M_abs"]), 1.0e-300)
        maximum_odd = max(maximum_odd, max(
            float(row["kernel_M_abs"]) / quadrupole for row in local if int(row["ell"]) % 2
        ))
        maximum_gauss_quad = max(maximum_gauss_quad, max(
            float(row["gauss_quad_relative_difference"]) for row in local
        ))
    maximum_tail = max(float(row["maximum_tail_relative"]) for row in budget_summaries)

    theory_rows: list[dict[str, Any]] = []
    robust_points: set[str] = set()
    three_sigma_points: set[str] = set()
    v2_kernel_rows = json.loads((atlas / "kernels.json").read_text(encoding="utf-8"))
    central_kernels = {row["point_id"]: row for row in v2_kernel_rows}
    for row in q_rows:
        point_id = row["point_id"]
        kernel = central_kernels[point_id]
        eta_h = float(row["epsilon_2"]) * complex(
            kernel["hydrogenic_kernel_M_real"], kernel["hydrogenic_kernel_M_imag"]
        )
        eta_c = complex(row["eta_covariant_corrected_M_real"], row["eta_covariant_corrected_M_imag"])
        physical_shift = abs(eta_c - eta_h)
        envelope = float(row["sigma_multipole_abs"]) + float(row["sigma_covariant_high_abs"])
        correction = physical_shift / max(abs(eta_h), 1.0e-300)
        passes_domain = float(row["r_99_over_b"]) <= 0.07
        passes_effect = correction >= 0.10
        passes_envelope = physical_shift >= 3.0 * envelope
        if passes_domain and passes_effect:
            robust_points.add(point_id)
        if passes_domain and passes_effect and passes_envelope:
            three_sigma_points.add(point_id)
        validation_status = (
            "robust_physical_effect"
            if passes_domain and passes_effect and passes_envelope
            else "theory_systematics_dominated"
            if passes_domain and passes_effect
            else str(row["status"])
        )
        theory_rows.append({
            **row, "row_type": "q_scan", "kernel_correction": correction,
            "physical_eta_shift_abs": physical_shift,
            "high_order_envelope_abs": envelope,
            "passes_strict_domain": passes_domain,
            "passes_ten_percent": passes_effect,
            "passes_three_sigma_proxy": passes_envelope,
            "validation_status": validation_status,
        })
    theory_rows.extend({**row, "row_type": "refinement"} for row in refinement_rows)

    alpha035 = next(
        (row for row in budget_summaries if row["point_id"].startswith("a0.350000_")),
        budget_summaries[0],
    )
    depletion_by_orbit = {
        model: max(
            float(row["depletion_change_extremum"]) for row in refinement_rows
            if row["orbital_model"] == model
        ) for model in ("newtonian", "1pn")
    }
    independent_failures = sum(row["status"] != "ok" for row in independent_rows)
    high_order_checks = {
        "potential_reconstruction": reconstruction["passed"],
        "odd_multipoles": maximum_odd < 1.0e-12,
        "l8_l10": maximum_tail < 1.0e-4,
        "gauss_quad": maximum_gauss_quad < 1.0e-4,
        "schwarzschild_control": reference_pass,
        "strict_domain_effect": bool(robust_points),
        "three_sigma_proxy": bool(three_sigma_points),
        "alpha035_multipole": float(alpha035["maximum_multipole_relative_shift"]) < 0.03,
    }
    high_order_failures = sum(not value for value in high_order_checks.values())
    memory = _peak_rss(); elapsed = time.perf_counter() - started
    quick_block = bool(args.quick)
    passed = high_order_failures == 0 and independent_failures == 0 and not quick_block
    validation_gate = {
        "potential_reconstruction_residual": reconstruction["maximum_relative_residual"],
        "maximum_odd_relative_amplitude": maximum_odd,
        "maximum_l8_l10_relative_difference": maximum_tail,
        "maximum_gauss_quad_difference": maximum_gauss_quad,
        "maximum_independent_amplitude_difference": max(
            float(row["amplitude_relative_difference"]) for row in independent_rows
        ),
        "maximum_independent_phase_difference": max(
            float(row["phase_difference"]) for row in independent_rows
        ),
        "ten_percent_effect_in_strict_domain": sorted(robust_points),
        "effect_exceeds_three_sigma_proxy": sorted(three_sigma_points),
        "alpha035_multipole_shift": float(alpha035["maximum_multipole_relative_shift"]),
        "depletion_extrema": depletion_by_orbit,
        "depletion_abstract_gate": all(value >= 0.10 for value in depletion_by_orbit.values()),
        "checks": high_order_checks,
    }
    certificate = {
        "schema_version": SCHEMA_VERSION,
        "status": "kernel_validation_passed" if passed else (
            "quick_smoke_only" if quick_block else "kernel_validation_failed"
        ),
        "high_order_tide_failures": high_order_failures,
        "independent_kernel_failures": independent_failures,
        "code_sha256": code_sha, "config_sha256": config_sha,
        "dependency_sha256": dependency_sha, "dependencies": versions,
        "run_fingerprint": fingerprint,
        "input_v2_run_fingerprint": summary["run_fingerprint"],
        "input_v2_dataset_sha256": summary["dataset_sha256"],
        "counts": {
            "significant_points": len(points), "multipole_kernels": len(multipole_rows),
            "exact_tide_qscan": len(q_rows), "independent_kernels": len(independent_rows),
        },
        "validation_gate": validation_gate,
        "schwarzschild_control": schwarzschild_row,
        "failures": [key for key, value in high_order_checks.items() if not value] + [
            row["point_id"] for row in independent_rows if row["status"] != "ok"
        ],
        "peak_resident_memory_bytes": memory,
        "wall_time_seconds": elapsed,
    }

    _atomic_csv(output / "multipole_kernels.csv", multipole_rows)
    _atomic_json(output / "multipole_kernels.json", multipole_rows)
    _atomic_csv(output / "exact_tide_qscan.csv", q_rows)
    _atomic_json(output / "exact_tide_qscan.json", q_rows)
    _atomic_csv(output / "independent_kernels.csv", independent_rows)
    _atomic_json(output / "independent_kernels.json", independent_rows)
    _atomic_csv(output / "theory_error_budget.csv", theory_rows)
    _atomic_json(output / "theory_error_budget.json", theory_rows)
    _atomic_json(output / "kernel_validation_certificate.json", certificate)
    _report(output, certificate)
    if not args.no_plots:
        _plots(output, multipole_rows, independent_rows, theory_rows)
    print(json.dumps({
        "status": certificate["status"], "output": str(output),
        "high_order_tide_failures": high_order_failures,
        "independent_kernel_failures": independent_failures,
    }, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
