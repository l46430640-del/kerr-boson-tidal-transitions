"""Run the mandatory certification gate for the relativistic tide atlas."""

from __future__ import annotations

import argparse
import csv
import ctypes
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import tracemalloc
from typing import Any, Callable

import numpy as np
import scipy
import sympy

from boson_ep.certification import (
    compare_schwarzschild_benchmark,
    fit_weak_field_limit,
    operator_form_audit,
    ward_identity_audit,
)
from boson_ep.models import (
    GaugeVectorSpec,
    KerrModeSettings,
    RelativisticTideSettings,
    State,
    TransitionConfig,
)
from boson_ep.relativity import saturation_spin_cf, solve_kerr_mode
from boson_ep.relativistic_tides import (
    TRANSITION_CHANNELS,
    hydrogenic_newtonian_kernel_M,
    schwarzschild_irg_tidal_kernel_M,
    schwarzschild_newtonian_tidal_kernel_M,
    schwarzschild_rw_tidal_kernel_M,
    tidal_kernel_from_modes_M,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "relativistic_tides" / "certification"
REFERENCE_PATH = ROOT / "results/relativistic_tides/benchmarks/schwarzschild_kernel_reference.csv"

PILOTS = (
    (0.10, State(2, 1, 1), State(2, 1, -1)),
    (0.25, State(3, 2, 2), State(3, 0, 0)),
    (0.45, State(4, 3, 3), State(4, 1, 1)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="reuse completed atomic cache entries")
    parser.add_argument("--quick", action="store_true", help="smoke test only; can never certify")
    parser.add_argument("--overwrite", action="store_true", help="discard certification cache")
    return parser.parse_args()


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, complex):
        return {"real": _finite_json(value.real), "imag": _finite_json(value.imag)}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_finite_json(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        if keys:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _peak_rss_bytes() -> int | None:
    """Return the OS-reported peak resident working set without dependencies."""
    if sys.platform == "win32":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = ctypes.c_void_p
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
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

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except (ImportError, OSError):
        return None


def _settings_payload(settings: RelativisticTideSettings) -> dict[str, Any]:
    payload = asdict(settings)
    payload["horizon_cutoffs"] = list(settings.horizon_cutoffs)
    return payload


def _settings(
    *,
    truncation: int,
    radial_nodes: int,
    angular_nodes: int,
    cutoff: float,
    counterterm_order: int,
    quick: bool = False,
    mode_angular_nodes: int | None = None,
) -> RelativisticTideSettings:
    mode = KerrModeSettings(
        truncation=truncation,
        series_terms=truncation,
        angular_lmax=10 if quick else 14,
        angular_nodes=mode_angular_nodes or angular_nodes,
        horizon_cutoff=cutoff,
        counterterm_order=counterterm_order,
        radial_rtol=1.0e-8,
    )
    return RelativisticTideSettings(
        mode=mode,
        radial_rtol=3.0e-5 if quick else 3.0e-6,
        radial_atol=1.0e-9 if quick else 1.0e-10,
        radial_nodes=radial_nodes,
        angular_nodes=angular_nodes,
        horizon_cutoffs=(cutoff,),
    )


class ResumeCache:
    def __init__(self, path: Path, enabled: bool, code_hash: str) -> None:
        self.path = path
        self.enabled = enabled
        if enabled and path.exists():
            self.rows: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            if self.rows.get("__code_sha256__") != code_hash:
                self.rows = {"__code_sha256__": code_hash}
        else:
            self.rows = {"__code_sha256__": code_hash}
        self.prior_wall_time_seconds = float(
            self.rows.get("__cumulative_wall_time_seconds__", 0.0)
        )

    def get_or_run(self, key: str, function: Callable[[], Any]) -> Any:
        if self.enabled and key in self.rows:
            return self.rows[key]
        value = function()
        self.rows[key] = _finite_json(value)
        _atomic_json(self.path, self.rows)
        return self.rows[key]


def _weak_field_rows(
    settings: RelativisticTideSettings,
    alphas: tuple[float, ...],
    cache: ResumeCache,
) -> list[dict[str, Any]]:
    initial = State(2, 1, 1)
    final = State(2, 1, -1)
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        key = f"weak_{alpha:.6f}"

        def calculate(alpha: float = alpha) -> dict[str, Any]:
            hydrogenic = hydrogenic_newtonian_kernel_M(alpha, initial, final)
            irg = schwarzschild_irg_tidal_kernel_M(alpha, initial, final, settings)
            rw = schwarzschild_rw_tidal_kernel_M(alpha, initial, final, settings)
            gauge = abs(irg - rw) / max(abs(rw), 1.0e-300)
            return {
                "alpha": alpha,
                "hydrogenic_kernel_abs": abs(hydrogenic),
                "irg_kernel_abs": abs(irg),
                "rw_kernel_abs": abs(rw),
                "kernel_ratio": abs(irg) / max(abs(hydrogenic), 1.0e-300),
                "gauge_residual": gauge,
            }

        rows.append(cache.get_or_run(key, calculate))
    return rows


def _benchmark_rows(
    settings: RelativisticTideSettings,
    alphas: tuple[float, ...],
    cache: ResumeCache,
) -> list[dict[str, Any]]:
    initial = State(2, 1, 1)
    final = State(3, 1, -1)
    gauge_final = State(2, 1, -1)
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        key = f"benchmark_{alpha:.6f}"

        def calculate(alpha: float = alpha) -> dict[str, Any]:
            hydrogenic = hydrogenic_newtonian_kernel_M(alpha, initial, final)
            semirelativistic = schwarzschild_newtonian_tidal_kernel_M(
                alpha, initial, final, settings
            )
            # Cannizzaro's benchmark isolates relativistic QBS wavefunctions
            # under the Newtonian quadrupole.  The independent IRG/RW check is
            # therefore performed on the weak-field hyperfine channel rather
            # than mixed into the benchmark observable.
            irg = schwarzschild_irg_tidal_kernel_M(
                alpha, initial, gauge_final, settings
            )
            rw = schwarzschild_rw_tidal_kernel_M(
                alpha, initial, gauge_final, settings
            )
            return {
                "alpha": alpha,
                "benchmark_channel": f"{initial.label}->{final.label}",
                "gauge_channel": f"{initial.label}->{gauge_final.label}",
                "hydrogenic_kernel_abs": abs(hydrogenic),
                "schwarzschild_newtonian_kernel_abs": abs(semirelativistic),
                "relative_correction": abs(
                    1.0 - abs(semirelativistic) / abs(hydrogenic)
                ),
                "gauge_residual": abs(irg - rw) / max(abs(rw), 1.0e-300),
            }

        rows.append(cache.get_or_run(key, calculate))
    return rows


def _ward_rows(
    settings: RelativisticTideSettings,
    pilots: tuple[tuple[float, State, State], ...],
    cache: ResumeCache,
    quick: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ward_settings = replace(
        settings,
        radial_nodes=min(settings.radial_nodes, 32),
        angular_nodes=min(settings.angular_nodes, 24),
    )
    summary: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for alpha, initial, final in pilots:
        key = f"ward_{alpha:.6f}_{initial.label}_{final.label}"

        def calculate(
            alpha: float = alpha, initial: State = initial, final: State = final
        ) -> dict[str, Any]:
            chi = saturation_spin_cf(alpha, initial, ward_settings.mode)
            initial_mode = solve_kerr_mode(alpha, chi, initial, ward_settings.mode)
            final_mode = solve_kerr_mode(alpha, chi, final, ward_settings.mode)
            r_plus = 1.0 + math.sqrt(1.0 - chi**2)
            r_99 = max(initial_mode.r_99_M, final_mode.r_99_M)
            supports = (
                ("near_horizon", r_plus + 2.0, r_plus + 10.0),
                ("cloud_core", max(r_plus + 0.2, 0.2 * r_99), 0.7 * r_99),
            )
            if quick:
                supports = supports[:1]
            kinds = ("temporal",) if quick else ("temporal", "radial", "polar", "axial")
            amplitudes = (0.1, 1.0)
            specs = tuple(
                GaugeVectorSpec(kind, left, right, amplitude, support_name=name)
                for kind in kinds
                for name, left, right in supports
                for amplitude in amplitudes
                if right > left
            )
            result = ward_identity_audit(
                TransitionConfig(
                    alpha, initial, final, chi=chi, settings=ward_settings
                ),
                specs,
            )
            return result.to_dict(include_rows=True)

        result = cache.get_or_run(key, calculate)
        detail_rows = result.pop("rows", []) if "rows" in result else []
        summary.append(dict(result))
        details.extend(detail_rows)
    return summary, details


def _operator_rows(
    settings: RelativisticTideSettings,
    pilots: tuple[tuple[float, State, State], ...],
    cache: ResumeCache,
) -> tuple[float, list[dict[str, Any]]]:
    maximum = 0.0
    rows: list[dict[str, Any]] = []
    for alpha, initial, final in pilots:
        key = f"operator_{alpha:.6f}_{initial.label}_{final.label}"

        def calculate(
            alpha: float = alpha, initial: State = initial, final: State = final
        ) -> dict[str, Any]:
            chi = saturation_spin_cf(alpha, initial, settings.mode)
            r_plus = 1.0 + math.sqrt(1.0 - chi**2)
            points = tuple(
                (r_plus + offset, theta)
                for offset, theta in ((2.0, 0.7), (4.0, 1.0), (7.0, 1.3), (10.0, 1.8))
            )
            residual, local = operator_form_audit(
                TransitionConfig(alpha, initial, final, chi=chi, settings=settings),
                points,
            )
            return {"maximum": residual, "rows": local, "chi": chi}

        result = cache.get_or_run(key, calculate)
        value = result.get("maximum")
        maximum = max(maximum, math.inf if value is None else float(value))
        for row in result.get("rows", []):
            rows.append({"alpha": alpha, "initial": initial.label, "final": final.label, **row})
    return maximum, rows


def _evaluate_pilot(
    alpha: float,
    initial: State,
    final: State,
    settings: RelativisticTideSettings,
    radial_strategy: str,
) -> dict[str, Any]:
    chi = saturation_spin_cf(alpha, initial, settings.mode)
    mode_i = solve_kerr_mode(alpha, chi, initial, settings.mode)
    mode_f = solve_kerr_mode(alpha, chi, final, settings.mode)
    omega_res = (mode_i.frequency_M.real - mode_f.frequency_M.real) / (
        initial.m - final.m
    )
    kernel = tidal_kernel_from_modes_M(
        mode_i,
        mode_f,
        omega_res,
        settings,
        "irg",
        radial_strategy=radial_strategy,
    )
    return {
        "alpha": alpha,
        "chi": chi,
        "initial": initial.label,
        "final": final.label,
        "frequency_i_real": mode_i.frequency_M.real,
        "frequency_i_imag": mode_i.frequency_M.imag,
        "frequency_f_real": mode_f.frequency_M.real,
        "frequency_f_imag": mode_f.frequency_M.imag,
        "cf_residual": max(mode_i.cf_residual, mode_f.cf_residual),
        "radial_residual": max(mode_i.radial_residual, mode_f.radial_residual),
        "angular_residual": max(mode_i.angular_residual, mode_f.angular_residual),
        "bilinear_norm_error": max(
            abs(mode_i.bilinear_norm - 1.0), abs(mode_f.bilinear_norm - 1.0)
        ),
        "kernel_real": kernel.real,
        "kernel_imag": kernel.imag,
        "kernel_abs": abs(kernel),
        "radial_strategy": radial_strategy,
    }


def _convergence_rows(
    reference: RelativisticTideSettings,
    pilots: tuple[tuple[float, State, State], ...],
    cache: ResumeCache,
    quick: bool,
) -> list[dict[str, Any]]:
    dimensions: tuple[tuple[str, tuple[Any, ...]], ...]
    if quick:
        dimensions = (("N", (80, 100)), ("radial_nodes", (8, 12)))
    else:
        dimensions = (
            ("N", (150, 250, 400)),
            ("radial_nodes", (40, 56, 80)),
            ("angular_nodes", (48, 72, 96)),
            ("horizon_cutoff", (3.0e-4, 1.0e-4, 3.0e-5)),
            ("counterterm_order", (2, 3, 4)),
        )
    rows: list[dict[str, Any]] = []
    for alpha, initial, final in pilots:
        for dimension, values in dimensions:
            for value in values:
                mode = reference.mode
                local = reference
                if dimension == "N":
                    mode = replace(mode, truncation=int(value), series_terms=int(value))
                    local = replace(local, mode=mode)
                elif dimension == "radial_nodes":
                    local = replace(local, radial_nodes=int(value))
                elif dimension == "angular_nodes":
                    mode = replace(mode, angular_nodes=int(value))
                    local = replace(local, mode=mode, angular_nodes=int(value))
                elif dimension == "horizon_cutoff":
                    mode = replace(mode, horizon_cutoff=float(value))
                    local = replace(local, mode=mode, horizon_cutoffs=(float(value),))
                else:
                    mode = replace(mode, counterterm_order=int(value))
                    local = replace(local, mode=mode)
                settings_hash = _canonical_hash(_settings_payload(local))[:16]
                key = (
                    f"conv_{alpha:.6f}_{initial.label}_{final.label}_"
                    f"{settings_hash}_gauss"
                )
                result = cache.get_or_run(
                    key,
                    lambda alpha=alpha, initial=initial, final=final, local=local:
                    _evaluate_pilot(alpha, initial, final, local, "gauss"),
                )
                rows.append({"dimension": dimension, "value": value, **result})
        key = f"conv_{alpha:.6f}_{initial.label}_{final.label}_reference_quad"
        result = cache.get_or_run(
            key,
            lambda alpha=alpha, initial=initial, final=final:
            _evaluate_pilot(alpha, initial, final, reference, "quad"),
        )
        rows.append({"dimension": "radial_strategy", "value": "quad", **result})
    return rows


def _relative(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), 1.0e-300)


def _assess_convergence(rows: list[dict[str, Any]], quick: bool) -> tuple[bool, list[str], dict[str, Any]]:
    if quick:
        return False, ["quick mode cannot certify"], {}
    failures: list[str] = []
    selections = {
        "N": 250,
        "radial_nodes": 56,
        "angular_nodes": 72,
        "horizon_cutoff": 1.0e-4,
        "counterterm_order": 3,
    }
    thresholds = {
        "N": 3.0e-4,
        "radial_nodes": 1.0e-4,
        "angular_nodes": 1.0e-4,
        "horizon_cutoff": 1.0e-5,
        "counterterm_order": 1.0e-5,
    }
    high = {
        "N": 400,
        "radial_nodes": 80,
        "angular_nodes": 96,
        "horizon_cutoff": 3.0e-5,
        "counterterm_order": 4,
    }
    groups = sorted({(row["alpha"], row["initial"], row["final"]) for row in rows})

    def mode_quality_passes(row: dict[str, Any]) -> bool:
        return all(
            row.get(key) is not None
            and math.isfinite(float(row[key]))
            and float(row[key]) < limit
            for key, limit in (
                ("cf_residual", 1.0e-10),
                ("radial_residual", 1.0e-8),
                ("angular_residual", 1.0e-8),
                ("bilinear_norm_error", 1.0e-5),
            )
        )

    for group in groups:
        selected_rows = [
            row for row in rows
            if (row["alpha"], row["initial"], row["final"]) == group
        ]
        for row in selected_rows:
            for key in (
                "cf_residual", "radial_residual", "angular_residual",
                "bilinear_norm_error", "kernel_abs",
            ):
                value = row.get(key)
                if value is None or not math.isfinite(float(value)):
                    failures.append(
                        f"{group} {row['dimension']}={row['value']} missing/nonfinite {key}"
                    )
        for dimension, preferred in selections.items():
            low_row = next(
                (row for row in selected_rows if row["dimension"] == dimension and float(row["value"]) == preferred),
                None,
            )
            high_row = next(
                (row for row in selected_rows if row["dimension"] == dimension and float(row["value"]) == high[dimension]),
                None,
            )
            if low_row is None or high_row is None:
                failures.append(f"{group} missing {dimension} convergence pair")
                continue
            difference = _relative(float(low_row["kernel_abs"]), float(high_row["kernel_abs"]))
            low_row["relative_to_reference"] = difference
            if not mode_quality_passes(high_row):
                failures.append(f"{group} reference {dimension} mode quality")
            if (
                difference >= thresholds[dimension]
                or not mode_quality_passes(low_row)
            ):
                selections[dimension] = high[dimension]
        n250 = next(row for row in selected_rows if row["dimension"] == "N" and float(row["value"]) == 250)
        n400 = next(row for row in selected_rows if row["dimension"] == "N" and float(row["value"]) == 400)
        for suffix in ("i", "f"):
            real_difference = _relative(
                float(n250[f"frequency_{suffix}_real"]), float(n400[f"frequency_{suffix}_real"])
            )
            if real_difference >= 1.0e-8:
                failures.append(f"{group} N frequency_{suffix}_real")
            gamma_250 = float(n250[f"frequency_{suffix}_imag"])
            gamma_400 = float(n400[f"frequency_{suffix}_imag"])
            if abs(gamma_400) < 1.0e-14:
                gamma_failed = abs(gamma_250 - gamma_400) >= 1.0e-15
            else:
                gamma_failed = _relative(gamma_250, gamma_400) >= 1.0e-3
            if gamma_failed:
                failures.append(f"{group} N frequency_{suffix}_imag")
        gauss_reference = next(
            row for row in selected_rows
            if row["dimension"] == "N" and float(row["value"]) == 400
        )
        quad_reference = next(
            (row for row in selected_rows if row["dimension"] == "radial_strategy"), None
        )
        if quad_reference is None or _relative(
            float(quad_reference["kernel_abs"]), float(gauss_reference["kernel_abs"])
        ) >= 1.0e-4:
            failures.append(f"{group} radial integration strategies")
    return not failures, failures, selections


def _recheck_frozen_configuration(
    reference: RelativisticTideSettings,
    pilots: tuple[tuple[float, State, State], ...],
    convergence_rows: list[dict[str, Any]],
    selections: dict[str, Any],
    cache: ResumeCache,
) -> tuple[bool, list[str], dict[str, Any], list[dict[str, Any]]]:
    """Certify the combined low-cost configuration, promoting as needed."""
    high = {
        "N": 400,
        "radial_nodes": 80,
        "angular_nodes": 96,
        "horizon_cutoff": 3.0e-5,
        "counterterm_order": 4,
    }
    added: list[dict[str, Any]] = []
    failures: list[str] = []
    while True:
        frozen = _settings(
            truncation=int(selections["N"]),
            radial_nodes=int(selections["radial_nodes"]),
            angular_nodes=int(selections["angular_nodes"]),
            cutoff=float(selections["horizon_cutoff"]),
            counterterm_order=int(selections["counterterm_order"]),
        )
        settings_hash = _canonical_hash(_settings_payload(frozen))[:16]
        iteration: list[dict[str, Any]] = []
        maximum_difference = 0.0
        modes_passed = True
        for alpha, initial, final in pilots:
            frozen_result = cache.get_or_run(
                f"frozen_{settings_hash}_{alpha:.6f}_{initial.label}_{final.label}",
                lambda alpha=alpha, initial=initial, final=final, frozen=frozen:
                _evaluate_pilot(alpha, initial, final, frozen, "gauss"),
            )
            reference_result = next(
                row for row in convergence_rows
                if row["alpha"] == alpha
                and row["initial"] == initial.label
                and row["final"] == final.label
                and row["dimension"] == "N"
                and float(row["value"]) == 400.0
            )
            difference = _relative(
                float(frozen_result["kernel_abs"]),
                float(reference_result["kernel_abs"]),
            )
            maximum_difference = max(maximum_difference, difference)
            local_modes = all(
                float(frozen_result[name]) < threshold
                for name, threshold in (
                    ("cf_residual", 1.0e-10),
                    ("radial_residual", 1.0e-8),
                    ("angular_residual", 1.0e-8),
                    ("bilinear_norm_error", 1.0e-5),
                )
            )
            modes_passed = modes_passed and local_modes
            iteration.append(
                {
                    "dimension": "frozen_configuration",
                    "value": settings_hash,
                    "relative_to_reference": difference,
                    "modes_passed": local_modes,
                    **frozen_result,
                }
            )
        added.extend(iteration)
        if maximum_difference < 3.0e-4 and modes_passed:
            return True, failures, selections, added

        remaining = [name for name in high if selections[name] != high[name]]
        if not remaining:
            failures.append(
                f"frozen configuration failed: max kernel difference {maximum_difference:.3e}"
            )
            return False, failures, selections, added

        impacts: dict[str, float] = {}
        for dimension in remaining:
            preferred = selections[dimension]
            impact = 0.0
            groups = {
                (row["alpha"], row["initial"], row["final"])
                for row in convergence_rows
            }
            for group in groups:
                local = [
                    row for row in convergence_rows
                    if (row["alpha"], row["initial"], row["final"]) == group
                    and row["dimension"] == dimension
                ]
                low_row = next(
                    (row for row in local if float(row["value"]) == float(preferred)), None
                )
                high_row = next(
                    (row for row in local if float(row["value"]) == float(high[dimension])), None
                )
                if low_row is not None and high_row is not None:
                    impact = max(
                        impact,
                        _relative(float(low_row["kernel_abs"]), float(high_row["kernel_abs"])),
                    )
            impacts[dimension] = impact
        promote = max(remaining, key=lambda name: impacts[name])
        selections[promote] = high[promote]


def _frozen_config(selections: dict[str, Any]) -> dict[str, Any]:
    settings = _settings(
        truncation=int(selections["N"]),
        radial_nodes=int(selections["radial_nodes"]),
        angular_nodes=int(selections["angular_nodes"]),
        cutoff=float(selections["horizon_cutoff"]),
        counterterm_order=int(selections["counterterm_order"]),
    )
    return {
        "schema_version": 1,
        "alphas": [float(value) for value in np.linspace(0.10, 0.45, 15)],
        "q_values": [float(value) for value in np.geomspace(1.0e-4, 1.0, 33)],
        "channels": [[initial.label, final.label] for initial, final in TRANSITION_CHANNELS],
        "settings": _settings_payload(settings),
    }


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    tracemalloc.start()
    output = DEFAULT_OUTPUT / "smoke" if args.quick else DEFAULT_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "certification_cache.json"
    source_files = list((ROOT / "src" / "boson_ep").glob("*.py"))
    source_files += [Path(__file__), ROOT / "scripts" / "scan_relativistic_atlas.py", REFERENCE_PATH]
    code_hash = _sha256_files(source_files)
    if args.overwrite and cache_path.exists():
        cache_path.unlink()
    cache = ResumeCache(
        cache_path, args.resume and not args.overwrite, code_hash
    )

    if args.quick:
        reference = _settings(
            truncation=150, radial_nodes=12, angular_nodes=6,
            cutoff=3.0e-5, counterterm_order=4, quick=True,
            mode_angular_nodes=30,
        )
        weak_alphas = (0.10,)
        benchmark_alphas = (0.10,)
        pilots = PILOTS[:1]
    else:
        reference = _settings(
            truncation=400, radial_nodes=80, angular_nodes=96,
            cutoff=3.0e-5, counterterm_order=4
        )
        weak_alphas = (0.030, 0.040, 0.050, 0.060, 0.075, 0.100)
        benchmark_alphas = (0.05, 0.10, 0.15, 0.20)
        pilots = PILOTS

    failures: list[str] = []
    try:
        weak_rows = _weak_field_rows(reference, weak_alphas, cache)
        weak_fit = fit_weak_field_limit(weak_rows)
    except Exception as error:  # A certificate records failures; it never falls back.
        weak_rows = []
        weak_fit = fit_weak_field_limit([])
        failures.append(f"weak field exception: {error}")
    try:
        benchmark_rows = _benchmark_rows(reference, benchmark_alphas, cache)
        benchmark = compare_schwarzschild_benchmark(
            benchmark_rows, _read_csv(REFERENCE_PATH)
        )
    except Exception as error:
        benchmark_rows = []
        benchmark = compare_schwarzschild_benchmark([], _read_csv(REFERENCE_PATH))
        failures.append(f"benchmark exception: {error}")
    try:
        ward_summary, ward_rows = _ward_rows(reference, pilots, cache, args.quick)
    except Exception as error:
        ward_summary, ward_rows = [], []
        failures.append(f"Ward exception: {error}")
    try:
        operator_maximum, operator_rows = _operator_rows(reference, pilots, cache)
    except Exception as error:
        operator_maximum, operator_rows = math.inf, []
        failures.append(f"operator exception: {error}")
    try:
        convergence_rows = _convergence_rows(reference, pilots, cache, args.quick)
        convergence_passed, convergence_failures, selections = _assess_convergence(
            convergence_rows, args.quick
        )
        failures.extend(convergence_failures)
        if convergence_passed and not args.quick:
            (
                frozen_passed,
                frozen_failures,
                selections,
                frozen_rows,
            ) = _recheck_frozen_configuration(
                reference, pilots, convergence_rows, selections, cache
            )
            convergence_rows.extend(frozen_rows)
            convergence_passed = convergence_passed and frozen_passed
            failures.extend(frozen_failures)
    except Exception as error:
        convergence_rows = []
        convergence_passed = False
        selections = {}
        failures.append(f"convergence exception: {error}")

    if weak_fit.status != "ok":
        failures.append(f"weak-field fit: {weak_fit.status}")
    if benchmark.status != "ok":
        failures.append(f"Schwarzschild benchmark: {benchmark.status}")
    if len(ward_summary) != len(pilots) or any(row.get("status") != "ok" for row in ward_summary):
        failures.append("compact-support Ward audit")
    if operator_maximum >= 1.0e-10 or len(operator_rows) != 4 * len(pilots):
        failures.append("connection/divergence operator audit")
    if not convergence_passed:
        failures.append("pilot convergence matrix")
    if args.quick:
        failures.append("quick mode is smoke-only")

    frozen: dict[str, Any] = {}
    config_hash: str | None = None
    if selections and not args.quick:
        frozen = _frozen_config(selections)
        config_hash = _canonical_hash(frozen)
        frozen["config_sha256"] = config_hash
        _atomic_json(output / "frozen_atlas_config.json", frozen)

    _write_csv(output / "ward_audit.csv", ward_rows)
    _write_csv(output / "weak_field_fit.csv", [*weak_rows, {"row_type": "fit", **weak_fit.to_dict()}])
    _write_csv(output / "schwarzschild_benchmark.csv", list(benchmark.rows))
    _write_csv(output / "pilot_convergence.csv", convergence_rows)
    _write_csv(output / "operator_audit.csv", operator_rows)

    elapsed = time.perf_counter() - start
    cumulative_elapsed = cache.prior_wall_time_seconds + elapsed
    cache.rows["__cumulative_wall_time_seconds__"] = cumulative_elapsed
    _atomic_json(cache.path, cache.rows)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_rss = _peak_rss_bytes()
    status = "pre_atlas_certified" if not failures else (
        "smoke_only" if args.quick else "pre_atlas_failed"
    )
    certificate = {
        "status": status,
        "schema_version": 1,
        "code_sha256": code_hash,
        "config_sha256": config_hash,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sympy": sympy.__version__,
            "platform": platform.platform(),
        },
        "reference_settings": _settings_payload(reference),
        "thresholds": {
            "ward": 1.0e-8,
            "operator_forms": 1.0e-10,
            "cf_residual": 1.0e-10,
            "kernel_N": 3.0e-4,
            "kernel_radial_angular": 1.0e-4,
            "kernel_horizon_counterterm": 1.0e-5,
        },
        "checks": {
            "weak_field": weak_fit.to_dict(),
            "schwarzschild_benchmark": benchmark.to_dict(include_rows=False),
            "ward": ward_summary,
            "operator_form_maximum": operator_maximum,
            "convergence_passed": convergence_passed,
        },
        "failures": sorted(set(failures)),
        "wall_time_seconds": cumulative_elapsed,
        "last_resume_wall_time_seconds": elapsed,
        "peak_python_heap_bytes": peak,
        "peak_resident_memory_bytes": peak_rss,
        "peak_memory_limit_bytes": 4 * 1024**3,
    }
    if peak_rss is None or peak_rss >= 4 * 1024**3:
        reason = (
            "peak resident memory unavailable"
            if peak_rss is None
            else "peak memory exceeded 4 GiB"
        )
        failures.append(reason)
        certificate["failures"] = sorted(set(failures))
        certificate["status"] = "pre_atlas_failed"
    _atomic_json(output / "pre_atlas_certificate.json", certificate)

    lines = [
        "# 完整图谱启动前认证",
        "",
        f"- 状态：`{certificate['status']}`",
        f"- 代码 SHA-256：`{code_hash}`",
        f"- 配置 SHA-256：`{config_hash or 'not_frozen'}`",
        f"- 累计墙钟时间：`{cumulative_elapsed:.1f} s`",
        f"- 本次续跑时间：`{elapsed:.1f} s`",
        f"- Python 堆内存峰值：`{peak / 1024**2:.1f} MiB`",
        f"- OS 常驻内存峰值：`{(peak_rss or 0) / 1024**2:.1f} MiB`",
        "",
        "## 验收结果",
        "",
        f"- Weak-field fit: `{weak_fit.status}`",
        f"- Schwarzschild benchmark: `{benchmark.status}`",
        f"- Ward pilots: `{sum(row.get('status') == 'ok' for row in ward_summary)}/{len(pilots)}`",
        f"- Operator-form maximum residual: `{operator_maximum:.6e}`",
        f"- Pilot convergence: `{'passed' if convergence_passed else 'failed'}`",
        "",
        "## 失败项",
        "",
    ]
    lines.extend(f"- {failure}" for failure in sorted(set(failures)))
    if not failures:
        lines.extend(["- 无。认证通过，可立即启动正式图谱。"])
    temporary = output / "pre_atlas_certification.md.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(output / "pre_atlas_certification.md")
    print(json.dumps({"status": certificate["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
