"""Certify the repaired Kerr tidal-atlas implementation before a v2 scan.

The gate is deliberately stricter than the atlas driver: it traces every
saturation branch in both directions, resolves every unique mode used by the
atlas, and runs direct production Ward audits on three representative
transitions.  A formal scan may consume the emitted configuration only when
the certificate says ``pre_atlas_certified`` and all fingerprints agree.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from boson_ep.models import (
    GaugeVectorSpec,
    KerrModeResult,
    KerrModeSettings,
    RelativisticTideSettings,
    SaturationResult,
    State,
    TransitionConfig,
)
from boson_ep.relativistic_tides import TRANSITION_CHANNELS


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "relativistic_tides" / "v2" / "certification"
REFERENCE_PATH = ROOT / "results/relativistic_tides/benchmarks/schwarzschild_kernel_reference.csv"
SCANNER_PATH = ROOT / "scripts" / "scan_relativistic_atlas_v2.py"
SCHEMA_VERSION = 2
ALPHAS = tuple(float(value) for value in np.linspace(0.10, 0.45, 15))
Q_VALUES = tuple(float(value) for value in np.geomspace(1.0e-4, 1.0, 33))
INITIAL_STATES = (State(2, 1, 1), State(3, 2, 2), State(4, 3, 3))
BRANCH_STATES = {
    State(2, 1, 1): (State(2, 1, 1), State(2, 1, -1)),
    State(3, 2, 2): (State(3, 2, 2), State(3, 2, 0), State(3, 0, 0)),
    State(4, 3, 3): (State(4, 3, 3), State(4, 3, 1), State(4, 1, 1)),
}
PILOTS = (
    (0.10, State(2, 1, 1), State(2, 1, -1)),
    (0.25, State(3, 2, 2), State(3, 0, 0)),
    (0.45, State(4, 3, 3), State(4, 1, 1)),
)
EXPECTED_COUNTS = {
    "saturation": 45,
    "unique_modes": 120,
    "ward_pilots": 3,
    "operator_pilots": 3,
    "weak_field": 6,
    "schwarzschild_benchmark": 4,
}

WEAK_ALPHAS = (0.030, 0.040, 0.050, 0.060, 0.075, 0.100)
BENCHMARK_ALPHAS = (0.05, 0.10, 0.15, 0.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--quick", action="store_true", help="one-point smoke test; never certifies"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def _finite_json(value: Any) -> Any:
    """Return JSON-safe data, representing failed non-finite values as null."""
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, complex):
        return {"real": _finite_json(value.real), "imag": _finite_json(value.imag)}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        _finite_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _finite_json(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    flattened: list[dict[str, Any]] = []
    for source in rows:
        row: dict[str, Any] = {}
        for key, value in _finite_json(source).items():
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, sort_keys=True, allow_nan=False)
            else:
                row[key] = value
            if key not in keys:
                keys.append(key)
        flattened.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        if keys:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(flattened)
    temporary.replace(path)


def _sha256_files(paths: Sequence[Path], root: Path = ROOT) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(f"certification source is missing: {path}")
        try:
            label = path.relative_to(root).as_posix()
        except ValueError:
            label = path.resolve().as_posix()
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _source_paths() -> list[Path]:
    return [
        *sorted((ROOT / "src" / "boson_ep").glob("*.py")),
        Path(__file__).resolve(),
        SCANNER_PATH,
        REFERENCE_PATH,
    ]


def _dependency_record() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "sympy": importlib.metadata.version("sympy"),
    }


def _settings() -> RelativisticTideSettings:
    mode = KerrModeSettings(
        truncation=400,
        series_terms=400,
        angular_lmax=14,
        angular_nodes=72,
        radial_rtol=1.0e-8,
        horizon_cutoff=1.0e-4,
        counterterm_order=3,
        outer_decay_lengths=28.0,
        adaptive_truncations=(400, 600, 800, 1200, 1600),
    )
    return RelativisticTideSettings(
        mode=mode,
        radial_rtol=3.0e-6,
        radial_atol=1.0e-10,
        radial_nodes=80,
        angular_nodes=72,
        horizon_cutoffs=(1.0e-4,),
    )


def _settings_payload(settings: RelativisticTideSettings) -> dict[str, Any]:
    return _finite_json(asdict(settings))


def _frozen_configuration(settings: RelativisticTideSettings) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "alphas": list(ALPHAS),
        "q_values": list(Q_VALUES),
        "channels": [
            [initial.label, final.label]
            for initial, final in TRANSITION_CHANNELS
        ],
        "settings": _settings_payload(settings),
        "expected_counts": EXPECTED_COUNTS,
        "thresholds": {
            "cf_residual": 1.0e-10,
            "saturation_residual": 1.0e-12,
            "branch_chi_agreement": 1.0e-10,
            "radial_residual": 1.0e-8,
            "radial_ode_residual": 1.0e-6,
            "angular_residual": 1.0e-8,
            "bilinear_norm_error": 1.0e-5,
            "maximum_selected_truncation": 1200,
            "ward": 1.0e-8,
            "operator_forms": 1.0e-10,
            "peak_memory_bytes": 4 * 1024**3,
        },
    }
    payload["config_sha256"] = _canonical_hash(payload)
    return payload


def _run_fingerprint(code_hash: str, config_hash: str, dependency_hash: str) -> str:
    return _canonical_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "code_sha256": code_hash,
            "config_sha256": config_hash,
            "dependency_sha256": dependency_hash,
        }
    )


class CertificationCache:
    """Fingerprint-bound, payload-authenticated, atomic stage cache."""

    def __init__(
        self,
        root: Path,
        *,
        enabled: bool,
        code_sha256: str,
        config_sha256: str,
        dependency_sha256: str,
        run_fingerprint: str,
    ) -> None:
        self.root = root
        self.enabled = enabled
        self.identity = {
            "schema_version": SCHEMA_VERSION,
            "code_sha256": code_sha256,
            "config_sha256": config_sha256,
            "dependency_sha256": dependency_sha256,
            "run_fingerprint": run_fingerprint,
        }

    def _path(self, stage: str, point_id: str) -> Path:
        key = hashlib.sha256(point_id.encode("utf-8")).hexdigest()[:20]
        return self.root / stage / f"{key}.json"

    def load(self, stage: str, point_id: str) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(stage, point_id)
        if not path.is_file():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if any(envelope.get(key) != value for key, value in self.identity.items()):
            return None
        if envelope.get("point_id") != point_id or envelope.get("stage") != stage:
            return None
        if envelope.get("stage_status") != "complete":
            return None
        payload = envelope.get("payload")
        if envelope.get("payload_sha256") != _canonical_hash(payload):
            return None
        return payload

    def store(self, stage: str, point_id: str, payload: Any) -> Any:
        safe = _finite_json(payload)
        envelope = {
            **self.identity,
            "point_id": point_id,
            "stage": stage,
            "stage_status": "complete",
            "payload_sha256": _canonical_hash(safe),
            "payload": safe,
        }
        _atomic_json(self._path(stage, point_id), envelope)
        return safe

    def get_or_run(
        self, stage: str, point_id: str, function: Callable[[], Any]
    ) -> Any:
        cached = self.load(stage, point_id)
        if cached is not None:
            return cached
        return self.store(stage, point_id, function())


def _state_from_label(label: str) -> State:
    if len(label) < 4 or label[2] not in "+-":
        raise ValueError(f"invalid state label: {label}")
    return State(int(label[0]), int(label[1]), int(label[2:]))


def _saturation_to_payload(result: SaturationResult) -> dict[str, Any]:
    return result.to_dict()


def _saturation_from_payload(payload: Mapping[str, Any]) -> SaturationResult:
    return SaturationResult(
        alpha=float(payload["alpha"]),
        state=_state_from_label(str(payload["state"])),
        chi=float(payload["chi"]),
        frequency_M=complex(
            float(payload["omega_real_M"]), float(payload["gamma_M"])
        ),
        cf_residual=float(payload["cf_residual"]),
        saturation_residual=float(payload["saturation_residual"]),
        truncation=int(payload["truncation"]),
        converged=bool(payload["converged"]),
        message=str(payload.get("message", "")),
    )


def _mode_to_payload(result: KerrModeResult) -> dict[str, Any]:
    return result.to_dict(include_coefficients=True)


def _mode_from_payload(payload: Mapping[str, Any]) -> KerrModeResult:
    def number(key: str, default: float = math.nan) -> float:
        value = payload.get(key)
        return default if value is None else float(value)

    angular_coefficients = np.asarray(
        [complex(*value) for value in payload["angular_coefficients"]], complex
    )
    radial_coefficients = np.asarray(
        [complex(*value) for value in payload["radial_coefficients"]], complex
    )
    history = tuple(payload.get("convergence_history", ()))
    return KerrModeResult(
        alpha=float(payload["alpha"]),
        chi=float(payload["chi"]),
        state=_state_from_label(str(payload["state"])),
        frequency_M=complex(
            number("omega_real_M"), number("gamma_M")
        ),
        separation_constant=complex(
            number("separation_real"), number("separation_imag")
        ),
        angular_l_values=np.asarray(payload["angular_l_values"], int),
        angular_coefficients=angular_coefficients,
        radial_coefficients=radial_coefficients,
        bilinear_norm=complex(
            number("bilinear_norm_real"),
            number("bilinear_norm_imag"),
        ),
        r_99_M=number("r_99_M"),
        cf_residual=number("cf_residual", math.inf),
        radial_residual=number("radial_residual", math.inf),
        angular_residual=number("angular_residual", math.inf),
        converged=bool(payload["converged"]),
        message=str(payload.get("message", "")),
        selected_truncation=(
            None
            if payload.get("selected_truncation") is None
            else int(payload["selected_truncation"])
        ),
        convergence_history=history,
        radial_ode_residual=(
            None
            if payload.get("radial_ode_residual") is None
            else float(payload["radial_ode_residual"])
        ),
    )


def _runtime_apis():
    """Import v2 APIs late so helper tests remain useful during development."""
    from boson_ep.certification import direct_ward_audit, operator_form_audit
    from boson_ep.relativity import (
        solve_kerr_mode_adaptive,
        trace_saturation_branch_cf,
    )

    return (
        trace_saturation_branch_cf,
        solve_kerr_mode_adaptive,
        direct_ward_audit,
        operator_form_audit,
    )


def _mode_point_id(alpha: float, branch: State, state: State) -> str:
    return f"a{alpha:.6f}_{branch.label}_{state.label}"


def _mode_quality_failures(mode: KerrModeResult, point_id: str) -> list[str]:
    failures: list[str] = []
    checks = (
        (mode.cf_residual, 1.0e-10, "cf_residual"),
        (mode.radial_residual, 1.0e-8, "radial_residual"),
        (mode.angular_residual, 1.0e-8, "angular_residual"),
        (
            math.inf if mode.radial_ode_residual is None else mode.radial_ode_residual,
            1.0e-6,
            "radial_ode_residual",
        ),
        (abs(mode.bilinear_norm - 1.0), 1.0e-5, "bilinear_norm"),
    )
    if not mode.converged:
        failures.append(f"mode {point_id}: not converged")
    for value, threshold, name in checks:
        if not math.isfinite(float(value)) or value >= threshold:
            failures.append(f"mode {point_id}: {name}={value!r}")
    if mode.selected_truncation is None or mode.selected_truncation > 1200:
        failures.append(
            f"mode {point_id}: selected truncation={mode.selected_truncation!r}"
        )
    return failures


def _ward_specs(
    chi: float, r_99_M: float, *, quick: bool = False
) -> tuple[GaugeVectorSpec, ...]:
    r_plus = 1.0 + math.sqrt(1.0 - chi * chi)
    supports = (
        ("near_horizon", r_plus + 2.0, r_plus + 10.0),
        ("cloud_core", max(r_plus + 1.0e-3, 0.2 * r_99_M), 0.7 * r_99_M),
    )
    if any(not right > left for _, left, right in supports):
        raise ValueError("invalid compact Ward support for resolved pilot modes")
    specs = tuple(
        GaugeVectorSpec(kind, left, right, amplitude, support_name=name)
        for kind in ("temporal", "radial", "polar", "axial")
        for name, left, right in supports
        for amplitude in (0.1, 1.0)
    )
    if quick:
        return tuple(
            spec for spec in specs
            if spec.kind == "temporal" and spec.support_name == "cloud_core"
        )
    return specs


def _peak_rss_bytes() -> int | None:
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
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
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = ctypes.c_void_p
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(Counters),
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

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, OSError):
        return None


def _count_record(
    expected: int, completed: int, passed: int
) -> dict[str, int]:
    return {"expected": expected, "completed": completed, "passed": passed}


def _below(row: Mapping[str, Any], key: str, threshold: float) -> bool:
    value = row.get(key)
    return value is not None and math.isfinite(float(value)) and float(value) < threshold


def _read_reference() -> list[dict[str, Any]]:
    with REFERENCE_PATH.open(newline="", encoding="utf-8") as handle:
        output: list[dict[str, Any]] = []
        for row in csv.DictReader(handle):
            converted: dict[str, Any] = {}
            for key, value in row.items():
                if value == "":
                    continue
                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value
            output.append(converted)
        return output


def _weak_field_rows(
    settings: RelativisticTideSettings,
    alphas: Sequence[float],
    cache: CertificationCache,
) -> list[dict[str, Any]]:
    from boson_ep.relativistic_tides import (
        hydrogenic_newtonian_kernel_M,
        schwarzschild_irg_tidal_kernel_M,
        schwarzschild_rw_tidal_kernel_M,
    )

    initial, final = State(2, 1, 1), State(2, 1, -1)
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        point_id = f"weak_a{alpha:.6f}"

        def calculate(alpha=alpha):
            hydrogenic = hydrogenic_newtonian_kernel_M(alpha, initial, final)
            irg = schwarzschild_irg_tidal_kernel_M(alpha, initial, final, settings)
            rw = schwarzschild_rw_tidal_kernel_M(alpha, initial, final, settings)
            return {
                "alpha": alpha,
                "hydrogenic_kernel_abs": abs(hydrogenic),
                "irg_kernel_abs": abs(irg),
                "rw_kernel_abs": abs(rw),
                "kernel_ratio": abs(irg) / max(abs(hydrogenic), 1.0e-300),
                "gauge_residual": abs(irg - rw) / max(abs(rw), 1.0e-300),
            }

        rows.append(cache.get_or_run("weak_field", point_id, calculate))
    return rows


def _benchmark_rows(
    settings: RelativisticTideSettings,
    alphas: Sequence[float],
    cache: CertificationCache,
) -> list[dict[str, Any]]:
    from boson_ep.relativistic_tides import (
        hydrogenic_newtonian_kernel_M,
        schwarzschild_irg_tidal_kernel_M,
        schwarzschild_newtonian_tidal_kernel_M,
        schwarzschild_rw_tidal_kernel_M,
    )

    initial, final = State(2, 1, 1), State(3, 1, -1)
    gauge_final = State(2, 1, -1)
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        point_id = f"benchmark_a{alpha:.6f}"

        def calculate(alpha=alpha):
            hydrogenic = hydrogenic_newtonian_kernel_M(alpha, initial, final)
            semirelativistic = schwarzschild_newtonian_tidal_kernel_M(
                alpha, initial, final, settings
            )
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

        rows.append(cache.get_or_run("schwarzschild_benchmark", point_id, calculate))
    return rows


def _direction_difference(result: SaturationResult) -> float:
    marker = "direction_difference="
    if marker not in result.message:
        raise ValueError("saturation trace did not report its direction cross-check")
    encoded = result.message.split(marker, 1)[1].split(";", 1)[0].strip()
    return float(encoded)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = _settings()
    frozen = _frozen_configuration(settings)
    config_hash = str(frozen["config_sha256"])
    failures: list[str] = []

    try:
        code_hash = _sha256_files(_source_paths())
    except Exception as error:
        code_hash = "source_fingerprint_failed"
        failures.append(f"source fingerprint: {error}")
    dependencies = _dependency_record()
    dependency_hash = _canonical_hash(dependencies)
    run_fingerprint = _run_fingerprint(code_hash, config_hash, dependency_hash)
    if args.overwrite:
        cache_root = output / "cache"
        if cache_root.exists():
            # Only this run's small certification cache is removed.  v1 and
            # formal atlas data are outside this directory and remain intact.
            import shutil
            resolved_cache = cache_root.resolve()
            if output != resolved_cache.parent:
                raise RuntimeError("refusing to remove cache outside certification output")
            shutil.rmtree(resolved_cache)
    cache = CertificationCache(
        output / "cache",
        enabled=args.resume and not args.overwrite,
        code_sha256=code_hash,
        config_sha256=config_hash,
        dependency_sha256=dependency_hash,
        run_fingerprint=run_fingerprint,
    )

    trace_branch, solve_mode, ward_audit, operator_audit = _runtime_apis()
    alphas = (ALPHAS[0],) if args.quick else ALPHAS
    initial_states = INITIAL_STATES[:1] if args.quick else INITIAL_STATES
    pilots = PILOTS[:1] if args.quick else PILOTS
    saturation_rows: list[dict[str, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    ward_rows: list[dict[str, Any]] = []
    ward_detail_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    saturation_by_branch: dict[tuple[float, State], SaturationResult] = {}
    modes: dict[tuple[float, State, State], KerrModeResult] = {}

    from boson_ep.certification import (
        compare_schwarzschild_benchmark,
        fit_weak_field_limit,
    )

    weak_alphas = (0.10,) if args.quick else WEAK_ALPHAS
    benchmark_alphas = (0.10,) if args.quick else BENCHMARK_ALPHAS
    reference_settings = replace(
        settings,
        mode=replace(
            settings.mode,
            horizon_cutoff=3.0e-5,
            counterterm_order=4,
        ),
        radial_nodes=80,
        angular_nodes=96,
        horizon_cutoffs=(3.0e-5,),
    )
    try:
        weak_rows = _weak_field_rows(reference_settings, weak_alphas, cache)
        weak_fit = fit_weak_field_limit(weak_rows)
        if not args.quick and weak_fit.status != "ok":
            failures.append(f"weak-field fit: {weak_fit.status}")
    except Exception as error:
        weak_rows = []
        weak_fit = fit_weak_field_limit([])
        failures.append(f"weak-field exception: {error}")
    try:
        benchmark_rows = _benchmark_rows(reference_settings, benchmark_alphas, cache)
        reference_rows = [
            row for row in _read_reference()
            if any(abs(row["alpha"] - alpha) < 1.0e-12 for alpha in benchmark_alphas)
        ]
        benchmark = compare_schwarzschild_benchmark(
            benchmark_rows, reference_rows
        )
        if benchmark.status != "ok":
            failures.append(f"Schwarzschild benchmark: {benchmark.status}")
    except Exception as error:
        benchmark_rows = []
        benchmark = compare_schwarzschild_benchmark([], [])
        failures.append(f"Schwarzschild benchmark exception: {error}")

    for initial in initial_states:
        results: list[SaturationResult] = []
        point_id = f"{initial.label}_bidirectional"
        try:
            payload = cache.get_or_run(
                "saturation",
                point_id,
                lambda initial=initial: [
                    _saturation_to_payload(item)
                    for item in trace_branch(alphas, initial, settings.mode)
                ],
            )
            results = [_saturation_from_payload(item) for item in payload]
        except Exception as error:
            failures.append(f"saturation {point_id}: {error}")
        if len(results) != len(alphas):
            failures.append(f"saturation {initial.label}: incomplete bidirectional trace")
        for result in results:
            try:
                agreement = _direction_difference(result)
            except ValueError as error:
                agreement = math.inf
                failures.append(f"saturation a{result.alpha:.6f}_{initial.label}: {error}")
            row = {
                **result.to_dict(),
                "branch_chi_agreement": agreement,
            }
            saturation_rows.append(row)
            saturation_by_branch[(round(result.alpha, 12), initial)] = result
            if (
                not result.converged
                or result.cf_residual >= 1.0e-10
                or result.saturation_residual >= 1.0e-12
                or agreement >= 1.0e-10
            ):
                failures.append(
                    f"saturation a{result.alpha:.6f}_{initial.label}: acceptance"
                )

    for initial in initial_states:
        previous: dict[State, complex] = {}
        previous_alpha: dict[State, float] = {}
        for alpha in alphas:
            saturation = saturation_by_branch.get((round(alpha, 12), initial))
            if saturation is None:
                continue
            for state in BRANCH_STATES[initial]:
                point_id = _mode_point_id(alpha, initial, state)
                try:
                    seed = previous.get(state)
                    if seed is not None:
                        prior_alpha = previous_alpha[state]
                        current_hydrogenic = alpha * (
                            1.0 - alpha**2 / (2.0 * state.n**2)
                        )
                        prior_hydrogenic = prior_alpha * (
                            1.0 - prior_alpha**2 / (2.0 * state.n**2)
                        )
                        seed += current_hydrogenic - prior_hydrogenic
                    else:
                        # The saturation solve has already isolated the
                        # correct radial overtone at this alpha.  It is a far
                        # better seed for the whole nearly degenerate
                        # multiplet than a fresh unscaled hydrogenic guess.
                        seed = saturation.frequency_M
                    payload = cache.load("mode", point_id)
                    if payload is None:
                        result = solve_mode(
                            alpha,
                            saturation.chi,
                            state,
                            settings.mode,
                            seed=seed,
                        )
                        payload = cache.store("mode", point_id, _mode_to_payload(result))
                    result = _mode_from_payload(payload)
                    if result.converged:
                        previous[state] = result.frequency_M
                        previous_alpha[state] = alpha
                    else:
                        previous.pop(state, None)
                        previous_alpha.pop(state, None)
                    modes[(round(alpha, 12), initial, state)] = result
                    row = {"point_id": point_id, "branch": initial.label, **result.to_dict()}
                    row["bilinear_norm_error"] = abs(result.bilinear_norm - 1.0)
                    mode_rows.append(row)
                    failures.extend(_mode_quality_failures(result, point_id))
                except Exception as error:
                    failures.append(f"mode {point_id}: {error}")

    # Explicit regressions for the three v1 truncation failures.
    regression_caps = (
        (0.10, State(3, 2, 2), State(3, 0, 0), 1200),
        (0.125, State(3, 2, 2), State(3, 0, 0), 1200),
        (0.45, State(2, 1, 1), State(2, 1, -1), 800),
    )
    if not args.quick:
        for alpha, initial, state, cap in regression_caps:
            mode = modes.get((round(alpha, 12), initial, state))
            if mode is None or mode.selected_truncation is None or mode.selected_truncation > cap:
                failures.append(
                    f"adaptive truncation regression a{alpha:.6f}_{initial.label}_{state.label}"
                )

    for alpha, initial, final in pilots:
        key_i = (round(alpha, 12), initial, initial)
        key_f = (round(alpha, 12), initial, final)
        if key_i not in modes or key_f not in modes:
            failures.append(f"pilot a{alpha:.6f}_{initial.label}_{final.label}: missing modes")
            continue
        pair = (modes[key_i], modes[key_f])
        config = TransitionConfig(
            alpha=alpha,
            initial=initial,
            final=final,
            chi=pair[0].chi,
            settings=settings,
        )
        pilot_id = f"a{alpha:.6f}_{initial.label}_{final.label}"
        try:
            specs = _ward_specs(
                pair[0].chi,
                max(pair[0].r_99_M, pair[1].r_99_M),
                quick=args.quick,
            )
            ward_payload = cache.get_or_run(
                "ward",
                pilot_id,
                lambda config=config, pair=pair, specs=specs: ward_audit(
                    config, pair, specs, include_off_resonance=True
                ).to_dict(include_rows=True),
            )
            details = ward_payload.pop("rows")
            ward_rows.append({"point_id": pilot_id, **ward_payload})
            ward_detail_rows.extend(
                {"point_id": pilot_id, **row} for row in details
            )
            if (
                ward_payload.get("status") != "ok"
                or not _below(ward_payload, "maximum_invariance_residual", 1.0e-8)
                or not _below(ward_payload, "maximum_pure_gauge_residual", 1.0e-8)
                or not _below(ward_payload, "maximum_linearity_residual", 1.0e-8)
                or ward_payload.get("off_resonance_ratio") is None
                or float(ward_payload["off_resonance_ratio"]) <= 100.0
                or len(details) != (2 if args.quick else 16)
            ):
                failures.append(f"direct Ward pilot {pilot_id}: acceptance")
        except Exception as error:
            failures.append(f"direct Ward pilot {pilot_id}: {error}")
        try:
            operator_payload = cache.get_or_run(
                "operator",
                pilot_id,
                lambda config=config, pair=pair: (
                    lambda result: {"maximum": result[0], "rows": result[1]}
                )(operator_audit(config, resolved_modes=pair)),
            )
            maximum = operator_payload.get("maximum")
            rows = operator_payload.get("rows", [])
            operator_rows.extend(
                {"point_id": pilot_id, **row} for row in rows
            )
            if (
                maximum is None
                or not math.isfinite(float(maximum))
                or float(maximum) >= 1.0e-10
                or len(rows) != 12
            ):
                failures.append(f"operator pilot {pilot_id}: acceptance")
        except Exception as error:
            failures.append(f"operator pilot {pilot_id}: {error}")

    expected = (
        {
            "saturation": len(alphas) * len(initial_states),
            "unique_modes": len(alphas)
            * sum(len(BRANCH_STATES[state]) for state in initial_states),
            "ward_pilots": len(pilots),
            "operator_pilots": len(pilots),
            "weak_field": len(weak_alphas),
            "schwarzschild_benchmark": len(benchmark_alphas),
        }
        if args.quick
        else EXPECTED_COUNTS
    )
    counts = {
        "saturation": _count_record(
            expected["saturation"],
            len(saturation_rows),
            sum(
                bool(row.get("converged"))
                and _below(row, "cf_residual", 1.0e-10)
                and _below(row, "saturation_residual", 1.0e-12)
                and _below(row, "branch_chi_agreement", 1.0e-10)
                for row in saturation_rows
            ),
        ),
        "unique_modes": _count_record(
            expected["unique_modes"],
            len(mode_rows),
            sum(not _mode_quality_failures(mode, "count") for mode in modes.values()),
        ),
        "ward_pilots": _count_record(
            expected["ward_pilots"],
            len(ward_rows),
            sum(row.get("status") == "ok" for row in ward_rows),
        ),
        "operator_pilots": _count_record(
            expected["operator_pilots"],
            len({row["point_id"] for row in operator_rows}),
            sum(
                max(
                    float(row["residual"])
                    for row in operator_rows
                    if row["point_id"] == point_id
                )
                < 1.0e-10
                for point_id in {row["point_id"] for row in operator_rows}
            ),
        ),
        "weak_field": _count_record(
            expected["weak_field"],
            len(weak_rows),
            (
                len(weak_rows)
                if (weak_fit.status == "ok" or args.quick)
                and all(
                    _below(row, "gauge_residual", 1.0e-8)
                    and _below(row, "kernel_ratio", math.inf)
                    for row in weak_rows
                )
                else 0
            ),
        ),
        "schwarzschild_benchmark": _count_record(
            expected["schwarzschild_benchmark"],
            len(benchmark.rows),
            sum(bool(row.get("passed")) for row in benchmark.rows),
        ),
    }
    for name, record in counts.items():
        if record["completed"] != record["expected"] or record["passed"] != record["expected"]:
            failures.append(f"count {name}: {record}")
    if args.quick:
        failures.append("quick mode is smoke-only")
    peak_rss = _peak_rss_bytes()
    if peak_rss is None or peak_rss >= 4 * 1024**3:
        failures.append(
            "peak resident memory unavailable"
            if peak_rss is None
            else "peak resident memory exceeded 4 GiB"
        )

    failures = sorted(set(failures))
    elapsed = time.perf_counter() - started
    status = "pre_atlas_certified" if not failures else (
        "smoke_only" if args.quick else "pre_atlas_failed"
    )
    implementation_failures = 0 if status == "pre_atlas_certified" else len(failures)
    certified_data_sha256 = _canonical_hash({
        "saturation": saturation_rows,
        "modes": mode_rows,
        "ward": ward_rows,
        "ward_details": ward_detail_rows,
        "operator": operator_rows,
        "weak_field": weak_rows,
        "weak_fit": weak_fit.to_dict(),
        "schwarzschild_benchmark": benchmark.to_dict(),
    })
    certificate = {
        "status": status,
        "schema_version": SCHEMA_VERSION,
        "code_sha256": code_hash,
        "config_sha256": config_hash,
        "dependency_sha256": dependency_hash,
        "run_fingerprint": run_fingerprint,
        "implementation_failures": implementation_failures,
        "certified_data_sha256": certified_data_sha256,
        "failures": failures,
        "counts": counts,
        "checks": {
            "weak_field": weak_fit.to_dict(),
            "schwarzschild_benchmark": benchmark.to_dict(include_rows=False),
        },
        "environment": {**dependencies, "platform": platform.platform()},
        "wall_time_seconds": elapsed,
        "peak_resident_memory_bytes": peak_rss,
    }

    _atomic_json(output / "frozen_atlas_config.json", frozen)
    _atomic_json(output / "saturation_certification.json", saturation_rows)
    _atomic_json(output / "mode_certification.json", mode_rows)
    _atomic_json(output / "ward_pilots.json", ward_rows)
    _atomic_json(output / "operator_pilots.json", operator_rows)
    _atomic_json(output / "weak_field_fit.json", {
        "rows": weak_rows, "fit": weak_fit.to_dict()
    })
    _atomic_json(output / "schwarzschild_benchmark.json", benchmark.to_dict())
    _write_csv(output / "saturation_certification.csv", saturation_rows)
    _write_csv(output / "mode_certification.csv", mode_rows)
    _write_csv(output / "ward_pilots.csv", ward_detail_rows)
    _write_csv(output / "operator_pilots.csv", operator_rows)
    _write_csv(
        output / "weak_field_fit.csv",
        [*weak_rows, {"row_type": "fit", **weak_fit.to_dict()}],
    )
    _write_csv(output / "schwarzschild_benchmark.csv", list(benchmark.rows))
    _atomic_json(output / "pre_atlas_certificate.json", certificate)
    report = [
        "# Kerr 潮汐图谱 v2 启动认证",
        "",
        f"- 状态：`{status}`",
        f"- 实现失败：`{implementation_failures}`",
        f"- 运行指纹：`{run_fingerprint}`",
        f"- 代码 SHA-256：`{code_hash}`",
        f"- 配置 SHA-256：`{config_hash}`",
        f"- 墙钟时间：`{elapsed:.1f} s`",
        f"- 峰值常驻内存：`{(peak_rss or 0) / 1024**2:.1f} MiB`",
        "",
        "## 计数",
        "",
    ]
    report.extend(
        f"- {name}: `{record['passed']}/{record['expected']}`"
        for name, record in counts.items()
    )
    report.extend([
        "",
        f"- Weak-field fit: `{weak_fit.status}`",
        f"- Schwarzschild benchmark: `{benchmark.status}`",
    ])
    report.extend(["", "## 失败项", ""])
    report.extend(f"- {failure}" for failure in failures)
    if not failures:
        report.append("- 无。v2 正式图谱可以启动。")
    temporary = output / "pre_atlas_certification.md.tmp"
    temporary.write_text("\n".join(report) + "\n", encoding="utf-8")
    temporary.replace(output / "pre_atlas_certification.md")
    print(json.dumps({"status": status}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
