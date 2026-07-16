"""Small immutable data models for the baseline calculation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np


@dataclass(frozen=True, order=True)
class State:
    n: int
    l: int
    m: int

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be positive")
        if not 0 <= self.l < self.n:
            raise ValueError("state must satisfy 0 <= l < n")
        if abs(self.m) > self.l:
            raise ValueError("state must satisfy |m| <= l")

    @property
    def label(self) -> str:
        return f"{self.n}{self.l}{self.m:+d}"


PRIMARY_INITIAL = State(2, 1, 1)
PRIMARY_FINAL = State(2, 1, -1)


@dataclass(frozen=True)
class EPResult:
    alpha: float
    chi: float
    q: float | None
    status: str
    residual: float | None
    discriminant_normalized: float | None
    omega_res_M: float
    radius_M: float | None
    radius_over_cloud: float | None
    eta_M: float | None
    delta_gamma_M: float
    analytic_q: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimescaleResult:
    alpha: float
    chi: float
    q: float
    mass_msun: float
    omega_res_M: float
    delta_e_M: float
    gamma_grow_M: float
    gamma_absorb_M: float
    eta_M: float
    domega_M2: float
    landau_zener_z: float
    radius_M: float
    radius_over_cloud: float
    times_M: dict[str, float]
    times_seconds: dict[str, float]
    hierarchy: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hierarchy"] = list(self.hierarchy)
        return data


@dataclass(frozen=True)
class EvolutionConfig:
    alpha: float
    chi: float
    q: float
    edge_factor: float = 100.0
    rtol: float = 1.0e-9
    atol: float = 1.0e-12
    max_steps: int = 200_000
    chirp_order: str = "newtonian"
    coupling_scale: float = 1.0
    initial_admixture: float = 0.0
    initial_phase: float = 0.0
    spectrum_model: str = "hydrogenic_detweiler"
    omega_initial_M: complex | None = None
    omega_final_M: complex | None = None
    store_trajectory: bool = True
    crosscheck: bool = False
    direct_scaled_edge_limit: float = 250.0
    direct_z_limit: float = 100.0


@dataclass(frozen=True)
class EvolutionResult:
    alpha: float
    chi: float
    q: float
    status: str
    spectrum_model: str
    chirp_order: str
    omega_res_M: float
    detuning_edge_M: float
    scaled_edge: float
    eta_res_M: float
    delta_gamma_M: float
    sweep_rate_M2: float
    landau_zener_z: float
    survival_nh: float
    survival_hermitian: float
    survival_factorized: float
    survival_lz: float
    initial_population_exit: float
    final_population_exit: float
    norm_exit: float
    absorbed_asymptotic: float
    effect_abs: float
    probability_balance_residual: float
    solver_error: float | None
    accepted_steps: int
    rejected_steps: int
    x: np.ndarray
    omega_M: np.ndarray
    initial_population: np.ndarray
    final_population: np.ndarray
    norm: np.ndarray

    def to_dict(self, include_trajectory: bool = True) -> dict[str, Any]:
        data = asdict(self)
        for name in ("x", "omega_M", "initial_population", "final_population", "norm"):
            value = data[name]
            data[name] = value.tolist() if include_trajectory else []
        return data


@dataclass(frozen=True)
class CFSettings:
    truncation: int = 200
    angular_lmax: int = 12
    residual_tolerance: float = 1.0e-10
    max_function_evaluations: int = 2_000


@dataclass(frozen=True)
class CFResult:
    alpha: float
    chi: float
    state: State
    frequency_M: complex
    residual: float
    truncation: int
    angular_lmax: int
    converged: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "chi": self.chi,
            "state": self.state.label,
            "omega_real_M": self.frequency_M.real,
            "gamma_M": self.frequency_M.imag,
            "residual": self.residual,
            "truncation": self.truncation,
            "angular_lmax": self.angular_lmax,
            "converged": self.converged,
            "message": self.message,
        }


@dataclass(frozen=True)
class SaturationResult:
    alpha: float
    state: State
    chi: float
    frequency_M: complex
    cf_residual: float
    saturation_residual: float
    truncation: int
    converged: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "state": self.state.label,
            "chi": self.chi,
            "omega_real_M": self.frequency_M.real,
            "gamma_M": self.frequency_M.imag,
            "cf_residual": self.cf_residual,
            "saturation_residual": self.saturation_residual,
            "truncation": self.truncation,
            "converged": self.converged,
            "message": self.message,
        }


@dataclass(frozen=True)
class KerrModeSettings:
    """Numerical controls for a normalized Kerr scalar quasibound mode."""

    truncation: int = 250
    angular_lmax: int = 14
    series_terms: int = 250
    residual_tolerance: float = 1.0e-10
    angular_nodes: int = 96
    radial_rtol: float = 1.0e-8
    horizon_cutoff: float = 1.0e-5
    counterterm_order: int = 2
    outer_decay_lengths: float = 28.0
    adaptive_truncations: tuple[int, ...] = (400, 600, 800, 1200, 1600)

    def cf_settings(self) -> CFSettings:
        return CFSettings(
            truncation=self.truncation,
            angular_lmax=self.angular_lmax,
            residual_tolerance=self.residual_tolerance,
        )


@dataclass(frozen=True)
class KerrModeResult:
    alpha: float
    chi: float
    state: State
    frequency_M: complex
    separation_constant: complex
    angular_l_values: np.ndarray
    angular_coefficients: np.ndarray
    radial_coefficients: np.ndarray
    bilinear_norm: complex
    r_99_M: float
    cf_residual: float
    radial_residual: float
    angular_residual: float
    converged: bool
    message: str
    selected_truncation: int | None = None
    convergence_history: tuple[dict[str, Any], ...] = ()
    radial_ode_residual: float | None = None

    def to_dict(self, include_coefficients: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "alpha": self.alpha,
            "chi": self.chi,
            "state": self.state.label,
            "omega_real_M": self.frequency_M.real,
            "gamma_M": self.frequency_M.imag,
            "separation_real": self.separation_constant.real,
            "separation_imag": self.separation_constant.imag,
            "bilinear_norm_real": self.bilinear_norm.real,
            "bilinear_norm_imag": self.bilinear_norm.imag,
            "r_99_M": self.r_99_M,
            "cf_residual": self.cf_residual,
            "radial_residual": self.radial_residual,
            "angular_residual": self.angular_residual,
            "converged": self.converged,
            "message": self.message,
            "selected_truncation": self.selected_truncation,
            "convergence_history": list(self.convergence_history),
            "radial_ode_residual": self.radial_ode_residual,
        }
        if include_coefficients:
            data["angular_l_values"] = self.angular_l_values.tolist()
            data["angular_coefficients"] = [
                [value.real, value.imag] for value in self.angular_coefficients
            ]
            data["radial_coefficients"] = [
                [value.real, value.imag] for value in self.radial_coefficients
            ]
        return data


@dataclass(frozen=True)
class RelativisticTideSettings:
    mode: KerrModeSettings = field(default_factory=KerrModeSettings)
    radial_rtol: float = 3.0e-6
    radial_atol: float = 1.0e-10
    angular_nodes: int = 72
    radial_nodes: int = 56
    horizon_cutoffs: tuple[float, ...] = (3.0e-4, 1.0e-4, 3.0e-5)
    max_subdivisions: int = 250
    tidal_radius_limit: float = 0.10
    adiabatic_frequency_limit: float = 1.0e-2


@dataclass(frozen=True)
class TransitionConfig:
    alpha: float
    initial: State
    final: State
    q: float = 1.0e-3
    chi: float | None = None
    settings: RelativisticTideSettings = field(default_factory=RelativisticTideSettings)


@dataclass(frozen=True)
class TransitionResult:
    alpha: float
    chi: float
    q: float
    initial: State
    final: State
    status: str
    omega_res_M: float
    separation_M: float
    tidal_strength: float
    r_99_over_b: float
    hydrogenic_kernel_M: complex
    semirelativistic_kernel_M: complex
    covariant_kernel_M: complex
    eta_M: complex
    landau_zener_z: float
    depletion_probability: float
    numerical_error: float
    systematic_error: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "alpha": self.alpha,
            "chi": self.chi,
            "q": self.q,
            "initial": self.initial.label,
            "final": self.final.label,
            "status": self.status,
            "omega_res_M": self.omega_res_M,
            "separation_M": self.separation_M,
            "tidal_strength": self.tidal_strength,
            "r_99_over_b": self.r_99_over_b,
            "landau_zener_z": self.landau_zener_z,
            "depletion_probability": self.depletion_probability,
            "numerical_error": self.numerical_error,
            "systematic_error": self.systematic_error,
            "message": self.message,
        }
        for name in (
            "hydrogenic_kernel_M",
            "semirelativistic_kernel_M",
            "covariant_kernel_M",
            "eta_M",
        ):
            value = getattr(self, name)
            data[f"{name}_real"] = value.real
            data[f"{name}_imag"] = value.imag
            data[f"{name}_abs"] = abs(value)
        return data


@dataclass(frozen=True)
class TransitionKernelResult:
    point_id: str
    alpha: float
    chi: float
    initial: State
    final: State
    kernel_status: str
    failure_stage: str | None
    omega_res_M: float | None
    r_99_M: float | None
    hydrogenic_kernel_M: complex | None
    semirelativistic_kernel_M: complex | None
    covariant_kernel_M: complex | None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "point_id": self.point_id,
            "alpha": self.alpha,
            "chi": self.chi,
            "initial": self.initial.label,
            "final": self.final.label,
            "kernel_status": self.kernel_status,
            "failure_stage": self.failure_stage,
            "omega_res_M": self.omega_res_M,
            "r_99_M": self.r_99_M,
            "message": self.message,
        }
        for name in (
            "hydrogenic_kernel_M",
            "semirelativistic_kernel_M",
            "covariant_kernel_M",
        ):
            value = getattr(self, name)
            data[f"{name}_real"] = None if value is None else value.real
            data[f"{name}_imag"] = None if value is None else value.imag
            data[f"{name}_abs"] = None if value is None else abs(value)
        return data


@dataclass(frozen=True)
class PhenomenologyResult:
    point_id: str
    alpha: float
    chi: float
    initial: State
    final: State
    q: float
    status: str
    separation_M: float | None
    r_99_over_b: float | None
    tidal_valid: bool
    adiabatic_valid: bool
    publication_valid: bool
    eta_hydrogenic_M: complex | None
    eta_semirelativistic_M: complex | None
    eta_covariant_M: complex | None
    z_hydrogenic: float | None
    z_semirelativistic: float | None
    z_covariant: float | None
    depletion_hydrogenic: float | None
    depletion_semirelativistic: float | None
    depletion_covariant: float | None
    power_ratio_covariant_to_hydrogenic: float | None
    error_lower: float | None = None
    error_upper: float | None = None
    eta_covariant_abs_lower: float | None = None
    eta_covariant_abs_upper: float | None = None
    z_covariant_lower: float | None = None
    z_covariant_upper: float | None = None
    power_ratio_lower: float | None = None
    power_ratio_upper: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["initial"] = self.initial.label
        data["final"] = self.final.label
        for name in (
            "eta_hydrogenic_M",
            "eta_semirelativistic_M",
            "eta_covariant_M",
        ):
            value = data.pop(name)
            data[f"{name}_real"] = None if value is None else value.real
            data[f"{name}_imag"] = None if value is None else value.imag
            data[f"{name}_abs"] = None if value is None else abs(value)
        return data


@dataclass(frozen=True)
class KernelErrorBudgetResult:
    point_id: str
    status: str
    sources: tuple[dict[str, Any], ...]
    rss_relative: float | None
    worst_relative: float | None
    worst_source: str | None
    systematic_error_abs: float | None
    complete: bool

    def to_dict(self, include_sources: bool = True) -> dict[str, Any]:
        data = asdict(self)
        data["sources"] = list(self.sources) if include_sources else []
        return data


@dataclass(frozen=True)
class MultipoleTideSettings:
    """Controls for the exact Newtonian point-companion multipole sum."""

    ell_max: int = 10
    radial_nodes: int = 112
    angular_nodes: int = 128
    radial_rtol: float = 1.0e-8
    q_points: int = 257
    tidal_limits: tuple[float, ...] = (0.05, 0.07, 0.10)


@dataclass(frozen=True)
class MultipoleKernelResult:
    point_id: str
    ell: int
    harmonic_m: int
    source_coefficient: complex
    kernel_M: complex
    gauss_quad_relative_difference: float
    status: str
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "ell": self.ell,
            "harmonic_m": self.harmonic_m,
            "source_coefficient_real": self.source_coefficient.real,
            "source_coefficient_imag": self.source_coefficient.imag,
            "source_coefficient_abs": abs(self.source_coefficient),
            "kernel_M_real": self.kernel_M.real,
            "kernel_M_imag": self.kernel_M.imag,
            "kernel_M_abs": abs(self.kernel_M),
            "gauss_quad_relative_difference": self.gauss_quad_relative_difference,
            "status": self.status,
            "message": self.message,
        }


@dataclass(frozen=True)
class MultipoleEtaResult:
    point_id: str
    q: float
    orbital_model: str
    separation_M: float
    epsilon_2: float
    eta_quadrupole_M: complex
    eta_exact_M: complex
    eta_covariant_corrected_M: complex
    eta_l8_M: complex
    eta_l10_M: complex
    multipole_shift_abs: float
    multipole_tail_abs: float
    sigma_multipole_abs: float
    sigma_covariant_high_abs: float
    r_99_over_b: float
    valid_r99_005: bool
    valid_r99_007: bool
    valid_r99_010: bool
    chirp_M2: float
    depletion_hydrogenic: float
    depletion_exact: float
    depletion_covariant_corrected: float
    depletion_change: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for name in (
            "eta_quadrupole_M", "eta_exact_M", "eta_covariant_corrected_M",
            "eta_l8_M", "eta_l10_M"
        ):
            value = data.pop(name)
            data[f"{name}_real"] = value.real
            data[f"{name}_imag"] = value.imag
            data[f"{name}_abs"] = abs(value)
        return data


@dataclass(frozen=True)
class HighOrderTideBudgetResult:
    point_id: str
    status: str
    rows: tuple[MultipoleEtaResult, ...]
    maximum_multipole_relative_shift: float
    maximum_tail_relative: float
    robust_q_low: float | None
    robust_q_high: float | None
    message: str = ""

    def to_dict(self, include_rows: bool = True) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "status": self.status,
            "rows": [row.to_dict() for row in self.rows] if include_rows else [],
            "maximum_multipole_relative_shift": self.maximum_multipole_relative_shift,
            "maximum_tail_relative": self.maximum_tail_relative,
            "robust_q_low": self.robust_q_low,
            "robust_q_high": self.robust_q_high,
            "message": self.message,
        }


@dataclass(frozen=True)
class ContourKernelSettings:
    """Independent fixed-frequency ODE/contour validation controls."""

    angular_epsilon: float = 1.0e-4
    angular_nodes: int = 192
    frobenius_order: int = 12
    asymptotic_order: int = 12
    match_decay_lengths: tuple[float, ...] = (8.0, 12.0, 16.0)
    contour_angles: tuple[float, ...] = (
        math.pi / 6.0, math.pi / 4.0, math.pi / 3.0
    )
    horizon_offsets: tuple[float, ...] = (1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5)
    rtol: float = 1.0e-10
    atol: float = 1.0e-12


@dataclass(frozen=True)
class ContourModeResult:
    alpha: float
    chi: float
    state: State
    frequency_M: complex
    separation_constant: complex
    angular_residual: float
    shooting_wronskian_residual: float
    contour_norm_spread: float
    normalization: complex
    r_peak_M: float
    converged: bool
    status: str
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha, "chi": self.chi, "state": self.state.label,
            "omega_real_M": self.frequency_M.real,
            "gamma_M": self.frequency_M.imag,
            "separation_real": self.separation_constant.real,
            "separation_imag": self.separation_constant.imag,
            "angular_residual": self.angular_residual,
            "shooting_wronskian_residual": self.shooting_wronskian_residual,
            "contour_norm_spread": self.contour_norm_spread,
            "normalization_real": self.normalization.real,
            "normalization_imag": self.normalization.imag,
            "r_peak_M": self.r_peak_M,
            "converged": self.converged, "status": self.status,
            "message": self.message,
        }


@dataclass(frozen=True)
class IndependentKernelResult:
    point_id: str
    certified_kernel_M: complex | None
    independent_kernel_M: complex
    amplitude_relative_difference: float | None
    phase_difference: float | None
    contour_spread: float
    shooting_wronskian_residual: float
    angular_eigenvalue_relative_difference: float
    status: str
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        certified = self.certified_kernel_M
        return {
            "point_id": self.point_id,
            "certified_kernel_M_real": None if certified is None else certified.real,
            "certified_kernel_M_imag": None if certified is None else certified.imag,
            "certified_kernel_M_abs": None if certified is None else abs(certified),
            "independent_kernel_M_real": self.independent_kernel_M.real,
            "independent_kernel_M_imag": self.independent_kernel_M.imag,
            "independent_kernel_M_abs": abs(self.independent_kernel_M),
            "amplitude_relative_difference": self.amplitude_relative_difference,
            "phase_difference": self.phase_difference,
            "contour_spread": self.contour_spread,
            "shooting_wronskian_residual": self.shooting_wronskian_residual,
            "angular_eigenvalue_relative_difference": self.angular_eigenvalue_relative_difference,
            "status": self.status, "message": self.message,
        }


KernelCrosscheckResult = IndependentKernelResult


@dataclass(frozen=True)
class GaugeAuditResult:
    alpha: float
    chi: float
    initial: State
    final: State
    status: str
    irg_trace_residual: float
    irg_tetrad_residual: float
    operator_form_residual: float
    ward_identity_residual: float
    schwarzschild_gauge_residual: float | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["initial"] = self.initial.label
        data["final"] = self.final.label
        return data


@dataclass(frozen=True)
class GaugeVectorSpec:
    kind: str
    support_inner: float
    support_outer: float
    relative_amplitude: float
    absolute_amplitude: float = 1.0
    support_name: str = "custom"

    def __post_init__(self) -> None:
        if self.kind not in {"temporal", "radial", "polar", "axial"}:
            raise ValueError("unknown compact-support gauge-vector kind")
        if not self.support_outer > self.support_inner:
            raise ValueError("gauge support must have positive width")
        if self.relative_amplitude <= 0.0:
            raise ValueError("relative gauge amplitude must be positive")


@dataclass(frozen=True)
class GaugeMetricConfig:
    radius: float
    theta: float
    chi: float
    omega_orb_M: float


@dataclass(frozen=True)
class GaugeMetricResult:
    metric_coefficient: np.ndarray
    gauge_vector: np.ndarray
    bump_value: float
    active: bool


@dataclass(frozen=True)
class WardAuditResult:
    alpha: float
    chi: float
    initial: State
    final: State
    status: str
    physical_kernel_M: complex
    maximum_invariance_residual: float
    maximum_pure_gauge_residual: float
    maximum_linearity_residual: float
    off_resonance_ratio: float
    rows: tuple[dict[str, Any], ...]

    def to_dict(self, include_rows: bool = True) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "chi": self.chi,
            "initial": self.initial.label,
            "final": self.final.label,
            "status": self.status,
            "physical_kernel_real": self.physical_kernel_M.real,
            "physical_kernel_imag": self.physical_kernel_M.imag,
            "physical_kernel_abs": abs(self.physical_kernel_M),
            "maximum_invariance_residual": self.maximum_invariance_residual,
            "maximum_pure_gauge_residual": self.maximum_pure_gauge_residual,
            "maximum_linearity_residual": self.maximum_linearity_residual,
            "off_resonance_ratio": self.off_resonance_ratio,
            "rows": list(self.rows) if include_rows else [],
        }


@dataclass(frozen=True)
class WeakFieldFitResult:
    status: str
    c0: float
    c2: float
    c4: float
    maximum_fit_residual: float
    maximum_gauge_residual: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkResult:
    status: str
    maximum_absolute_difference: float
    rows: tuple[dict[str, Any], ...]

    def to_dict(self, include_rows: bool = True) -> dict[str, Any]:
        return {
            "status": self.status,
            "maximum_absolute_difference": self.maximum_absolute_difference,
            "rows": list(self.rows) if include_rows else [],
        }


@dataclass(frozen=True)
class AtlasConfig:
    alphas: tuple[float, ...] = tuple(np.linspace(0.10, 0.45, 15))
    q_values: tuple[float, ...] = tuple(np.geomspace(1.0e-4, 1.0, 33))
    settings: RelativisticTideSettings = field(default_factory=RelativisticTideSettings)


@dataclass(frozen=True)
class UncertaintyBudget:
    numerical: float
    spectrum: float
    tidal_matrix: float
    chirp: float
    two_level: float
    total: float
    worst_component: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FormationConfig:
    alpha: float
    chi: float
    q: float
    mass_msun: float
    birth_frequency_fraction: float = 0.5
    seed_occupancy: float = 1.0


@dataclass(frozen=True)
class FormationResult:
    alpha: float
    chi: float
    q: float
    mass_msun: float
    status: str
    saturation_spin: float
    cloud_mass_fraction: float
    log_required_occupancy: float
    pre_resonance_efolds: float
    birth_frequency_fraction: float
    latest_birth_frequency_fraction: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WidthScanConfig:
    alpha: float
    chi: float
    mass_msun: float = 10.0
    q_min: float = 1.0e-4
    q_max: float = 1.0
    q_points: int = 257
    effect_threshold: float = 0.10
    sigma_multiplier: float = 3.0
    edge_factor: float = 100.0
    include_systematics: bool = True
    rtol: float = 1.0e-5
    atol: float = 1.0e-8


@dataclass(frozen=True)
class EffectWidthResult:
    alpha: float
    chi: float
    mass_msun: float
    status: str
    q_ep: float | None
    q_low: float | None
    q_high: float | None
    formal_relative_width: float
    physical_relative_width: float
    effect_at_ep: float | None
    sigma_at_ep: float | None
    uncertainty: UncertaintyBudget | None
    formation_status: str | None
    q_grid: np.ndarray
    effect_grid: np.ndarray
    sigma_grid: np.ndarray
    survival_nh_grid: np.ndarray
    survival_factorized_grid: np.ndarray

    def to_dict(self, include_grid: bool = True) -> dict[str, Any]:
        data = asdict(self)
        for name in (
            "q_grid",
            "effect_grid",
            "sigma_grid",
            "survival_nh_grid",
            "survival_factorized_grid",
        ):
            value = data[name]
            data[name] = value.tolist() if include_grid else []
        return data
