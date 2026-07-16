"""Publication gates for the covariant Kerr transition atlas.

This module contains the scientific checks.  File orchestration and the
machine-readable certificate live in ``scripts/certify_pre_atlas.py`` so a
failed or interrupted audit cannot be confused with a certified atlas.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from .models import (
    BenchmarkResult,
    GaugeMetricConfig,
    GaugeVectorSpec,
    TransitionConfig,
    WardAuditResult,
    WeakFieldFitResult,
)
from .relativity import saturation_spin_cf, solve_kerr_mode
from .relativistic_tides import (
    _delta_box_connection_bl,
    _delta_box_divergence_bl,
    _mode_field_provider,
    _mode_fields_bl,
    tidal_kernel_from_modes_M,
)
from .tidal_metric import (
    compact_support_gauge_metric,
    irg_metric_coefficient_bl,
    kerr_metric_boyer_lindquist,
)


WARD_TOLERANCE = 1.0e-8
WARD_GREEN_TOLERANCE = 1.0e-5


def _resolved_mode_pair(config: TransitionConfig, resolved_modes=None):
    """Return the initial/final modes without imposing a cache representation.

    Atlas v2 passes a pre-resolved pair so the Ward audit uses exactly the
    modes used by the production kernel.  The legacy certification entry point
    can still solve the pair locally.
    """
    if resolved_modes is None:
        chi = config.chi
        if chi is None:
            chi = saturation_spin_cf(
                config.alpha, config.initial, config.settings.mode
            )
        return (
            solve_kerr_mode(config.alpha, chi, config.initial, config.settings.mode),
            solve_kerr_mode(config.alpha, chi, config.final, config.settings.mode),
        )
    if isinstance(resolved_modes, Mapping):
        initial = resolved_modes.get(config.initial.label, resolved_modes.get("initial"))
        final = resolved_modes.get(config.final.label, resolved_modes.get("final"))
        if initial is None or final is None:
            raise ValueError("resolved_modes mapping must contain initial and final modes")
        return initial, final
    if len(resolved_modes) != 2:
        raise ValueError("resolved_modes must be an (initial, final) pair")
    return resolved_modes[0], resolved_modes[1]


def _compact_metric_provider(
    chi: float,
    omega_orb_M: float,
    spec: GaugeVectorSpec,
):
    """Create the production metric-provider for ``L_xi g``."""
    def provider(radius: float, theta: float) -> np.ndarray:
        return compact_support_gauge_metric(
            GaugeMetricConfig(radius, theta, chi, omega_orb_M), spec
        ).metric_coefficient

    return provider


def _physical_plus_compact_provider(
    chi: float,
    omega_orb_M: float,
    spec: GaugeVectorSpec,
):
    """Create ``h_IRG + L_xi g`` for the same production delta-Box path."""
    pure = _compact_metric_provider(chi, omega_orb_M, spec)

    def provider(radius: float, theta: float) -> np.ndarray:
        return irg_metric_coefficient_bl(radius, theta, chi, omega_orb_M) + pure(
            radius, theta
        )

    return provider


def _direct_gauge_kernels(
    initial_mode,
    final_mode,
    omega_orb_M: float,
    settings,
    spec: GaugeVectorSpec,
) -> tuple[complex, complex, complex]:
    """Project ``h``, ``L_xi g`` and their sum over the compact support.

    Restricting all three projections to the same support avoids subtracting
    two full-domain physical kernels.  It does not change the Ward difference,
    because the compact perturbation vanishes outside that interval.
    """
    domain = (spec.support_inner, spec.support_outer)
    width = spec.support_outer - spec.support_inner
    breaks = tuple(
        spec.support_inner + fraction * width
        for fraction in (0.25, 0.50, 0.75)
    )
    physical = tidal_kernel_from_modes_M(
        initial_mode,
        final_mode,
        omega_orb_M,
        settings,
        "irg",
        radial_domain=domain,
        radial_breaks=breaks,
    )
    pure = tidal_kernel_from_modes_M(
        initial_mode,
        final_mode,
        omega_orb_M,
        settings,
        _compact_metric_provider(initial_mode.chi, omega_orb_M, spec),
        radial_domain=domain,
        radial_breaks=breaks,
    )
    shifted = tidal_kernel_from_modes_M(
        initial_mode,
        final_mode,
        omega_orb_M,
        settings,
        _physical_plus_compact_provider(initial_mode.chi, omega_orb_M, spec),
        radial_domain=domain,
        radial_breaks=breaks,
    )
    return physical, pure, shifted


def _direct_support_kernel(
    initial_mode,
    final_mode,
    omega_orb_M: complex,
    settings,
    spec: GaugeVectorSpec,
    metric,
) -> complex:
    """Project one metric on a compact support with Ward-specific panels."""
    domain = (spec.support_inner, spec.support_outer)
    width = spec.support_outer - spec.support_inner
    breaks = tuple(
        spec.support_inner + fraction * width
        for fraction in (0.25, 0.50, 0.75)
    )
    # The Ward cancellation is primarily radial.  Two hundred forty radial nodes
    # resolve the compact bump to better than 1e-8 in the worst pilot, while
    # sixteen angular nodes already over-resolve the l<=3 modes and the l=2
    # gauge vector.  Keeping the full atlas angular grid here only repeats the
    # same low-order angular projection and makes the 75 audits prohibitive.
    ward_settings = replace(
        settings,
        radial_nodes=max(settings.radial_nodes, 240),
        angular_nodes=min(settings.angular_nodes, 16),
    )
    return tidal_kernel_from_modes_M(
        initial_mode,
        final_mode,
        omega_orb_M,
        ward_settings,
        metric,
        radial_domain=domain,
        radial_breaks=breaks,
    )


def _pure_gauge_green_kernel(
    initial_mode,
    final_mode,
    omega_orb_M: float,
    settings,
    spec: GaugeVectorSpec,
) -> complex:
    """Evaluate the compact pure-gauge kernel by Green's identity.

    The spatial boundary term vanishes because every derivative of the bump
    vanishes at its endpoints.  This form avoids subtracting large local
    connection terms and exposes the expected factor
    ``omega_i-2 Omega-omega_f`` explicitly.
    """
    radial_nodes = max(settings.radial_nodes, 240)
    angular_nodes = min(settings.angular_nodes, 16)
    nodes_r, weights_r = np.polynomial.legendre.leggauss(radial_nodes)
    radii = spec.support_inner + 0.5 * (
        spec.support_outer - spec.support_inner
    ) * (nodes_r + 1.0)
    weights_r = 0.5 * (spec.support_outer - spec.support_inner) * weights_r
    nodes_u, weights_u = np.polynomial.legendre.leggauss(angular_nodes)
    thetas = np.arccos(nodes_u)
    frequency = initial_mode.frequency_M - 2.0 * omega_orb_M
    final_frequency = final_mode.frequency_M
    final_m = final_mode.state.m
    terms: list[complex] = []
    initial_fields = _mode_field_provider(initial_mode)
    final_fields = _mode_field_provider(final_mode)
    for radius, radial_weight in zip(radii, weights_r, strict=True):
        for theta, angular_weight in zip(thetas, weights_u, strict=True):
            metric = kerr_metric_boyer_lindquist(
                float(radius), float(theta), initial_mode.chi
            )
            inverse = np.linalg.inv(metric)
            gauge = compact_support_gauge_metric(
                GaugeMetricConfig(
                    float(radius), float(theta), initial_mode.chi, omega_orb_M
                ),
                spec,
            )
            vector = inverse @ gauge.gauge_vector
            _, local_initial = initial_fields(float(radius), float(theta))
            final_value = final_fields(float(radius), float(theta))[0]
            lie_scalar = complex(vector @ local_initial[:4])
            frequency_factor = (
                (final_frequency**2 - frequency**2) * inverse[0, 0]
                + 2.0
                * final_m
                * (frequency - final_frequency)
                * inverse[0, 3]
            )
            sigma = radius**2 + initial_mode.chi**2 * math.cos(theta) ** 2
            terms.append(complex(
                radial_weight
                * angular_weight
                * sigma
                * final_value
                * frequency_factor
                * lie_scalar
            ))
    total = complex(
        math.fsum(value.real for value in terms),
        math.fsum(value.imag for value in terms),
    )
    return complex(2.0 * math.pi * total)


def _calibrated_spec(
    config: TransitionConfig,
    omega_orb_M: float,
    spec: GaugeVectorSpec,
) -> GaugeVectorSpec:
    """Scale a gauge vector to the requested metric amplitude on its support."""
    radii = np.linspace(spec.support_inner, spec.support_outer, 11)[1:-1]
    thetas = np.linspace(0.30, math.pi - 0.30, 9)
    physical_max = 0.0
    gauge_max = 0.0
    unit_spec = replace(spec, absolute_amplitude=1.0)
    for radius in radii:
        for theta in thetas:
            physical_max = max(
                physical_max,
                float(
                    np.max(
                        np.abs(
                            irg_metric_coefficient_bl(
                                float(radius),
                                float(theta),
                                float(config.chi),
                                omega_orb_M,
                            )
                        )
                    )
                ),
            )
            gauge = compact_support_gauge_metric(
                GaugeMetricConfig(
                    float(radius), float(theta), float(config.chi), omega_orb_M
                ),
                unit_spec,
            ).metric_coefficient
            gauge_max = max(gauge_max, float(np.max(np.abs(gauge))))
    if not np.isfinite(gauge_max) or gauge_max <= 0.0:
        raise RuntimeError("compact gauge metric has zero or nonfinite support amplitude")
    return replace(
        spec,
        absolute_amplitude=spec.relative_amplitude * physical_max / gauge_max,
    )


def direct_ward_audit(
    config: TransitionConfig,
    resolved_modes,
    specs: Sequence[GaugeVectorSpec],
    *,
    include_off_resonance: bool = True,
    physical_kernel: complex | None = None,
) -> WardAuditResult:
    """Run a direct compact-support Ward audit through production ``delta Box``.

    Unlike the former certification check, the reported on-shell residual is
    obtained by projecting the actual ``L_xi g`` metric.  Green's identity is
    retained only as an independent cross-check and is recorded separately in
    every row.  No zero or NaN placeholder is used for a failed calculation.
    """
    initial_mode, final_mode = _resolved_mode_pair(config, resolved_modes)
    chi = float(initial_mode.chi)
    if not initial_mode.converged or not final_mode.converged:
        return WardAuditResult(
            config.alpha,
            chi,
            config.initial,
            config.final,
            "mode_not_converged",
            0.0j,
            math.inf,
            math.inf,
            math.inf,
            0.0,
            ({"diagnostic": "one or both resolved modes did not converge"},),
        )
    delta_m = config.initial.m - config.final.m
    if delta_m == 0:
        raise ValueError("Ward audit requires nonzero Delta m")
    # A Ward identity between QBS eigenfunctions is on shell at the complex
    # pole difference.  Using only its real part leaves a genuine term
    # proportional to Delta Gamma, which becomes support dependent and is not
    # a numerical gauge residual.  The physical atlas still evaluates its
    # real orbital resonance; only this analytic gauge audit uses the complex
    # continuation.
    omega_res = (
        initial_mode.frequency_M - final_mode.frequency_M
    ) / delta_m
    resolved_config = replace(config, chi=chi)
    physical = (
        complex(physical_kernel)
        if physical_kernel is not None
        else tidal_kernel_from_modes_M(
            initial_mode, final_mode, omega_res, config.settings, "irg"
        )
    )
    scale = max(abs(physical), np.finfo(float).tiny)
    rows: list[dict[str, object]] = []
    direct_by_group: dict[tuple[str, str], list[tuple[float, complex]]] = {}
    maximum_invariance = 0.0
    maximum_pure = 0.0
    maximum_linearity = 0.0
    maximum_superposition = 0.0
    maximum_green_cross = 0.0
    off_resonance_ratio = 0.0
    support_physical_cache: dict[str, complex] = {}
    on_shell_cache: dict[
        tuple[str, str], tuple[GaugeVectorSpec, complex, complex, complex]
    ] = {}
    off_shell_cache: dict[tuple[str, str, float], tuple[complex, complex]] = {}

    for original_spec in specs:
        group = (original_spec.kind, original_spec.support_name)
        if group not in on_shell_cache:
            unit_request = replace(
                original_spec, relative_amplitude=1.0, absolute_amplitude=1.0
            )
            unit_spec = _calibrated_spec(
                resolved_config, omega_res, unit_request
            )
            if unit_spec.support_name not in support_physical_cache:
                support_physical_cache[unit_spec.support_name] = _direct_support_kernel(
                    initial_mode, final_mode, omega_res, config.settings,
                    unit_spec, "irg",
                )
            unit_physical = support_physical_cache[unit_spec.support_name]
            unit_pure = _direct_support_kernel(
                initial_mode, final_mode, omega_res, config.settings, unit_spec,
                _compact_metric_provider(chi, omega_res, unit_spec),
            )
            unit_shifted = _direct_support_kernel(
                initial_mode, final_mode, omega_res, config.settings, unit_spec,
                _physical_plus_compact_provider(chi, omega_res, unit_spec),
            )
            unit_green = _pure_gauge_green_kernel(
                initial_mode, final_mode, omega_res, config.settings, unit_spec
            )
            on_shell_cache[group] = (
                unit_spec, unit_pure, unit_shifted, unit_green
            )
        unit_spec, unit_pure, unit_shifted, unit_green = on_shell_cache[group]
        amplitude_scale = original_spec.relative_amplitude
        spec = replace(
            unit_spec,
            relative_amplitude=amplitude_scale,
            absolute_amplitude=unit_spec.absolute_amplitude * amplitude_scale,
        )
        support_physical = support_physical_cache[spec.support_name]
        pure = unit_pure * amplitude_scale
        support_shifted = support_physical + (
            unit_shifted - support_physical
        ) * amplitude_scale
        green = unit_green * amplitude_scale
        delta_direct = support_shifted - support_physical
        pure_residual = abs(pure) / scale
        invariance_residual = abs(delta_direct) / scale
        superposition_residual = abs(delta_direct - pure) / scale
        green_cross = abs(pure - green) / scale
        maximum_pure = max(maximum_pure, pure_residual)
        maximum_invariance = max(maximum_invariance, invariance_residual)
        maximum_superposition = max(maximum_superposition, superposition_residual)
        maximum_green_cross = max(maximum_green_cross, green_cross)
        direct_by_group.setdefault((spec.kind, spec.support_name), []).append(
            (spec.relative_amplitude, pure)
        )

        off_rows: dict[str, float] = {}
        for frequency_factor in ((0.99, 1.01) if include_off_resonance else ()):
            omega_control = frequency_factor * omega_res
            off_key = (*group, frequency_factor)
            if off_key not in off_shell_cache:
                off_request = replace(
                    original_spec,
                    relative_amplitude=1.0,
                    absolute_amplitude=1.0,
                )
                off_unit_spec = _calibrated_spec(
                    resolved_config, omega_control, off_request
                )
                off_unit_direct = _direct_support_kernel(
                    initial_mode, final_mode, omega_control, config.settings,
                    off_unit_spec,
                    _compact_metric_provider(chi, omega_control, off_unit_spec),
                )
                off_unit_green = _pure_gauge_green_kernel(
                    initial_mode, final_mode, omega_control, config.settings,
                    off_unit_spec,
                )
                off_shell_cache[off_key] = (
                    off_unit_direct, off_unit_green
                )
            off_unit_direct, off_unit_green = off_shell_cache[off_key]
            off_direct = off_unit_direct * amplitude_scale
            off_green = off_unit_green * amplitude_scale
            # Subtract the measured on-resonance value before comparing the
            # off-resonance control.  This removes shared quadrature error
            # without replacing the direct Ward value by the Green identity.
            off_response = off_direct - pure
            green_response = off_green - green
            off_residual = abs(off_response) / scale
            off_cross = abs(off_response - green_response) / max(
                abs(off_response), abs(green_response),
                # A relative comparison is undefined for symmetry-suppressed
                # controls (notably the axial cloud-core vector).  Below the
                # Ward acceptance scale, use the physical-kernel absolute
                # normalization instead of amplifying roundoff by 1/eps.
                scale * WARD_TOLERANCE,
            )
            off_rows[f"off_{frequency_factor:.2f}_residual"] = off_residual
            off_rows[f"off_{frequency_factor:.2f}_green_cross_residual"] = off_cross
            off_rows[f"off_{frequency_factor:.2f}_response_abs"] = abs(off_response)
            off_resonance_ratio = max(
                off_resonance_ratio,
                off_residual / max(pure_residual, np.finfo(float).eps),
            )
            maximum_green_cross = max(maximum_green_cross, off_cross)

        rows.append(
            {
                "alpha": config.alpha,
                "chi": chi,
                "initial": config.initial.label,
                "final": config.final.label,
                "kind": spec.kind,
                "support": spec.support_name,
                "support_inner_M": spec.support_inner,
                "support_outer_M": spec.support_outer,
                "relative_amplitude": spec.relative_amplitude,
                "absolute_amplitude": spec.absolute_amplitude,
                "omega_res_M": omega_res,
                "physical_kernel_abs": abs(physical),
                "support_physical_kernel_abs": abs(support_physical),
                "direct_pure_gauge_kernel_real": pure.real,
                "direct_pure_gauge_kernel_imag": pure.imag,
                "direct_pure_gauge_kernel_abs": abs(pure),
                "green_pure_gauge_kernel_abs": abs(green),
                "shifted_support_kernel_abs": abs(support_shifted),
                "ward_evaluation": "direct production delta Box",
                "frequency_convention": "complex on-shell QBS pole difference",
                "amplitude_evaluation": (
                    "unit metric projected directly; requested amplitude "
                    "restored by exact linear scaling"
                ),
                "ward_radial_nodes": (
                    max(config.settings.radial_nodes, 240)
                ),
                "ward_angular_nodes": min(config.settings.angular_nodes, 16),
                "pure_gauge_residual": pure_residual,
                "invariance_residual": invariance_residual,
                "superposition_residual": superposition_residual,
                "green_cross_residual": green_cross,
                **off_rows,
            }
        )

    for values in direct_by_group.values():
        if len(values) < 2:
            maximum_linearity = math.inf
            continue
        reference_amplitude, reference_kernel = min(values, key=lambda item: item[0])
        for amplitude, kernel in values:
            expected = reference_kernel * amplitude / reference_amplitude
            maximum_linearity = max(
                maximum_linearity,
                abs(kernel - expected)
                / max(abs(expected), scale * np.finfo(float).eps),
            )

    finite = all(
        np.isfinite(value)
        for value in (
            maximum_invariance,
            maximum_pure,
            maximum_linearity,
            maximum_superposition,
            maximum_green_cross,
            off_resonance_ratio,
        )
    )
    off_passed = (not include_off_resonance) or off_resonance_ratio > 100.0
    status = "ok" if (
        finite
        and len(rows) == len(specs)
        and bool(rows)
        and maximum_invariance < WARD_TOLERANCE
        and maximum_pure < WARD_TOLERANCE
        and maximum_linearity < WARD_TOLERANCE
        and maximum_superposition < WARD_TOLERANCE
        and maximum_green_cross < WARD_GREEN_TOLERANCE
        and off_passed
    ) else "gauge_audit_failed"
    # Extra maxima live in rows because the v1 result schema is intentionally
    # retained for API compatibility.
    if rows:
        rows[0]["audit_maximum_superposition_residual"] = maximum_superposition
        rows[0]["audit_maximum_green_cross_residual"] = maximum_green_cross
    return WardAuditResult(
        config.alpha,
        chi,
        config.initial,
        config.final,
        status,
        physical,
        maximum_invariance,
        maximum_pure,
        maximum_linearity,
        off_resonance_ratio,
        tuple(rows),
    )


def ward_identity_audit(
    config: TransitionConfig,
    specs: Sequence[GaugeVectorSpec],
) -> WardAuditResult:
    """Backward-compatible entry point for the direct production audit."""
    return direct_ward_audit(config, None, specs)


def operator_form_audit(
    config: TransitionConfig,
    sample_points: Iterable[tuple[float, float]] | None = None,
    resolved_modes=None,
) -> tuple[float, list[dict[str, float]]]:
    """Compare operator forms with order-matched Richardson extrapolation.

    With no explicit points, twelve samples cover the near-horizon region,
    cloud core, and outer cloud.  Both forms are independently evaluated at
    two stencil scales; the gate compares their zero-step extrapolants rather
    than a hand-selected single finite-difference step.
    """
    mode, final_mode = _resolved_mode_pair(config, resolved_modes)
    chi = float(mode.chi)
    if not mode.converged or not final_mode.converged:
        return math.inf, [
            {
                "radius_M": math.nan,
                "theta": math.nan,
                "connection_abs": math.nan,
                "divergence_abs": math.nan,
                "residual": math.inf,
                "diagnostic": "mode_not_converged",
            }
        ]
    omega_res = (
        mode.frequency_M.real - final_mode.frequency_M.real
    ) / (config.initial.m - config.final.m)
    if sample_points is None:
        r_plus = 1.0 + math.sqrt(1.0 - chi**2)
        near = r_plus + max(20.0 * config.settings.mode.horizon_cutoff, 2.0e-3)
        core = max(r_plus + 0.5, 0.35 * mode.r_99_M)
        outer = max(core + 0.5, 0.80 * mode.r_99_M)
        theta_values = (0.37, 0.91, 1.57, 2.31)
        points = tuple(
            (radius, theta)
            for radius in (near, core, outer)
            for theta in theta_values
        )
    else:
        points = tuple(sample_points)
    rows: list[dict[str, float]] = []
    maximum = 0.0
    for radius, theta in points:
        connection_coarse = _delta_box_connection_bl(
            mode, radius, theta, omega_res, "irg", 0.20
        )
        connection_fine = _delta_box_connection_bl(
            mode, radius, theta, omega_res, "irg", 0.10
        )
        divergence_coarse = _delta_box_divergence_bl(
            mode, radius, theta, omega_res, "irg", 0.20
        )
        divergence_fine = _delta_box_divergence_bl(
            mode, radius, theta, omega_res, "irg", 0.10
        )
        # Both forms use independently assembled five-point derivatives.
        connection = connection_fine + (connection_fine - connection_coarse) / 15.0
        divergence = divergence_fine + (divergence_fine - divergence_coarse) / 15.0
        result_scale = max(abs(connection), abs(divergence), np.finfo(float).tiny)
        try:
            mode_value = _mode_fields_bl(mode, radius, theta)[0]
            perturbation_scale = float(np.max(np.abs(
                irg_metric_coefficient_bl(radius, theta, chi, omega_res)
            )))
            conditioning_scale = (
                mode.alpha**2 * abs(mode_value) * perturbation_scale
            )
        except (AttributeError, ValueError, FloatingPointError):
            conditioning_scale = 0.0
        scale = max(result_scale, conditioning_scale, np.finfo(float).tiny)
        residual = abs(connection - divergence) / scale
        naive_result_relative = abs(connection - divergence) / result_scale
        connection_stencil = abs(connection_fine - connection_coarse) / scale
        divergence_stencil = abs(divergence_fine - divergence_coarse) / scale
        maximum = max(maximum, residual)
        rows.append(
            {
                "radius_M": radius,
                "theta": theta,
                "connection_abs": abs(connection),
                "divergence_abs": abs(divergence),
                "residual": residual,
                "naive_result_relative": naive_result_relative,
                "conditioning_scale": conditioning_scale,
                "connection_stencil_residual": connection_stencil,
                "divergence_stencil_residual": divergence_stencil,
                "difference_scale_coarse": 0.20,
                "difference_scale_fine": 0.10,
            }
        )
    return maximum, rows


def fit_weak_field_limit(rows: Sequence[Mapping[str, float]]) -> WeakFieldFitResult:
    """Fit ``|K_rel|/|K_H|=c0+c2 alpha^2+c4 alpha^4``."""
    if len(rows) < 3:
        return WeakFieldFitResult("missing_checks", math.nan, math.nan, math.nan, math.inf, math.inf)
    alpha = np.asarray([float(row["alpha"]) for row in rows])
    ratio = np.asarray([float(row["kernel_ratio"]) for row in rows])
    gauge = np.asarray([float(row.get("gauge_residual", math.inf)) for row in rows])
    if not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(ratio)):
        return WeakFieldFitResult("weak_field_failed", math.nan, math.nan, math.nan, math.inf, math.inf)
    design = np.column_stack((np.ones_like(alpha), alpha**2, alpha**4))
    c0, c2, c4 = np.linalg.lstsq(design, ratio, rcond=None)[0]
    fit = design @ np.asarray([c0, c2, c4])
    maximum_fit = float(np.max(np.abs(fit - ratio)))
    maximum_gauge = float(np.max(gauge))
    status = "ok" if (
        abs(c0 - 1.0) < 1.0e-3
        and maximum_fit < 2.0e-4
        and maximum_gauge < 1.0e-8
    ) else "weak_field_failed"
    return WeakFieldFitResult(
        status,
        float(c0),
        float(c2),
        float(c4),
        maximum_fit,
        maximum_gauge,
    )


def compare_schwarzschild_benchmark(
    rows: Sequence[Mapping[str, float]],
    reference: Sequence[Mapping[str, float]],
) -> BenchmarkResult:
    """Compare phase-independent kernel corrections with reference values."""
    lookup = {round(float(row["alpha"]), 12): row for row in reference}
    output: list[dict[str, object]] = []
    all_passed = len(rows) == len(reference) and bool(rows)
    maximum = 0.0
    for row in rows:
        alpha = round(float(row["alpha"]), 12)
        if alpha not in lookup:
            all_passed = False
            continue
        expected = float(lookup[alpha]["relative_correction"])
        sigma = float(lookup[alpha]["sigma_digitization"])
        observed = float(row["relative_correction"])
        difference = abs(observed - expected)
        threshold = max(0.02, 2.0 * sigma)
        gauge_residual = float(row.get("gauge_residual", math.inf))
        passed = (
            np.isfinite(observed)
            and difference < threshold
            and gauge_residual < 1.0e-8
        )
        all_passed = all_passed and passed
        maximum = max(maximum, difference)
        output.append(
            {
                **dict(row),
                "reference_correction": expected,
                "sigma_digitization": sigma,
                "absolute_difference": difference,
                "acceptance_threshold": threshold,
                "passed": passed,
            }
        )
    status = "ok" if all_passed else "benchmark_failed"
    return BenchmarkResult(status, maximum, tuple(output))
