#!/usr/bin/env python
"""Reproduce symbolic and numerical baseline checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from boson_ep.ep import delta_gamma_M, find_ep_roots
from boson_ep.models import PRIMARY_FINAL, PRIMARY_INITIAL
from boson_ep.spectrum import (
    gamma_detweiler_M,
    saturation_spin_approx,
    saturation_spin_numeric,
)
from boson_ep.symbolic import derive_primary_symbolics
from boson_ep.tides import resonance_frequency_M, tidal_eta_M_array, tidal_eta_far_M
from boson_ep.timescales import compute_timescales


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def benchmark_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for alpha in (0.05, 0.10, 0.20, 0.30):
        policies = (
            ("saturation_approx", saturation_spin_approx(alpha)),
            ("saturation_numeric", saturation_spin_numeric(alpha)),
            ("high_0.9", 0.9),
            ("high_0.99", 0.99),
        )
        for policy, chi in policies:
            roots = find_ep_roots(alpha, chi)
            root = roots[0]
            rows.append(
                {
                    "alpha": alpha,
                    "spin_policy": policy,
                    "chi": chi,
                    "M_Gamma_211": gamma_detweiler_M(alpha, chi, PRIMARY_INITIAL),
                    "M_Gamma_21-1": gamma_detweiler_M(alpha, chi, PRIMARY_FINAL),
                    "M_Delta_Gamma": delta_gamma_M(alpha, chi),
                    "M_Omega_res": resonance_frequency_M(alpha, chi),
                    "status": root.status,
                    "q_EP": root.q,
                    "q_EP_far": root.analytic_q,
                    "R_res_over_r_cloud": root.radius_over_cloud,
                    "root_residual": root.residual,
                }
            )
    return rows


def plot_scalings(output: Path) -> None:
    alphas = np.geomspace(0.03, 0.30, 160)
    chi = 0.99
    omega = np.asarray([resonance_frequency_M(a, chi) for a in alphas])
    delta_gamma = np.asarray([abs(delta_gamma_M(a, chi)) for a in alphas])
    eta = np.asarray([tidal_eta_far_M(a, chi, 0.5) for a in alphas])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].loglog(alphas, omega, label=r"$M\Omega_{\rm res}$")
    axes[0].loglog(
        alphas,
        omega[0] * (alphas / alphas[0]) ** 6,
        "--",
        label=r"$\alpha^6$",
    )
    axes[0].set(xlabel=r"$\alpha$", ylabel="dimensionless frequency")
    axes[0].legend()
    axes[0].grid(alpha=0.25, which="both")

    axes[1].loglog(alphas, delta_gamma, label=r"$M|\Delta\Gamma|$")
    axes[1].loglog(alphas, eta, label=r"$M\eta(q=0.5)$")
    axes[1].loglog(
        alphas,
        eta[0] * (alphas / alphas[0]) ** 9,
        "--",
        label=r"$\alpha^9$",
    )
    axes[1].set(xlabel=r"$\alpha$", ylabel="dimensionless rate")
    axes[1].legend()
    axes[1].grid(alpha=0.25, which="both")
    fig.suptitle(r"Hydrogenic scaling checks at $\chi=0.99$")
    fig.tight_layout()
    fig.savefig(output / "scaling_checks.png", dpi=180)
    plt.close(fig)


def plot_eta_intersections(output: Path) -> None:
    q_grid = np.geomspace(1.0e-4, 10.0, 600)
    chi = 0.99
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)
    for axis, alpha in zip(axes, (0.05, 0.10), strict=True):
        eta = tidal_eta_M_array(alpha, chi, q_grid)
        target = abs(delta_gamma_M(alpha, chi))
        roots = find_ep_roots(alpha, chi)
        axis.loglog(q_grid, 2.0 * eta, label=r"$2M\eta(q)$")
        axis.axhline(target, color="black", linestyle="--", label=r"$M|\Delta\Gamma|$")
        for root in roots:
            if root.q is not None and root.q <= 10.0:
                axis.axvline(root.q, color="tab:red", linestyle=":")
        axis.set_title(fr"$\alpha={alpha:.2f},\ \chi={chi}$")
        axis.set_xlabel(r"$q$")
        axis.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("dimensionless rate")
    axes[0].legend()
    fig.suptitle("Full piecewise tidal coupling and EP intersections")
    fig.tight_layout()
    fig.savefig(output / "eta_intersections.png", dpi=180)
    plt.close(fig)


def build_report(
    output: Path,
    symbolic: dict[str, str],
    rows: list[dict[str, object]],
) -> None:
    physical = [row for row in rows if row["status"] == "physical_root"]
    saturation = [row for row in rows if str(row["spin_policy"]).startswith("saturation")]
    lines = [
        "# Hydrogenic Baseline Report",
        "",
        "## Conventions",
        "",
        "- Units: `G=c=hbar=M=1`; rates are reported as `M Gamma`.",
        "- Real spectrum: `M omega_R = alpha[1-alpha^2/8-17alpha^4/128+m chi alpha^5/12]`.",
        "- Imaginary spectrum: `M Gamma = 2 r_+ C_21m (m Omega_H-omega_R) alpha^9`, "
        "with `C_21m=[1-chi^2+(chi m-2 r_+ alpha)^2]/48`.",
        "- Factor-of-two convention: `Gamma` is the amplitude rate used on the Hamiltonian diagonal as `+i Gamma`; "
        "the occupation rate is `2 Gamma`.",
        "- Transition: circular, equatorial, corotating `|211> -> |21-1>` with `l*=m*=2`.",
        "- `q_EP` is obtained from all sign changes on 512 logarithmic samples over `10^-6<=q<=10^3`, "
        "then refined with `scipy.optimize.brentq` using the complete radial pieces.",
        "",
        "## Symbolic checks",
        "",
    ]
    for name, value in symbolic.items():
        lines.append(f"- `{name}` = `{value}`")
    lines.extend(
        [
            "",
            "The symbolic calculation gives `M Omega_res = chi alpha^6 / 12`, "
            "`|K|=9`, and a Detweiler rate scaling `Gamma ~ alpha^9/M`, not "
            "`alpha^10/M`.",
            "",
            "## Benchmark EP results",
            "",
            "| alpha | spin policy | chi | status | q_EP | q_EP far | R_res/(4 r0) |",
            "|---:|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        q_value = "-" if row["q_EP"] is None else f"{float(row['q_EP']):.8g}"
        q_far = "-" if row["q_EP_far"] is None else f"{float(row['q_EP_far']):.8g}"
        radius = (
            "-"
            if row["R_res_over_r_cloud"] is None
            else f"{float(row['R_res_over_r_cloud']):.6g}"
        )
        lines.append(
            f"| {float(row['alpha']):.2f} | {row['spin_policy']} | "
            f"{float(row['chi']):.8f} | {row['status']} | {q_value} | {q_far} | {radius} |"
        )

    lines.extend(["", "## Go/No-Go summary", ""])
    if physical:
        best = min(physical, key=lambda row: float(row["q_EP"]))
        lines.append(
            "- **GO (high-spin baseline):** a physical `q_EP<=1` exists; the smallest "
            f"benchmark root is `q_EP={float(best['q_EP']):.6g}` at "
            f"`alpha={float(best['alpha']):.2f}`, `chi={float(best['chi']):.3f}`."
        )
    else:
        lines.append("- **NO-GO:** no physical high-spin benchmark root was found.")
    saturation_roots = [row for row in saturation if row["q_EP"] is not None]
    if saturation_roots:
        lines.append("- Saturation-spin roots exist only where explicitly listed in the table.")
    else:
        lines.append("- No saturation-spin EP root was found in the diagnostic `q<=10^3` interval.")

    selected = next(
        (
            row
            for row in rows
            if row["status"] == "physical_root"
            and row["spin_policy"] == "high_0.99"
            and abs(float(row["alpha"]) - 0.10) < 1.0e-12
        ),
        physical[0] if physical else None,
    )
    if selected is not None:
        result = compute_timescales(
            float(selected["alpha"]),
            float(selected["chi"]),
            float(selected["q_EP"]),
            10.0,
        )
        lines.extend(["", "## Reference timescale hierarchy", ""])
        lines.append(
            f"At `alpha={result.alpha}`, `chi={result.chi}`, `q={result.q:.8g}`, "
            "and `M=10 M_sun`:"
        )
        lines.append("")
        for name in result.hierarchy:
            lines.append(
                f"- `{name}` = `{result.times_M[name]:.8e} M` = "
                f"`{result.times_seconds[name]:.8e} s`"
            )
        lines.append(f"- `z = {result.landau_zener_z:.8e}`")
        lines.append(f"- `R_res/(4 r0) = {result.radius_over_cloud:.8e}`")

    lines.extend(
        [
            "",
            "## Approximation warning",
            "",
            "These roots are exact only within the stated hydrogenic, corrected-Detweiler, "
            "quadrupolar model. The dominant systematic is the small-alpha Detweiler rate; "
            "subleading relativistic wavefunctions, higher tidal multipoles, additional levels, "
            "eccentricity/inclination, cloud self-gravity, and orbital backreaction are not included. "
            "A numerical root is therefore not yet a model-independent robustness claim.",
            "",
        ]
    )
    (output / "baseline_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/baseline"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    symbolic_raw = derive_primary_symbolics()
    symbolic = {name: str(value) for name, value in symbolic_raw.items()}
    rows = benchmark_rows()
    write_csv(args.output / "baseline_benchmarks.csv", rows)
    (args.output / "baseline_summary.json").write_text(
        json.dumps({"symbolic": symbolic, "benchmarks": rows}, indent=2),
        encoding="utf-8",
    )
    plot_scalings(args.output)
    plot_eta_intersections(args.output)
    build_report(args.output, symbolic, rows)


if __name__ == "__main__":
    main()
