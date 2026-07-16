"""Audit weak-field normalization and Schwarzschild IRG/RW gauge agreement."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from boson_ep import (
    KerrModeSettings,
    RelativisticTideSettings,
    State,
    schwarzschild_irg_tidal_kernel_M,
    schwarzschild_rw_tidal_kernel_M,
)
from boson_ep.relativistic_tides import hydrogenic_newtonian_kernel_M


OUTPUT = Path("results/relativistic_tides")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    initial = State(2, 1, 1)
    final = State(2, 1, -1)
    settings = RelativisticTideSettings(
        mode=KerrModeSettings(
            truncation=150,
            angular_lmax=10,
            series_terms=150,
            angular_nodes=48,
        ),
        angular_nodes=16,
    )
    rows: list[dict[str, object]] = []
    for alpha in (0.05, 0.075, 0.10):
        hydrogenic = hydrogenic_newtonian_kernel_M(alpha, initial, final)
        irg = schwarzschild_irg_tidal_kernel_M(
            alpha, initial, final, settings
        )
        rw = schwarzschild_rw_tidal_kernel_M(alpha, initial, final, settings)
        weak_ratio = abs(irg) / abs(hydrogenic)
        gauge_residual = abs(irg - rw) / max(abs(rw), 1.0e-300)
        rows.append(
            {
                "alpha": alpha,
                "hydrogenic_abs": abs(hydrogenic),
                "irg_real": irg.real,
                "irg_imag": irg.imag,
                "irg_abs": abs(irg),
                "rw_real": rw.real,
                "rw_imag": rw.imag,
                "rw_abs": abs(rw),
                "irg_over_hydrogenic": weak_ratio,
                "leading_alpha2_coefficient": (1.0 - weak_ratio) / alpha**2,
                "irg_rw_relative_residual": gauge_residual,
                "gauge_pass": gauge_residual < 1.0e-8,
                "weak_field_pass": (
                    abs(weak_ratio - 1.0) < 0.01 if alpha <= 0.075 else None
                ),
            }
        )
    with (OUTPUT / "weak_field_gauge_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "weak_field_gauge_audit.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.plot(
        [row["alpha"] ** 2 for row in rows],
        [row["irg_over_hydrogenic"] for row in rows],
        marker="o",
        label="IRG / hydrogenic",
    )
    axis.plot(
        [row["alpha"] ** 2 for row in rows],
        [row["rw_abs"] / row["hydrogenic_abs"] for row in rows],
        linestyle="--",
        label="RW / hydrogenic",
    )
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.set_xlabel(r"$\alpha^2$")
    axis.set_ylabel("kernel magnitude ratio")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / "weak_field_gauge_audit.png", dpi=180)
    plt.close(figure)

    summary = {
        "all_gauge_pass": all(bool(row["gauge_pass"]) for row in rows),
        "alpha_005_weak_field_pass": bool(rows[0]["weak_field_pass"]),
        "alpha_0075_weak_field_pass": bool(rows[1]["weak_field_pass"]),
        "alpha2_coefficient_range": [
            min(float(row["leading_alpha2_coefficient"]) for row in rows),
            max(float(row["leading_alpha2_coefficient"]) for row in rows),
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
