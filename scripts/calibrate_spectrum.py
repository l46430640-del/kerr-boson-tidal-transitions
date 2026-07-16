"""Calibrate scalar quasibound frequencies and audit truncation convergence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from boson_ep.models import CFSettings, PRIMARY_FINAL, PRIMARY_INITIAL
from boson_ep.relativity import solve_quasibound_cf
from boson_ep.spectrum import gamma_detweiler_M


OUTPUT = Path("results/dynamics")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    continuations: dict[tuple[float, str, int], complex] = {}
    for chi in (0.90, 0.99):
        for state in (PRIMARY_INITIAL, PRIMARY_FINAL):
            previous_by_n: dict[int, complex] = {}
            for alpha in (0.15, 0.20, 0.25, 0.30):
                for truncation in (100, 200, 400):
                    result = solve_quasibound_cf(
                        alpha,
                        chi,
                        state,
                        CFSettings(truncation=truncation),
                        initial_guess=previous_by_n.get(truncation),
                    )
                    previous_by_n[truncation] = result.frequency_M
                    rows.append(
                        {
                            **result.to_dict(),
                            "detweiler_gamma_M": gamma_detweiler_M(alpha, chi, state),
                        }
                    )
                    continuations[(chi, state.label, truncation)] = result.frequency_M

    for row in rows:
        peers = [
            item
            for item in rows
            if item["alpha"] == row["alpha"]
            and item["chi"] == row["chi"]
            and item["state"] == row["state"]
        ]
        by_truncation = {int(item["truncation"]): item for item in peers}
        gamma_200 = float(by_truncation[200]["gamma_M"])
        gamma_400 = float(by_truncation[400]["gamma_M"])
        relative_change = abs(gamma_400 - gamma_200) / max(
            abs(gamma_400), 1.0e-300
        )
        row["truncation_relative_change_200_400"] = relative_change
        row["truncation_converged"] = bool(
            all(bool(item["converged"]) for item in peers)
            and relative_change < 1.0e-3
        )

    # Independent published maximum-rate check from Dolan 2007.  It is not a
    # substitute for the requested point-by-point Berti et al. calibration.
    benchmark = solve_quasibound_cf(
        0.42, 0.99, PRIMARY_INITIAL, CFSettings(truncation=400)
    )
    reference_gamma = 1.5e-7
    benchmark_relative_error = abs(benchmark.frequency_M.imag / reference_gamma - 1.0)
    summary = {
        "dolan_2007_alpha": 0.42,
        "dolan_2007_chi": 0.99,
        "computed_gamma_M": benchmark.frequency_M.imag,
        "reference_gamma_M": reference_gamma,
        "relative_error": benchmark_relative_error,
        "passes_2_percent": benchmark.converged and benchmark_relative_error < 0.02,
        "berti_2019_pointwise_benchmark_available": False,
        "physical_calibration_status": "calibration_failed",
        "n100_200_400_converged_groups": sum(
            1
            for row in rows
            if row["truncation"] == 400 and row["truncation_converged"]
        ),
        "total_calibration_groups": len(rows) // 3,
    }

    fieldnames = list(rows[0])
    with (OUTPUT / "spectrum_calibration.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT / "spectrum_calibration.json").open("w", encoding="utf-8") as handle:
        json.dump({"rows": rows, "benchmark": summary}, handle, indent=2)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for chi in (0.90, 0.99):
        selected = [
            row
            for row in rows
            if row["chi"] == chi
            and row["state"] == PRIMARY_INITIAL.label
            and row["truncation"] == 400
        ]
        axis.plot(
            [row["alpha"] for row in selected],
            [row["gamma_M"] for row in selected],
            marker="o",
            label=fr"CF $\chi={chi}$",
        )
        axis.plot(
            [row["alpha"] for row in selected],
            [row["detweiler_gamma_M"] for row in selected],
            linestyle="--",
            label=fr"Detweiler $\chi={chi}$",
        )
    axis.set_yscale("symlog", linthresh=1.0e-14)
    axis.set_xlabel(r"$\alpha$")
    axis.set_ylabel(r"$M\Gamma_{211}$")
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT / "cf_detweiler_comparison.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
