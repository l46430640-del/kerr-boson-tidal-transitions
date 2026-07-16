#!/usr/bin/env python
"""Generate benchmark timescale tables and a hierarchy plot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from boson_ep.ep import find_ep_roots
from boson_ep.spectrum import saturation_spin_numeric
from boson_ep.timescales import compute_timescales


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/baseline"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    plot_result = None
    for alpha in (0.05, 0.10, 0.20, 0.30):
        for policy, chi in (
            ("saturation_numeric", saturation_spin_numeric(alpha)),
            ("high_0.9", 0.9),
            ("high_0.99", 0.99),
        ):
            roots = find_ep_roots(alpha, chi)
            for root in roots:
                if root.q is None:
                    continue
                for mass in (10.0, 100.0, 1.0e6):
                    result = compute_timescales(alpha, chi, root.q, mass)
                    if (
                        plot_result is None
                        and root.status == "physical_root"
                        and policy == "high_0.99"
                        and alpha == 0.10
                        and mass == 10.0
                    ):
                        plot_result = result
                    row: dict[str, object] = {
                        "alpha": alpha,
                        "spin_policy": policy,
                        "chi": chi,
                        "q_EP": root.q,
                        "status": root.status,
                        "mass_msun": mass,
                        "M_Omega_res": result.omega_res_M,
                        "M_eta": result.eta_M,
                        "z": result.landau_zener_z,
                        "R_res_over_r_cloud": result.radius_over_cloud,
                    }
                    for name, value in result.times_M.items():
                        row[f"{name}_M"] = value
                        row[f"{name}_seconds"] = result.times_seconds[name]
                    rows.append(row)

    if rows:
        with (args.output / "timescales.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "timescales.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    if plot_result is not None:
        names = list(plot_result.hierarchy)
        values = np.asarray([plot_result.times_M[name] for name in names])
        fig, axis = plt.subplots(figsize=(8.5, 4.8))
        axis.bar(names, values, color="tab:blue")
        axis.set_yscale("log")
        axis.set_ylabel(r"timescale $T/M$")
        axis.set_title(
            fr"Timescale hierarchy: $\alpha={plot_result.alpha}$, "
            fr"$\chi={plot_result.chi}$, $q_{{EP}}={plot_result.q:.3g}$"
        )
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25, which="both")
        fig.tight_layout()
        fig.savefig(args.output / "timescale_hierarchy.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()

