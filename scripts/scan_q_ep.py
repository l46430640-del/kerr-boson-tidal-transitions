#!/usr/bin/env python
"""Scan q_EP over the fixed alpha-chi grid and draw the phase map."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from boson_ep.ep import find_ep_roots
from boson_ep.spectrum import saturation_spin_approx, saturation_spin_numeric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/baseline"))
    parser.add_argument("--n-alpha", type=int, default=81)
    parser.add_argument("--n-chi", type=int, default=181)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    alphas = np.geomspace(0.03, 0.30, args.n_alpha)
    chis = np.linspace(0.10, 0.999, args.n_chi)
    q_map = np.full((args.n_chi, args.n_alpha), np.nan)
    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()

    for alpha_index, alpha in enumerate(alphas):
        for chi_index, chi in enumerate(chis):
            results = find_ep_roots(float(alpha), float(chi))
            result = results[0]
            counts[result.status] += 1
            if result.q is not None:
                q_map[chi_index, alpha_index] = result.q
            rows.append(result.to_dict())

    csv_path = args.output / "q_ep_scan.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "q_ep_scan_summary.json").write_text(
        json.dumps(
            {
                "alpha_points": args.n_alpha,
                "chi_points": args.n_chi,
                "status_counts": dict(counts),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    masked = np.ma.masked_invalid(q_map)
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    mesh = axis.pcolormesh(
        alphas,
        chis,
        np.ma.log10(masked),
        shading="auto",
        cmap="viridis",
        vmin=-4,
        vmax=3,
    )
    colorbar = fig.colorbar(mesh, ax=axis)
    colorbar.set_label(r"$\log_{10}q_{\rm EP}$")
    sat_approx = np.asarray([saturation_spin_approx(float(a)) for a in alphas])
    sat_numeric = np.asarray([saturation_spin_numeric(float(a)) for a in alphas])
    axis.plot(alphas, sat_approx, color="black", linestyle="--", label="approx. saturation")
    axis.plot(alphas, sat_numeric, color="red", linestyle=":", label="numeric saturation")
    axis.contour(
        alphas,
        chis,
        q_map,
        levels=[1.0],
        colors=["white"],
        linewidths=[1.2],
    )
    axis.set_xscale("log")
    axis.set(xlabel=r"$\alpha$", ylabel=r"$\chi$", title=r"Hydrogenic $q_{\rm EP}(\alpha,\chi)$")
    handles, labels = axis.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", linewidth=1.2))
    labels.append(r"$q_{\rm EP}=1$")
    axis.legend(handles, labels, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.output / "q_ep_map.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
