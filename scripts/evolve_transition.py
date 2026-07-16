"""Run a single two-level transition and save the trajectory/model comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from boson_ep import EvolutionConfig, evolve_transition, find_ep_roots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--chi", type=float, default=0.99)
    parser.add_argument("--q", type=float)
    parser.add_argument("--edge-factor", type=float, default=100.0)
    parser.add_argument("--output", type=Path, default=Path("results/dynamics"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    q_value = args.q
    if q_value is None:
        roots = find_ep_roots(args.alpha, args.chi, (1.0e-4, 1.0))
        physical = [item for item in roots if item.status == "physical_root"]
        if not physical:
            raise SystemExit("no physical EP root; pass --q explicitly")
        q_value = float(physical[0].q)

    result = evolve_transition(
        EvolutionConfig(
            args.alpha,
            args.chi,
            q_value,
            edge_factor=args.edge_factor,
            store_trajectory=True,
            crosscheck=True,
        )
    )
    with (args.output / "transition_trajectories.json").open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2, allow_nan=False)
    with (args.output / "transition_trajectories.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["x", "omega_M", "population_211", "population_21m1", "norm"]
        )
        writer.writerows(
            zip(
                result.x,
                result.omega_M,
                result.initial_population,
                result.final_population,
                result.norm,
            )
        )

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(result.x, result.initial_population, label=r"$|b_{211}|^2$")
    axes[0].plot(result.x, result.final_population, label=r"$|b_{21-1}|^2$")
    axes[0].plot(result.x, result.norm, label="total", linestyle="--")
    axes[0].set_xlabel(r"scaled detuning $x$")
    axes[0].set_ylabel("population")
    axes[0].legend()
    names = ["LZ", "Hermitian", "factorized", "non-Hermitian"]
    values = [
        result.survival_lz,
        result.survival_hermitian,
        result.survival_factorized,
        result.survival_nh,
    ]
    axes[1].bar(names, values, color=["#777777", "#377eb8", "#4daf4a", "#e41a1c"])
    axes[1].set_ylabel(r"final $|211\rangle$ survival")
    axes[1].tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(args.output / "mode_trajectories.png", dpi=180)
    plt.close(figure)
    print(
        f"status={result.status} q={q_value:.8g} z={result.landau_zener_z:.6g} "
        f"D={result.effect_abs:.6g}"
    )


if __name__ == "__main__":
    main()

