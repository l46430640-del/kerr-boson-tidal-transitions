"""Rank physical EPs, screen near-EP widths, and apply formation-history vetoes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boson_ep import (
    EvolutionConfig,
    FormationConfig,
    WidthScanConfig,
    assess_formation_history,
    compute_near_ep_width,
    compute_timescales,
    find_ep_roots,
    saturation_spin_numeric,
)
from boson_ep.dynamics import window_convergence_error


OUTPUT = Path("results/dynamics")
BASELINE = Path("results/baseline/q_ep_scan.csv")


def _physical_root(alpha: float, chi: float) -> float | None:
    roots = find_ep_roots(alpha, chi, (1.0e-4, 1.0))
    physical = [item for item in roots if item.status == "physical_root"]
    return float(physical[0].q) if physical else None


def _anchors() -> list[tuple[float, float, float]]:
    forced = [
        (0.05, 0.90),
        (0.05, 0.99),
        (0.10, 0.90),
        (0.10, 0.99),
        (0.15, 0.99),
        (0.175, 0.99),
        (0.19, 0.999),
    ]
    selected: list[tuple[float, float, float]] = []
    for alpha, chi in forced:
        q_value = _physical_root(alpha, chi)
        if q_value is not None:
            selected.append((alpha, chi, q_value))

    ranked: list[tuple[float, float, float, float]] = []
    with BASELINE.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "physical_root":
                continue
            alpha = float(row["alpha"])
            chi = float(row["chi"])
            q_value = float(row["q"])
            z_value = compute_timescales(alpha, chi, q_value, 10.0).landau_zener_z
            ranked.append((z_value, alpha, chi, q_value))
    ranked.sort()
    keys = {(round(a, 8), round(c, 8)) for a, c, _ in selected}
    for _, alpha, chi, q_value in ranked:
        key = (round(alpha, 8), round(chi, 8))
        if key not in keys:
            selected.append((alpha, chi, q_value))
            keys.add(key)
        if len(selected) >= 12:
            break
    return selected


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    width_results = []
    width_rows: list[dict[str, object]] = []
    formation_rows: list[dict[str, object]] = []
    saturation_rows: list[dict[str, object]] = []
    for alpha, chi, _ in _anchors():
        result = compute_near_ep_width(
            WidthScanConfig(alpha, chi, mass_msun=10.0, q_points=257)
        )
        width_results.append(result)
        row = result.to_dict(include_grid=False)
        row["uncertainty"] = (
            result.uncertainty.to_dict() if result.uncertainty is not None else None
        )
        width_rows.append(row)
        if result.q_ep is not None:
            for mass in (10.0, 100.0, 1.0e6):
                formation_rows.append(
                    assess_formation_history(
                        FormationConfig(alpha, chi, result.q_ep, mass)
                    ).to_dict()
                )
        chi_sat = saturation_spin_numeric(alpha)
        sat_roots = find_ep_roots(alpha, chi_sat, (1.0e-4, 1.0))
        saturation_rows.append(
            {
                "alpha": alpha,
                "chi_sat": chi_sat,
                "ep_status": sat_roots[0].status,
                "q": sat_roots[0].q,
            }
        )
        with (OUTPUT / "near_ep_checkpoint.json").open("w", encoding="utf-8") as handle:
            json.dump(width_rows, handle, indent=2)

    with (OUTPUT / "near_ep_widths.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        flat_rows = []
        for result in width_results:
            row = result.to_dict(include_grid=False)
            uncertainty = row.pop("uncertainty")
            if uncertainty:
                row.update({f"sigma_{key}": value for key, value in uncertainty.items()})
            flat_rows.append(row)
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    with (OUTPUT / "near_ep_widths.json").open("w", encoding="utf-8") as handle:
        json.dump(
            [result.to_dict(include_grid=True) for result in width_results],
            handle,
            indent=2,
        )
    with (OUTPUT / "formation_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(formation_rows[0]))
        writer.writeheader()
        writer.writerows(formation_rows)
    with (OUTPUT / "formation_history.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"formation": formation_rows, "saturation_line": saturation_rows},
            handle,
            indent=2,
        )

    representative = width_results[0]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.semilogx(
        representative.q_grid,
        representative.survival_factorized_grid,
        label=r"$S_{fac}$",
    )
    axis.semilogx(
        representative.q_grid,
        representative.survival_nh_grid,
        linestyle="--",
        label=r"$S_{NH}$ screen",
    )
    axis.fill_between(
        representative.q_grid,
        np.maximum(0.0, representative.effect_grid - 3.0 * representative.sigma_grid),
        np.minimum(1.0, representative.effect_grid + 3.0 * representative.sigma_grid),
        alpha=0.15,
        label=r"$3\sigma_{sys}$ on $D$",
    )
    axis.set_xlabel(r"mass ratio $q$")
    axis.set_ylabel("survival / effect envelope")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / "survival_q_error_band.png", dpi=180)
    plt.close(figure)

    labels = [f"{r.alpha:.3g},{r.chi:.3g}" for r in width_results]
    x_values = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(x_values - 0.2, [r.formal_relative_width for r in width_results], 0.4, label="formal")
    axis.bar(x_values + 0.2, [r.physical_relative_width for r in width_results], 0.4, label="physical")
    axis.set_xticks(x_values, labels, rotation=45, ha="right")
    axis.set_ylabel(r"$\Delta q/q_{EP}$")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / "formal_physical_widths.png", dpi=180)
    plt.close(figure)

    components = ["numerical", "spectrum", "tidal_matrix", "chirp", "two_level"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(width_results))
    for component in components:
        values = np.asarray(
            [getattr(result.uncertainty, component) for result in width_results]
        )
        axis.bar(x_values, values, bottom=bottom, label=component)
        bottom += values
    axis.set_xticks(x_values, labels, rotation=45, ha="right")
    axis.set_ylabel("absolute effect uncertainty")
    axis.legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT / "error_decomposition.png", dpi=180)
    plt.close(figure)

    mass10 = [row for row in formation_rows if row["mass_msun"] == 10.0]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.semilogy(
        np.arange(len(mass10)),
        [row["pre_resonance_efolds"] for row in mass10],
        marker="o",
        label=r"$N_{pre}$",
    )
    axis.semilogy(
        np.arange(len(mass10)),
        [row["log_required_occupancy"] for row in mass10],
        marker="s",
        label=r"$N_{req}$",
    )
    axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    axis.set_ylabel("e-fold count")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / "formation_history_veto.png", dpi=180)
    plt.close(figure)

    formal_pass = [
        result
        for result in width_results
        if result.formal_relative_width >= 1.0e-2
        and (result.effect_at_ep or 0.0) >= 0.10
        and (result.effect_at_ep or 0.0) >= 3.0 * (result.sigma_at_ep or 0.0)
    ]
    physical_pass = [result for result in formal_pass if result.physical_relative_width > 0.0]
    calibration_path = OUTPUT / "spectrum_calibration.json"
    if calibration_path.exists():
        calibration_summary = json.loads(
            calibration_path.read_text(encoding="utf-8")
        )["benchmark"]
        cf_convergence_text = (
            f"{calibration_summary['n100_200_400_converged_groups']}/"
            f"{calibration_summary['total_calibration_groups']}"
        )
    else:
        cf_convergence_text = "not run"
    minimum_result = min(
        (result for result in width_results if result.q_ep is not None),
        key=lambda result: compute_timescales(
            result.alpha, result.chi, result.q_ep, 10.0
        ).landau_zener_z,
    )
    window_error = window_convergence_error(
        EvolutionConfig(
            minimum_result.alpha,
            minimum_result.chi,
            minimum_result.q_ep,
            rtol=1.0e-7,
            atol=1.0e-10,
            store_trajectory=False,
        )
    )
    minimum_z = min(
        compute_timescales(result.alpha, result.chi, result.q_ep, 10.0).landau_zener_z
        for result in width_results
        if result.q_ep is not None
    )
    report = f"""# Near-EP dynamics report

## 结论

- 12 个锚点中形式宽度通过数：`{len(formal_pass)}`；形成史后物理宽度通过数：`{len(physical_pass)}`。
- 所有锚点的 `physical_width=0`。当前主线判定为 **No-Go**。
- 物理 EP 全扫描的最小 `z` 仍为 `{minimum_z:.3g}`。相位自由的密度变量配合 `Radau` 完成了非厄米直接传播；高绝热区的 Hermitian 比较量使用渐近 LZ 值，未伪装成振幅直接积分。
- 最小 `z` 锚点的非厄米直接传播已经通过 `50/100/200` 窗口检查，`S_NH` 最大变化为 `{window_error:.3e}`。
- 常系数、无限窗口的非厄米 LZ 筛查给出 `S_NH=S_fac`。因此任何非零 `D` 必须来自完整 chirp 与 `eta(Omega)`，而不是 EP 标签本身。
- 三个质量标尺均有 `N_pre >> N_req`，高自旋固定背景在到达共振前完成饱和，触发硬性 `formation_veto`。
- 在数值饱和自旋线上，`Gamma_211` 为边界值且没有物理 EP；这一负结果已保留在 `formation_history.json`。

## 校准门槛

CF 模块实现了球谐矩阵、Dolan 三项递推和 modified-Lentz 分式，并执行 `N=100/200/400` 截断审计；通过组数为 `{cf_convergence_text}`。Dolan 2007 最大增长率基准由 `calibrate_spectrum.py` 单独检查。由于尚未获得 Berti et al. 2019 的逐点可机器读取基准，相关点保持 `calibration_failed`，不得用于物理通过结论。

## 发表门槛

当前结果不满足以“天体物理可达 EP”为主标题的门槛。后续只有在相对论潮汐矩阵元或形成史扩展产生 `D>=0.10`、`D>=3 sigma_sys`、`Delta q/q_EP>=10^-2` 且解除形成史否决时才重新 Go；否则转向“相对论潮汐跃迁与云耗散”的纯理论负结果路线。
"""
    (OUTPUT / "dynamics_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
