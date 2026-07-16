# Kerr 潮汐跃迁图谱 v2 报告

- Atlas gate: `atlas_certified`
- Input certificate: `pre_atlas_certified`
- Implementation failures: `0`
- Run fingerprint: `27785654437e5ab86f4c8a1ca3e833a2ece78373c000930f77b85ba671471720`

## 输出基数

- `saturation`: 45
- `modes`: 120
- `kernels`: 75
- `audits`: 75
- `error_budgets`: 75
- `phenomenology`: 2475

## 解释

`tidal_expansion_invalid`、`adiabatic_tide_invalid` 和 `no_valid_q` 是物理/有效域状态，不计作实现失败。只有求根、模态、Ward、误差预算、缓存或输出完整性失败才阻止认证。

## 物理门槛

- 有发表有效 q 的 kernel 点：`38`
- 无有效 q 的点：`37`
- 最大有效域 kernel 修正：`2.166756e-01`
- 通过 10% 且 3 sigma_worst 门槛的点：`['a0.200000_21+1_21-1', 'a0.225000_21+1_21-1', 'a0.325000_32+2_32+0', 'a0.350000_32+2_32+0', 'a0.375000_32+2_32+0', 'a0.400000_32+2_32+0', 'a0.425000_32+2_32+0', 'a0.450000_32+2_32+0']`
- 最大 depletion 绝对变化：`1.379874e-01`

实现认证通过后存在超过误差预算的显著相对论修正。
