# 协变绝热 Kerr 潮汐跃迁：实现与快速验证报告

> 日期：2026-07-14  
> 状态：方法原型完成；Gate 1/2 未通过；不得据此宣称协变跃迁率  
> 范围：covariant adiabatic Kerr tide，不是全局二体度规或全相对论 inspiral

## 1. 主线与实现

EP 分支已按形成史与直接演化结果判定为天体物理 No-Go。本仓库的当前主线是
饱和 Kerr 玻色云的协变绝热潮汐跃迁图谱。代码已提供：

- Dolan/Leaver continued fraction 和复对称 spheroidal 分支；
- Kerr 径向/角向本征函数、ODE 残差和 Cannizzaro 型双线性归一化；
- 近视界局部 Frobenius/Hadamard finite-part counterterm；
- Katagiri-Cardoso `l=2,m=-2` ingoing-radiation-gauge 潮汐度规；
- 保留 `exp[-2 i(phi-Omega v)]` 时间导数的 divergence-form `delta Box_g`；
- hydrogenic/Newtonian、Kerr/Newtonian、Kerr/covariant 三层 kernel；
- 饱和自旋、精确共振、有效域、LZ depletion、CSV/JSON 和断点缓存。

公开接口为 `solve_kerr_mode`、`saturation_spin_cf`、
`covariant_tidal_kernel_M`、`relativistic_tidal_eta_M`、
`compute_relativistic_transition`、`scan_relativistic_atlas` 和
`gauge_audit_transition`。旧 `boson_ep` API 保持兼容。

## 2. 快速验证点

运行：

```powershell
python scripts/scan_relativistic_atlas.py --quick --overwrite
```

该命令使用 `N=150` 设置，只验证端到端链，不替代 `N=250/400` 出版网格。

| quantity | `alpha=0.10`, `211->21-1` |
|---|---:|
| `chi_sat` | `0.3841668179` |
| `M Omega_res` | `3.4025e-8` |
| `r_99/b` at `q=1e-4` | `0.01207` |
| `K_H` | `9.0000e3` |
| `K_Kerr+N` | `-3.30e-3-8.7463e3 i` |
| `K_cov` | `6.92e-1-8.7427e3 i` |
| `|K_cov|/|K_H|` | `0.97141` |
| `z(q=1e-4)` | `0.64490` |
| LZ depletion | `0.98261` |

初态的快速模态检查给出 `CF residual=1.73e-13`、
`radial residual=1.48e-9`、`angular residual=5.19e-17`；末态径向残差为
`6.30e-9`。IRG algebraic audit 给出 `trace residual=2.41e-16`、
`l^a h_ab residual=5.04e-16`；connection form 与 divergence form 的逐点
相对差为 `8.14e-11`。

## 3. 弱场失配的定位与修复

原先 `|K_cov|/|K_H|=0.0626` 的常数失配来自三个独立实现错误：

1. `D^4 Psi=2 psi_0` 对应的 Hertz/metric reconstruction 少了因子二；
2. divergence form 遗漏了变分 `1/sqrt(-g)` 产生的
   `-h Box(Phi)/2=-h mu^2 Phi/2`；
3. 模态投影错误地在 null 的 `v=const` 切片上使用 advanced radial phase，且
   方位积分使用了 `pi` 而不是 J-reflected left mode 给出的 `2 pi`。

修复后统一在 Boyer-Lindquist `t=const` 切片投影。独立实现的 Schwarzschild
Regge-Wheeler gauge 与 IRG 在 `alpha=0.05,0.075,0.10` 的 kernel 相对差均
小于 `2e-12`。对应 `|K_IRG|/|K_H|` 为
`0.99289,0.98396,0.97142`，稳定拟合

```text
1 - |K_rel|/|K_H| = (2.85 + O(alpha^2)) alpha^2.
```

这证明常数归一化失配已消失。`alpha=0.075` 的差异为 `1.60%`，严格上没有
满足原先 `<1%` 的门槛，但三点共享稳定的 `O(alpha^2)` 系数，表明它是相对论
弱场修正而不是残余因子错误。详细数据见 `weak_field_gauge_audit.csv`。

紧支撑解析 gauge vector 的 on-resonance Ward identity 仍未实现，因此
`gauge_audit_transition` 继续返回 `gauge_audit_failed`；Schwarzschild IRG/RW
对照通过并不被包装成完整规范审计通过。

## 4. 下一步与门槛

1. 实现 compact-support gauge-vector 生成器，同时以 connection form 和
   divergence form 计算 `delta Box_g`，要求逐点差 `<1e-10`、on-resonance Ward
   residual `<1e-8`。
2. 复现 Schwarzschild `211->31-1` 文献曲线，并解释原定 `<1%` 门槛与测得
   `2.85 alpha^2` 弱场修正之间的差异。
3. 通过后运行 `N={150,250,400}` 和五通道 15 点图谱；失败点不得回退为
   hydrogenic 后继续标为 relativistic。
4. 只有五通道可信、至少一条有效域修正 `>=10%` 且超过 `3 sigma_sys`，或
   depletion/resonance power 改变 `>=10%`，才进入主论文 Go。

当前发表判断：**pending / No claim**。工程链可以继续，但科学上应先修复弱场与
规范校准，再消耗约 12 小时运行完整图谱。
