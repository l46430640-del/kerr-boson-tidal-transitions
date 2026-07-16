# Higher-Tide and Independent Kerr-Kernel Validation

- 状态：`kernel_validation_passed`
- 高阶潮汐失败：`0`
- 独立 kernel 失败：`0`
- 墙钟时间：`552.1 s`

## 数值结论

- 点伴星势重构最大残差：`4.426e-16`
- 奇数 electric multipole 最大相对幅值：`2.836e-16`
- `ell=8/10` 最大尾项：`0.000e+00`
- Gauss/quad 最大差异：`9.745e-10`
- 独立 kernel 最大幅值差：`3.236e-06`
- 独立 kernel 最大相位差：`2.525e-09 rad`

- Schwarzschild `alpha=0.15, 211->31-1` control difference: `3.534e-03`

## Physical thresholds

- `r99/b<=0.07` 且保留 10% kernel 修正：`['a0.200000_21+1_21-1', 'a0.325000_32+2_32+0', 'a0.350000_32+2_32+0', 'a0.375000_32+2_32+0']`
- 修正超过三倍高阶潮汐包络：`['a0.200000_21+1_21-1', 'a0.325000_32+2_32+0', 'a0.350000_32+2_32+0', 'a0.375000_32+2_32+0']`
- `alpha=0.35, 322->320` 精确多极修正：`0.066%`
- Newtonian 与 1PN 均有 10% depletion 改变：`True`

The validated quantities use local adiabatic Kerr quadrupolar tides; covariant higher multipoles are represented by the reported conservative envelope.