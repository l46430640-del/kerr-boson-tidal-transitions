# Hydrogenic Baseline Report

## Conventions

- Units: `G=c=hbar=M=1`; rates are reported as `M Gamma`.
- Real spectrum: `M omega_R = alpha[1-alpha^2/8-17alpha^4/128+m chi alpha^5/12]`.
- Imaginary spectrum: `M Gamma = 2 r_+ C_21m (m Omega_H-omega_R) alpha^9`, with `C_21m=[1-chi^2+(chi m-2 r_+ alpha)^2]/48`.
- Factor-of-two convention: `Gamma` is the amplitude rate used on the Hamiltonian diagonal as `+i Gamma`; the occupation rate is `2 Gamma`.
- Transition: circular, equatorial, corotating `|211> -> |21-1>` with `l*=m*=2`.
- `q_EP` is obtained from all sign changes on 512 logarithmic samples over `10^-6<=q<=10^3`, then refined with `scipy.optimize.brentq` using the complete radial pieces.

## Symbolic checks

- `radial_21` = `sqrt(6)*x*exp(-x/2)/12`
- `normalization` = `1`
- `radial_inner_infinite` = `30`
- `angular_gaunt` = `-sqrt(30)/(10*sqrt(pi))`
- `y22_equator` = `sqrt(30)/(8*sqrt(pi))`
- `angular_prefactor` = `-3/10`
- `total_far_tide_coefficient` = `-9`

The symbolic calculation gives `M Omega_res = chi alpha^6 / 12`, `|K|=9`, and a Detweiler rate scaling `Gamma ~ alpha^9/M`, not `alpha^10/M`.

## Benchmark EP results

| alpha | spin policy | chi | status | q_EP | q_EP far | R_res/(4 r0) |
|---:|---|---:|---|---:|---:|---:|
| 0.05 | saturation_approx | 0.19801980 | no_root | - | - | - |
| 0.05 | saturation_numeric | 0.19795899 | not_superradiant | - | - | - |
| 0.05 | high_0.9 | 0.90000000 | physical_root | 0.64819113 | 0.64819113 | 664.194 |
| 0.05 | high_0.99 | 0.99000000 | physical_root | 0.5381053 | 0.5381053 | 609.106 |
| 0.10 | saturation_approx | 0.38461538 | no_root | - | - | - |
| 0.10 | saturation_numeric | 0.38416693 | not_superradiant | - | - | - |
| 0.10 | high_0.9 | 0.90000000 | physical_root | 0.8584865 | 0.8584865 | 172.83 |
| 0.10 | high_0.99 | 0.99000000 | physical_root | 0.63736715 | 0.63736715 | 155.484 |
| 0.20 | saturation_approx | 0.68965517 | no_root | - | - | - |
| 0.20 | saturation_numeric | 0.68705489 | not_superradiant | - | - | - |
| 0.20 | high_0.9 | 0.90000000 | root_q_gt_1 | 2.782545 | 2.782545 | 54.7563 |
| 0.20 | high_0.99 | 0.99000000 | root_q_gt_1 | 1.2045564 | 1.2045564 | 42.9224 |
| 0.30 | saturation_approx | 0.88235294 | no_root | - | - | - |
| 0.30 | saturation_numeric | 0.87724159 | not_superradiant | - | - | - |
| 0.30 | high_0.9 | 0.90000000 | no_root | - | - | - |
| 0.30 | high_0.99 | 0.99000000 | root_q_gt_1 | 4.1601818 | 4.1601818 | 25.3289 |

## Go/No-Go summary

- **GO (high-spin baseline):** a physical `q_EP<=1` exists; the smallest benchmark root is `q_EP=0.538105` at `alpha=0.05`, `chi=0.990`.
- No saturation-spin EP root was found in the diagnostic `q<=10^3` interval.

## Reference timescale hierarchy

At `alpha=0.1`, `chi=0.99`, `q=0.63736715`, and `M=10 M_sun`:

- `T_split` = `6.06060606e+06 M` = `2.98514603e+02 s`
- `T_orb` = `7.61598219e+07 M` = `3.75124513e+03 s`
- `T_abs` = `2.62052926e+10 M` = `1.29073932e+06 s`
- `T_eta` = `4.19378335e+10 M` = `2.06564419e+06 s`
- `T_grow` = `1.04938342e+11 M` = `5.16872852e+06 s`
- `T_sweep` = `2.12731429e+12 M` = `1.04780673e+08 s`
- `T_width` = `2.15817828e+14 M` = `1.06300876e+10 s`
- `T_GW` = `7.46701903e+17 M` = `3.67787346e+13 s`
- `z = 2.57306840e+03`
- `R_res/(4 r0) = 1.55484130e+02`

## Approximation warning

These roots are exact only within the stated hydrogenic, corrected-Detweiler, quadrupolar model. The dominant systematic is the small-alpha Detweiler rate; subleading relativistic wavefunctions, higher tidal multipoles, additional levels, eccentricity/inclination, cloud self-gravity, and orbital backreaction are not included. A numerical root is therefore not yet a publication-level robustness claim.
