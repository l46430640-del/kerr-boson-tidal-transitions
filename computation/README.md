# Kerr Boson-Cloud Tidal Calculations

Numerical code and data for scalar quasibound states on the 211, 322 and
433 Kerr saturation branches, their tidal couplings, and finite-interval
resonance crossings on a prescribed circular orbit. Units are G=c=hbar=M=1.

## Run

Use Python 3.11 or newer. Install the pinned dependencies with
`python -m pip install -r requirements-lock.txt`. The commands below run
directly from this directory and do not need an editable package install.

```sh
python reproduce.py
python reproduce.py --components --multipoles
python reproduce.py --crossings
python reproduce.py --kernel-point a0.200000_21+1_21-1
python reproduce.py --tests
```

The default command verifies every distributed file and recalculates all
1095 tidal comparisons, retaining 1094 samples after the local 1.5PN
spatial condition. Components are reconstructed from the saved mode
coefficients; the 128 subset amplitudes and eight logarithmic decompositions
are checked. Multipoles use analytic Gaunt integrals of the finite
spheroidal expansions and split the radial integral at the companion.

The crossing command recomputes both Kerr and hydrogenic survivals for
two saved mass ratios with the same finite endpoints and local 1.5PN orbit.
It includes independent directed E2+B3 couplings, leading E4 and the
external electric amplitude. `--optimize` additionally repeats the bounded
mass-ratio searches, endpoint tests and feedback calculation. This is
more expensive. Each command uses one numerical thread and writes to
`output/reproduction.json`; no full mode scan is launched by default.

The kernel command reconstructs the quadrupolar and matched E2+B3/E4
couplings from the archived mode coefficients, including both directions.
Core mode solvers, normalization, metric reconstruction and propagation
routines are in `src/boson_ep`. `calculations.py` supplies the specific
factorization, interpolation and finite-interval comparison functions.

## Data Conventions

- `data/atlas/` contains 45 saturation samples, 75 transition rows and
  120 normalized mode expansions. Four non-prograde transitions have no
  quadrupolar kernel. The coefficient files retain their numerical residuals.
- Complex numbers in JSON use `[real, imaginary]`. The Kerr and
  Kerr-mode/Newtonian columns of `kernels.csv` use the coefficient in
  `i dc/dt`; they have been converted by a factor of i from the earlier
  implementation. Their absolute values are unchanged. The hydrogenic
  reference already uses the Hamiltonian convention.
- `data/directed_couplings.json` and `data/electric_l4.json` contain the
  frequency samples used to reconstruct the two directed couplings.
  `data/crossings.json` stores the finite-domain outcomes and convergence
  comparisons. `data/crossing_seeds.json` supplies the local-search seeds.
- `data/accuracy_coefficients.json` and `data/accuracy.csv` contain the
  six tighter-domain comparisons. Omitted-order scales are expansion
  estimates, not bounds on unknown coefficients. The small geometry
  increment is a comparison within the adopted quadrupolar model.
- `data/components.csv` and `data/hybrids.csv` describe an algebraic
  amplitude decomposition. The components are not separate observables.
  Hybrid values are given as magnitudes; their arbitrary phase is not used.
- `data/multipoles.json` contains the two Newtonian multipole examples
  evaluated by analytic angular integration. The very small terms refer
  to the finite computed spheroidal expansion, not an infinite-tail bound.

The crossing model prescribes the orbit. The supplied cloud-mass conditions
quantify where its neglect of orbital feedback is sufficient; the code does
not infer cloud populations or evolve a complete binary history.

## Integrity and License

`manifest.json` identifies the contents by SHA-256. This distribution contains
numerical source, tests and data only. The software is MIT licensed; see
`LICENSE`. No external paper source, manuscript, correspondence or workspace
history is required to run these commands.
