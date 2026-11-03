# Kerr Boson-Cloud Tidal Calculations

Numerical code and data for scalar quasibound states on the 211, 322 and
433 Kerr saturation branches, their tidal couplings, and finite-interval
resonance crossings. Physical scalar currents connect the cloud's tidal
torque and horizon absorption to weak orbital feedback and spectral drift.
Units are G=c=hbar=M=1, with M the initial black-hole mass.

## Run

Use Python 3.11 or newer. Install the pinned dependencies with
`python -m pip install -r requirements-lock.txt`. The commands below run
directly from this directory and do not need an editable package install.

```sh
python reproduce.py
python reproduce.py --components --multipoles
python reproduce.py --crossings
python reproduce.py --kernel-point a0.200000_21+1_21-1
python reproduce.py --scalar-currents --stress-overlaps --spectral-response
python reproduce.py --torque --feedback
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
mass-ratio searches, endpoint tests and population-transfer diagnostic. This is
more expensive. Each command uses one numerical thread and writes to
`output/reproduction.json`; no full mode scan is launched by default.

The kernel command reconstructs the quadrupolar and matched E2+B3/E4
couplings from the archived mode coefficients, including both directions.
Core mode solvers, normalization, metric reconstruction and propagation
routines are in `src/boson_ep`. `calculations.py` supplies the specific
factorization, interpolation and finite-interval comparison functions.

The current command integrates number, energy and angular momentum on
spacelike ingoing Kerr slices for the four crossing modes. The stress
command rebuilds all ten overlap samples from the mode expansions and
real tidal metric. The spectral command resolves both complex-frequency
derivatives using continued fractions and checks the saturated width
against the differentiated horizon balance. These commands verify the
saved intermediate inputs independently.

The torque command propagates the prescribed crossings, reconstructs the
physical stress torque and angular-current residual, and calculates the
continuous torque bounds including the local 1.5PN orbital derivative.
The feedback command repeats eighteen coupled evolutions: zero-cloud
limits, both coupling choices, tolerance refinements, separate torque and
absorption effects, and two larger-cloud examples. It reads the saved
overlaps and spectral derivatives. Run the current, stress and spectral
commands as well to check the entire chain from mode coefficients. `feedback.py` contains
these numerical functions; `feedback_checks.py` compares the independently
recalculated quantities with the distributed results.

## Physical Feedback

At initial cloud fraction f_c=1e-12, the coupled survivals are:

| Initial mode | Kerr couplings | Hydrogenic couplings | Difference |
| --- | ---: | ---: | ---: |
| 211 | 0.4141549131 | 0.3226483994 | 0.0915065137 |
| 322 | 0.4392076691 | 0.3008915071 | 0.1383161620 |

The physical mass normalization uses the initial mode energy. The
hydrogenic comparison replaces the couplings and uses their conjugate
transfer torque with the same initial number per cloud mass. It retains
the Kerr frequencies and their absorption response.

The evolution includes the leading tidal torque and resums the secular
effect of linear complex-frequency and horizon-flux changes. Mode
profiles, tidal couplings and orbital background parameters stay at their
initial values. Absorption redistributes mass and spin between cloud and
hole inside the companion orbit. Cloud self-gravity, additional modes and
a complete binary history are outside this calculation.

Small orbital torque alone does not ensure a small survival change. In
the 211 example at f_c=1e-10, accumulated damping of the initially
saturated mode lowers survival to 0.3846492242 although the relative chirp
correction stays below 2e-6. The f_c=1e-12 runs establish finite examples
with small feedback; the continuous torque bounds refer to the prescribed
crossings in the retained model.

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
- `data/scalar_currents.json`, `data/stress_overlaps.json` and
  `data/scalar_torque.json` give the physical normalizations, stress and
  boundary overlaps, and prescribed-orbit torque/current comparison.
- `data/spectral_response.json` contains the frequency derivatives and
  their horizon-balance check; `data/orbital_feedback.json` contains all
  eighteen coupled runs. The physical torque is positive when the cloud
  gains angular momentum; its opposite acts on the orbit.

The older `continuous_feedback_bound` fields in `data/crossings.json`
and the `feedback_bound` function describe a population-transfer
diagnostic. The physical stress torque and its continuous bounds are
computed by `--torque`. The integrated current residual measures the
angular-momentum-weighted two-mode error, not the norm of all omitted modes.

## Integrity and License

`manifest.json` identifies the contents by SHA-256. This distribution contains
numerical source, tests and data only. The software is MIT licensed; see
`LICENSE`. No external paper source, manuscript, correspondence or workspace
history is required to run these commands.
