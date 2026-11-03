# Kerr Boson-Cloud Tidal Transitions

This repository provides Python implementations and numerical datasets for
tidal transitions between scalar quasibound states around Kerr black holes.
It includes hydrogenic reference calculations, non-Hermitian two-level
dynamics, Kerr mode solvers, covariant tidal kernels, multipole corrections,
and convergence checks.

## Current calculation

The self-contained [computation](computation/README.md) directory contains
the current calculation and numerical data. It includes the Hamiltonian
projection, both directed complex couplings, matched electric and magnetic
tidal fields, and finite-interval dissipative crossings. Physical scalar
currents supply the tidal torque and horizon balance for coupled weak-cloud
examples with the leading secular response to absorption.

```sh
cd computation
python -m pip install -r requirements-lock.txt
python reproduce.py --components --multipoles --crossings --tests
python reproduce.py --scalar-currents --stress-overlaps --spectral-response --torque --feedback --tests
```

The [v0.3.0 release](https://github.com/l46430640-del/kerr-boson-tidal-transitions/releases/tag/v0.3.0)
provides the same numerical distribution as a standalone ZIP. Its manifest
records the hashes of all distributed inputs and source files. The
computation README specifies the phase conventions and additional commands.

## Earlier implementation

The top-level `src/`, `scripts/`, `tests/` and `results/` directories below
retain the earlier implementation. Use `computation/` for the current
Hamiltonian convention and matched-tide results.

## Repository layout

- `src/boson_ep/`: reusable numerical and symbolic routines.
- `scripts/`: command-line entry points for calculations and validation.
- `tests/`: unit and regression tests.
- `results/`: archived tables, machine-readable summaries, and figures.

## Requirements

- Python 3.11 or later
- NumPy
- SciPy
- SymPy
- Matplotlib

Install the package in editable mode:

```powershell
python -m pip install -e .
```

Run the test suite:

```powershell
python -m unittest discover -s tests -v
```

## Hydrogenic calculations

Reproduce the symbolic and numerical reference values:

```powershell
python scripts/reproduce_baseline.py
```

Scan exceptional-point roots and generate the associated timescale tables:

```powershell
python scripts/scan_q_ep.py
python scripts/timescale_table.py
```

## Two-level dynamics

Run a single inspiral-driven transition:

```powershell
python scripts/evolve_transition.py --alpha 0.10 --chi 0.99 --q 0.5
```

Generate the frequency calibration and near-critical parameter scan:

```powershell
python scripts/calibrate_spectrum.py
python scripts/scan_near_ep.py
```

## Relativistic tidal calculations

Run the weak-field comparison:

```powershell
python scripts/audit_weak_field.py
```

Generate or resume the numerical pre-scan checks:

```powershell
python scripts/certify_pre_atlas_v2.py --resume
```

Run the relativistic transition atlas with the archived configuration:

```powershell
python scripts/scan_relativistic_atlas_v2.py `
  --config results/relativistic_tides/v2/certification/frozen_atlas_config.json `
  --certificate results/relativistic_tides/v2/certification/pre_atlas_certificate.json
```

Validate higher companion multipoles and the independent Kerr-kernel path:

```powershell
python scripts/validate_kernel_results.py --resume
```

Use `--quick` for a smoke test, `--overwrite` for a clean recomputation, and
`--no-plots` to skip figure generation where those options are available.

## Archived outputs

- `results/baseline/`: hydrogenic coefficients, root scans, and timescales.
- `results/dynamics/`: transition trajectories, widths, and formation history.
- `results/relativistic_tides/v2/`: Kerr modes, kernels, error budgets, and the
  full parameter scan.
- `results/relativistic_tides/kernel_validation/`: multipole sums, independent
  kernel comparisons, and uncertainty estimates.

## License

The software is distributed under the MIT License. See `LICENSE` for details.
