# Kerr Boson-Cloud Tidal Transitions

This repository provides Python implementations and numerical datasets for
tidal transitions between scalar quasibound states around Kerr black holes.
It includes hydrogenic reference calculations, non-Hermitian two-level
dynamics, Kerr mode solvers, covariant tidal kernels, multipole corrections,
and convergence checks.

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
