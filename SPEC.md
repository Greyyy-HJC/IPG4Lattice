# S16T16 Coulomb + Integrated Polyakov Gauge SPEC

## Summary
- Goal: produce a new `CG+IPG` ensemble set from the existing `ensemble/S16T16_cg/gauge/wilson_b6.cg.*` inputs while leaving the original CG-fixed ensemble untouched.
- Runtime: all scripts are expected to run under `conda activate pygpt`.
- Scope v1: only `S16T16_cg`.
- Validation observable: keep the current spatial-line quark propagator slice from [`scripts/quark_prop.py`](/home/jinchen/git/anl/IPG4Lattice/scripts/quark_prop.py), namely `data[0,0,:,0]`.

## Integrated Polyakov Gauge
- Start from a Coulomb-gauge-fixed ensemble. Coulomb gauge is assumed complete before IPG runs.
- For each Euclidean time slice `t`, compute the spatial average of the temporal links:

```text
û(t) = (1 / L^3) Σ_x U4(x, t)
```

- Project `û(t)` back to `u(t) ∈ SU(3)` using Cabibbo-Marinari subgroup cooling / projection.
- Form the integrated Polyakov matrix:

```text
P = ∏_t u(t)
```

- Choose a constant target matrix:

```text
C = P^(1/T)
```

- Fix the residual gauge by setting `g(0) = I` and recursively solving:

```text
g(t + 1) = C† g(t) u(t)
```

  Equivalent form:

```text
g(t) u(t) g†(t + 1) = C
```

- Apply the time-only residual gauge transform:

```text
U'i(x, t) = g(t) Ui(x, t) g†(t)
U'4(x, t) = g(t) U4(x, t) g†(t + 1)
```

- Important inference from Appendix A:
  - Eq. (A4)/(A5) must be interpreted with matrix `P = ∏_t u(t)`, not a traced scalar.
  - Reason: `P^(1/T)` in Eq. (A5) must itself be an `SU(3)` matrix, so the matrix product is the only formula-consistent reading.

## Repository Outputs
- Input ensemble: `ensemble/S16T16_cg/gauge/wilson_b6.cg.*`
- Output ensemble: `ensemble/S16T16_cg_ipg/gauge/wilson_b6.cg.ipg.*`
- Optional residual transform archive: `ensemble/S16T16_cg_ipg/ipg_transform/*.npy`
- Original input directories are read-only by project convention.

## Implementation Layout
- [`scripts/ipg_utils.py`](/home/jinchen/git/anl/IPG4Lattice/scripts/ipg_utils.py)
  - PyQUDA initialization helpers
  - Gauge file discovery and output naming
  - `û(t) -> u(t)` projection
  - Integrated Polyakov matrix construction
  - Residual transform construction and application
  - Current propagator-slice observable extraction
- [`scripts/ipg_fix.py`](/home/jinchen/git/anl/IPG4Lattice/scripts/ipg_fix.py)
  - Batch IPG fixing on existing CG ensembles
  - Write new NERSC gauges to a separate output directory
  - Optional `.npy` archive of the time-only residual transform `g(t)`
- [`scripts/validate_cg_ipg.py`](/home/jinchen/git/anl/IPG4Lattice/scripts/validate_cg_ipg.py)
  - Compare CG and CG+IPG ensembles using the current quark propagator slice
  - Report per-configuration and global max differences
  - Exit non-zero if tolerance is violated

## Script Interfaces
- `python scripts/ipg_fix.py --input-dir ensemble/S16T16_cg/gauge --output-dir ensemble/S16T16_cg_ipg/gauge`
  - Required: `--input-dir`, `--output-dir`
  - Optional: `--glob`, `--cfg-start`, `--cfg-stop`, `--save-transform`, `--overwrite`
- `python scripts/validate_cg_ipg.py --cg-dir ensemble/S16T16_cg/gauge --ipg-dir ensemble/S16T16_cg_ipg/gauge`
  - Required: `--cg-dir`, `--ipg-dir`
  - Optional: `--n-conf`, `--cfg-list`, `--mass`, `--tol`

## Acceptance Criteria
- Numerical construction:
  - Each projected `u(t)` is unitary within tolerance and has `det(u(t)) = 1`.
  - `C` is unitary within tolerance and has `det(C) = 1`.
  - After the residual transform, the projected temporal averages satisfy:

```text
max_t ||u'(t) - u'(0)|| < 1e-10
```

- I/O:
  - Every written `CG+IPG` gauge can be re-read through `pyquda_utils.io.readNERSCGauge`.
  - `ensemble/S16T16_cg` remains unchanged.
- Physics validation:
  - On a sample configuration, the current spatial-line propagator from CG and CG+IPG must agree within solver/noise tolerance.
  - On the production batch, the same observable must agree per configuration within `1e-8` absolute tolerance.

## Notes
- The residual IPG transform is spatially constant on each time slice, so the working expectation is that the CG-sensitive spatial propagator should remain unchanged after IPG.
- `scripts/ipg_utils.py` uses `scipy.linalg.logm/expm` to build the matrix `T`-th root and then projects the result back to `SU(3)` to control numerical drift.
