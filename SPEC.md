# S16T16 Coulomb + Integrated Polyakov Gauge SPEC

## Summary
- Goal: produce a new `CG+IPG` ensemble set from the existing `ensemble/S16T16_cg/gauge/wilson_b6.cg.*` inputs while leaving the original CG-fixed ensemble untouched.
- Runtime: all scripts are expected to run under `conda activate pygpt`.
- Scope v1: only `S16T16_cg`.
- Validation approach: definition-level validation via [`scripts/validate_cg_ipg.py`](/home/jinchen/git/anl/IPG4Lattice/scripts/validate_cg_ipg.py), which rebuilds the Appendix A transform directly from the original CG gauge and checks the written CG+IPG gauge against metrics including `post_spread`, `boundary`, `residual`, `target_dev`, and `reconstruct`. Quark-propagator scripts are kept as supplementary physics sanity checks.

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
  - Definition-level validation: rebuild the Appendix A transform from each original CG gauge and compare against the written CG+IPG gauge
  - Report per-configuration and global-maximum metrics: `post_spread`, `boundary`, `residual`, `target_dev`, `reconstruct`, unitarity/determinant checks, `logm_err`
  - Optional `--repair-spread` mode: rebuild and overwrite failing configurations with up to `--repair-max-iters` IPG passes
  - Exit non-zero if any configuration exceeds the configured tolerances

## Script Interfaces
- `python scripts/ipg_fix.py --input-dir ensemble/S16T16_cg/gauge --output-dir ensemble/S16T16_cg_ipg/gauge`
  - Required: `--input-dir`, `--output-dir`
  - Optional: `--glob`, `--cfg-start`, `--cfg-stop`, `--save-transform`, `--overwrite`, `--projection-method`
- `python scripts/validate_cg_ipg.py --cg-dir ensemble/S16T16_cg/gauge --ipg-dir ensemble/S16T16_cg_ipg/gauge`
  - Required: `--cg-dir`, `--ipg-dir`
  - Optional: `--glob`, `--n-conf`, `--cfg-list`, `--projection-method`
  - Tolerance options: `--spread-tol` (default `1e-10`), `--boundary-tol` (default `1e-10`), `--reconstruct-tol` (default `1e-10`)
  - Repair options: `--repair-spread`, `--repair-max-iters` (default `3`)

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
- Physics validation (definition-level, from `scripts/validate_cg_ipg.py`):
  - `post_spread`: spread of the projected temporal-link averages after IPG (Z_3-aligned to `C`) must be below `--spread-tol`.
  - `boundary`: periodic-closure error of the recursive `g(t)` construction must be below `--boundary-tol`.
  - `residual`: maximum violation of `g(t) u(t) g†(t+1) = C` must be below `--boundary-tol`.
  - `reconstruct`: maximum link-wise difference between the written CG+IPG gauge and the gauge reconstructed from the original CG gauge must be below `--reconstruct-tol`.
  - Default tolerances are `1e-10` for all three thresholds.
  - Quark-propagator agreement checks (temporal propagator, momentum-projected correlator, momentum-space Green's functions) are carried out as supplementary physics sanity checks via `scripts/qprop_static.py`, `scripts/qprop_mom.py`, and `scripts/qprop_greens.py`.

## Cached analysis data

The quark-propagator scripts cache their raw per-configuration results in `artifacts/data/` so that downstream analyses can reuse them without rerunning inversions:

- `artifacts/data/qprop_static_{ensemble}.npz`: wall-source tdir correlator `wall_quark_corr_t` (shape `N_conf × Lt`, real) and point-source zdir correlator `point_quark_corr_z` (shape `N_conf × Lz`, real).
- `artifacts/data/qprop_greens_{ensemble}.npz`: complex Green's function arrays keyed by gamma name (`I`, `gX`, `gY`, `gZ`, `gT`), each of shape `(N_conf, n_mom)`, plus `momentum_list`, `momentum_label`, and `latt_size`.

## Notes
- IPG only constrains the spatially-averaged temporal links `u(t)`. Consequently, only the spatially-averaged time-direction quark propagator (wall source → spatial-sum sink) carries a clean signal after IPG. This motivates the wall-source choice in `qprop_static.py`.
- `scripts/ipg_utils.py` uses `scipy.linalg.logm/expm` to build the matrix `T`-th root and then projects the result back to `SU(3)` to control numerical drift.
