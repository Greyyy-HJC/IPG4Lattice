# IPG4Lattice

Utilities and notes for implementing the Integrated Polyakov gauge (IPG) on top of already Coulomb-gauge-fixed lattice ensembles, and validating that the residual IPG fixing does not spoil the Coulomb-gauge quark-propagator signal.

## Current scope

- `v1` targets the `S16T16` Coulomb-gauge-fixed ensemble in `ensemble/S16T16_cg/gauge`.
- The source paper is [`doc/1204.0716v2.pdf`](/home/jinchen/git/anl/IPG4Lattice/doc/1204.0716v2.pdf), especially Appendix A.
- The detailed project spec is in [`SPEC.md`](/home/jinchen/git/anl/IPG4Lattice/SPEC.md).

## Environment

The repository now includes two environment descriptions:

- [`requirements.txt`](/home/jinchen/git/anl/IPG4Lattice/requirements.txt): Python packages used by the current scripts.
- [`environment.yml`](/home/jinchen/git/anl/IPG4Lattice/environment.yml): a conda environment template including MPI/CUDA-related base packages.

In practice, `PyQUDA` and CUDA/MPI compatibility are machine-dependent, so `environment.yml` should be treated as a starting point rather than a guaranteed one-command setup on every system.

## Repository layout

- `doc/1204.0716v2.pdf`: reference paper.
- `scripts/quark_prop.py`: existing quark-propagator script used as the validation baseline.
- `scripts/ipg_utils.py`: shared IPG and validation helpers.
- `scripts/ipg_fix.py`: batch CG -> CG+IPG fixing script.
- `scripts/validate_cg_ipg.py`: CG vs CG+IPG propagator comparison.
- `ensemble/`: local input and generated ensembles. The entire directory is ignored by git.

## What the code does

The current implementation follows Appendix A as:

1. Read a Coulomb-gauge-fixed NERSC gauge field.
2. Compute the spatial average of the temporal links on each time slice.
3. Project the averaged temporal links back to `SU(3)` using a Cabibbo-Marinari style subgroup projection.
4. Build the integrated Polyakov matrix `P = ∏_t u(t)`.
5. Compute a constant target matrix `C = P^(1/T)` using `scipy.linalg.logm/expm`.
6. Construct the residual time-only gauge transform `g(t)`.
7. Apply the residual transform to produce a new `CG+IPG` gauge field.
8. Validate that the current spatial-line quark propagator agrees with the original CG result.

The implementation also records the important Appendix A interpretation used in practice: the quantity entering `P^(1/T)` must be the matrix product of projected temporal links, not a traced scalar.

## Typical workflow

Generate CG+IPG gauges:

```bash
python scripts/ipg_fix.py \
  --input-dir ensemble/S16T16_cg/gauge \
  --output-dir ensemble/S16T16_cg_ipg/gauge \
  --glob 'wilson_b6.cg.1e-08.*' \
  --save-transform \
  --verify-readback
```

Validate the propagator against the original CG ensemble:

```bash
python scripts/validate_cg_ipg.py \
  --cg-dir ensemble/S16T16_cg/gauge \
  --ipg-dir ensemble/S16T16_cg_ipg/gauge \
  --glob 'wilson_b6.cg.1e-08.*' \
  --n-conf 1 \
  --tol 1e-8
```

## Output conventions

- Input CG ensemble: `ensemble/S16T16_cg/gauge/wilson_b6.cg.*`
- Output CG+IPG ensemble: `ensemble/S16T16_cg_ipg/gauge/wilson_b6.cg.ipg.*`
- Saved residual transform: `ensemble/S16T16_cg_ipg/ipg_transform/*.npy`
- The intended output layout is an output root containing sibling `gauge/` and `ipg_transform/` directories.

Generated ensemble files are intentionally ignored by git.

## Notes

- The current validation observable is exactly the same slice used in `scripts/quark_prop.py`: `data[0,0,:,0]`.
- `--cfg-start/--cfg-stop` filter by the trailing configuration index, so combine them with `--glob` if multiple solver tolerances exist for the same configuration number.
- QUDA, CUDA, MPI, and GPU driver compatibility are not fully portable; adjust [`environment.yml`](/home/jinchen/git/anl/IPG4Lattice/environment.yml) to match your local system.
