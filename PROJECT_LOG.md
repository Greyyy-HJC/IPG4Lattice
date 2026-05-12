# PROJECT_LOG.md

Append-only development history for `IPG4Lattice`.

## 2026-05-11

- Added `AGENTS.md` to define durable agent workflow, coding boundaries, and documentation maintenance rules for this repository.
- Added `PROJECT_LOG.md` as an append-only project history file per initialization protocol in `INIT.md`.
- Updated `README.md` to document agent-facing files, default `.venv` setup, and `environment.yml` as optional machine-specific template.
- Updated `SPEC.md` to align runtime/environment guidance with repository-root `.venv` default and clarified `environment.yml` role.
- Removed `.codex` from git tracking (`git rm --cached .codex`) while keeping it ignored locally.
- Fixed `scripts/qprop_static.py`: re-enabled `point_quark_corr_z.append` so jackknife/plots match the still-computed point propagator (empty list had caused `ValueError` in `jk_ls_avg`).
- Added `tests/validate_S16T16_cg_ipg_gauge.sh` to run `scripts/validate_cg_ipg.py` on `ensemble/S16T16_cg` vs `ensemble/S16T16_cg_ipg` with fixed paths and tolerances.
