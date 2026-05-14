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
- Added `scripts/qprop_fig3_mass.py` for fresh Fig. 3-style `M(|k|)` mass scans from `S16T16_cg_ipg` gauge files, with lattice-unit plotting and configurable bare clover masses.
- Extended `scripts/qprop_fig3_mass.py` to infer `S32T32` lattice sizes, optionally plot in GeV via `--a-fm`, and restrict analysis to middle-momentum windows with `--min-k-*`, `--max-k-*`, and `--skip-k-bins`.
- Added a default spatial physical-branch mode cut to `scripts/qprop_fig3_mass.py` so `sin(p)`-degenerate doubler shells are not averaged into the Fig. 3-style mass-function bins.
- Trimmed `presentation/progress_report.html` so the HTML progress report omits Fig. 3 / `qprop_fig3_mass` narrative and plot gallery (four test categories in the summary).
- Tracked `scripts/qprop_fig3_mass.py` and the `artifacts/plots/qprop_fig3_mass_*.pdf` figures in git for reproducibility (the progress-report HTML intentionally does not highlight this workflow).

## 2026-05-12

- Moved the progress report from `presentation/progress_report.html` to repository-root `progress_report.html` with `artifacts/plots/...` links so local `file://` PDFs work in Safari.
- Removed the `presentation/` directory and shortened the `README.md` bullet and report header (no extra viewing note).

## 2026-05-14

- Added `tests/static_potential_scale.py` to measure on-axis Wilson loops for `S16T16`/`S16T16_cg_ipg`, fit the static potential, plot the Cornell fit, and estimate `a` from the Sommer `r0` scale.
- Updated `SPEC.md` to require PDF-only analysis figures and prohibit generated PNG figures.
