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
## 2026-05-19

- Added [`scripts/qprop_dressing_utils.py`](scripts/qprop_dressing_utils.py) with shared Eq. 20 dressing extraction, physical-branch shell grouping, and paper Eq. 25 `M(|k|) = mean_p4(Bm)/mean_p4(As)` analysis helpers.
- Restored [`tests/qprop_dressing.py`](tests/qprop_dressing.py) for offline post-processing of cached clover or staggered greens, including p4-spread reports and `M_vs_k` plots.
- Refactored [`scripts/qprop_dressing.py`](scripts/qprop_dressing.py): cache staggered raw greens to `artifacts/data/qprop_greens_staggered_{ensemble}.npz`, drop staggered `M_corr`, extend temporal momenta to `T/2`, average over random point sources, and plot paper-style `M(|k|)` versus per-point `(p,p4)` diagnostics.
- Clover baseline on existing `qprop_greens_S16T16_cg_ipg.npz` confirms moderate p4 spread (~8–17%) and monotonic uncorrected `M(|k|)` from Wilson spatial terms; corrected/clover comparisons remain offline via `tests/qprop_dressing.py`.
- Consolidated paper-style `M(|k|)` plotting to a single figure [`artifacts/plots/qprop_dressing_S16T16_cg_ipg_M_vs_k.pdf`](artifacts/plots/qprop_dressing_S16T16_cg_ipg_M_vs_k.pdf) via [`tests/plot_qprop_M_vs_k.py`](tests/plot_qprop_M_vs_k.py), using 12 dense k-bins from [`qprop_fig3_mass_S16T16_cg_ipg.npz`](artifacts/data/qprop_fig3_mass_S16T16_cg_ipg.npz) (the coarse dressing grid only yields two physical-branch shells).
- Expanded staggered dressing momentum grid to fig3 cylinder-cut shells (25 × 5 p4 = 125 4-momenta) via `build_dressing_momentum_list()` in [`scripts/qprop_dressing_utils.py`](scripts/qprop_dressing_utils.py); reran PyQUDA inversions on 50 configs (~58 s). New `M(|k|)` curve has 23 finite-|k| shells with large staggered p4 spread (median As ~167%, Bm ~303%); plot now written directly from staggered dressing npz in `scripts/qprop_dressing.py`.
- Staggered-only plotting: [`tests/plot_qprop_M_vs_k.py`](tests/plot_qprop_M_vs_k.py) reads `qprop_dressing_*.npz` (no clover fig3 default). Added [`tests/qprop_volume_check.py`](tests/qprop_volume_check.py) for finite-volume / statistics diagnostics.
- Reworked staggered measurement: point source + lattice FFT greens, then Fig.~3-style scalar projection `A=(k·Im Γ)/|k|²`, `B=Re(I)` (not full Eq.~20 complex inversion). Yields 12 |k| bins with low-|k| enhancement (~0.19–0.20) decaying toward a high-|k| plateau (~0.11) on S16T16; p4 spread on `A` still ~70%. Recommend **L=24** as next ensemble (~5× cost vs 16⁴) before jumping to 32⁴.
- Added [`scripts/qprop_dressing_mass_scan.py`](scripts/qprop_dressing_mass_scan.py): staggered bare-mass scan (default 5 masses, FFT + fig3_style) → `artifacts/data/qprop_dressing_mass_scan_S16T16_cg_ipg.npz` and overlay PDF. On 50 cfg: heavier mass gives larger low-|k| M and sensible ordering; `am_0=0` yields no finite points.
- Consolidated dressing workflow into two notebook-style scripts: [`scripts/qprop_M.py`](scripts/qprop_M.py) (shared FFT/fig3 helpers + dense-grid `M(|k|)` and `mass_ls` scan, staggered/clover) and [`scripts/qprop_dressing.py`](scripts/qprop_dressing.py) (coarse 20-point grid, p4 panels, no `M(|k|)` plot). Removed [`scripts/qprop_dressing_utils.py`](scripts/qprop_dressing_utils.py), [`scripts/qprop_dressing_mass_scan.py`](scripts/qprop_dressing_mass_scan.py), and offline tests [`tests/plot_qprop_M_vs_k.py`](tests/plot_qprop_M_vs_k.py), [`tests/qprop_dressing.py`](tests/qprop_dressing.py), [`tests/qprop_volume_check.py`](tests/qprop_volume_check.py). New artifacts: `artifacts/data/qprop_M_{fermion}_{ensemble}.npz`, `artifacts/plots/qprop_M_{fermion}_{ensemble}.pdf`, `artifacts/data/qprop_dressing_{fermion}_{ensemble}.npz`, `artifacts/plots/qprop_dressing_{fermion}_{ensemble}_*.pdf`.

## 2026-05-20

- Extended [`scripts/qprop_M.py`](scripts/qprop_M.py) with `S24T24_cg_ipg` support in `gauge_path()` (`wilson_b6.cg.ipg.1e-14.{cfg}`) and added `lattice_size_for_ensemble()` so runs no longer assume fixed `16^4`.
- Updated defaults for [`scripts/qprop_M.py`](scripts/qprop_M.py) and [`scripts/qprop_dressing.py`](scripts/qprop_dressing.py) to run on `ensemble = "S24T24_cg_ipg"` and use inferred lattice size (`24^4` for this ensemble).
- Ran `/home/jinchen/miniconda3/envs/pygpt/bin/python scripts/qprop_dressing.py` on 50 configs; script completed and wrote:
  - `artifacts/data/qprop_dressing_clover_S24T24_cg_ipg.npz`
  - `artifacts/plots/qprop_dressing_clover_S24T24_cg_ipg_As.pdf`
  - `artifacts/plots/qprop_dressing_clover_S24T24_cg_ipg_Bm.pdf`
  - `artifacts/plots/qprop_dressing_clover_S24T24_cg_ipg_M.pdf`
  - `artifacts/plots/qprop_dressing_clover_S24T24_cg_ipg_M_corr.pdf`
- Ran `/home/jinchen/miniconda3/envs/pygpt/bin/python scripts/qprop_M.py` on 50 configs and 4 staggered masses; script completed and wrote:
  - `artifacts/data/qprop_M_staggered_S24T24_cg_ipg.npz`
  - `artifacts/plots/qprop_M_staggered_S24T24_cg_ipg.pdf`

## 2026-05-20 — Simplify qprop_M pipeline

- Added [`scripts/qprop_utils.py`](scripts/qprop_utils.py): staggered-only wall_fft measurement, `A_s`/`B_m` projection, physical-branch shells, Eq. 25 `M(|k|)`, plotting (`M`, GeV, `As`, `Bm`); no `ylim` on mass-scan plots.
- Slimmed [`scripts/qprop_M.py`](scripts/qprop_M.py) to a thin runner (`--ensemble`, `--n-conf`, `--replot-only`); fixed staggered + wall_fft; `reference_mass` for coefficient diagnostics.
- Removed [`scripts/qprop_dressing.py`](scripts/qprop_dressing.py) and [`scripts/qprop_M_compare.py`](scripts/qprop_M_compare.py); clover / `M_corr` / point-FFT paths dropped from the active workflow.
- New artifact names: `artifacts/data/qprop_M_{ensemble}.npz`, `artifacts/plots/qprop_M_{ensemble}.pdf` (+ `_gev`, `_As`, `_Bm`). `--replot-only` reads legacy `qprop_M_staggered_{ensemble}.npz` when present.
- Verified `--replot-only` on S24T24_cg_ipg from existing 50-config staggered npz.

## 2026-05-20 — Staggered mom propagator + dispersion check

- Refactored [`scripts/qprop_mom.py`](scripts/qprop_mom.py) to **staggered** fermion (`getStaggered`, `staggered_wall_corr_t_by_gamma` in [`scripts/qprop_utils.py`](scripts/qprop_utils.py)); spatial momenta `(0,0,0)`, `(2,2,2)`, `(3,3,3)`, `(4,4,4)` on `S24T24_cg_ipg`.
- Fixed 12-point coefficient/M diagnostic grid in `qprop_utils`: `COEFFICIENT_PLOT_SPATIAL × COEFFICIENT_PLOT_PT` with shell-orbit matching (negative lattice reps); added [`plot_M_vs_momentum`](scripts/qprop_utils.py) → `artifacts/plots/qprop_M_{ensemble}_M.pdf`.
- Added [`scripts/qprop_dispersion_check.py`](scripts/qprop_dispersion_check.py): compares `E_eff` from `qprop_mom` with `√(M²+|k|²)` from `qprop_M` caches (PDF only).

## 2026-05-31

- Added [`interp_gauge/gfix_S16T16_itpg.sh`](interp_gauge/gfix_S16T16_itpg.sh) to run interpolating gauge fixing on numeric `ensemble/S16T16/wilson_b6.<n>` inputs for `GF_EPSILON = 0.5, 0.3, 0.1`, writing distinct outputs under `ensemble/S16T16_itpg`.
