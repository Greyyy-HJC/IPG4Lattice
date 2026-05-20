# %%
"""Staggered M(|k|) scan on CG+IPG ensembles (wall source + FFT + Eq. 25).

Physics pipeline is documented in ``scripts/qprop_utils.py``.
Edit parameters below or run::

    python scripts/qprop_M.py --ensemble S24T24_cg_ipg --n-conf 50
    python scripts/qprop_M.py --ensemble S24T24_cg_ipg --replot-only
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from pyquda import init
from pyquda_utils import core, io
from tqdm.auto import tqdm

from lametlat.utils.resampling import jackknife, jk_ls_avg

from scripts.qprop_utils import (
    P4_SPREAD_WARN,
    MomentumGrid,
    build_momentum_list,
    collect_mass_curve,
    data_path,
    default_temporal_momenta,
    filter_k_bins,
    gauge_path,
    greens_from_wall_fft,
    lattice_size_for_ensemble,
    load_a_fm,
    replot_from_npz,
    scalar_ab_from_greens,
    staggered_eta_ops,
    summarize_shell_analysis,
    write_all_plots,
)

# --- run parameters ---
ensemble = "S24T24_cg_ipg"
mass_ls = [-0.08, -0.06, -0.038888, -0.02]
reference_mass = -0.038888
N_conf = 50
exclude_k_latt: tuple[float, ...] = ()
max_mode_fraction = 0.25
tol = 1e-8
maxiter = 10000
Nc = 3


def run_mass_scan(
    ensemble: str,
    mass_ls: list[float],
    n_conf: int,
    *,
    max_mode_fraction: float,
    exclude_k_latt: Sequence[float],
    reference_mass: float,
    a_fm_override: float | None,
) -> None:
    cache_dir = os.path.join(ROOT, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    init([1, 1, 1, 1], resource_path=cache_dir)

    latt_size = lattice_size_for_ensemble(ensemble)
    latt_info = core.LatticeInfo(latt_size, -1, 1.0)
    is_root = latt_info.mpi_rank == 0
    eta_ops = staggered_eta_ops(latt_info)

    t_momenta = default_temporal_momenta(latt_size)
    momentum_array = build_momentum_list(
        latt_size, t_momenta, max_mode_fraction=max_mode_fraction
    )
    grid = MomentumGrid.from_arrays(momentum_array, np.asarray(latt_size))

    if is_root:
        print(
            f"ensemble={ensemble}, masses={mass_ls}, n_conf={n_conf}, "
            f"n_mom={len(momentum_array)}, wall_fft, max_mode_fraction={max_mode_fraction}"
        )

    scan_results = []
    for bare_mass in mass_ls:
        dirac = core.getStaggered(latt_info, bare_mass, tol, maxiter)
        as_cfg, bm_cfg = [], []
        for cfg in tqdm(range(n_conf), desc=f"mass {bare_mass:.6f}", disable=not is_root):
            gauge = io.readNERSCGauge(gauge_path(ensemble, cfg))
            with dirac.useGauge(gauge):
                cfg_greens = greens_from_wall_fft(latt_info, dirac, grid, eta_ops)
            if is_root:
                a_val, b_val = scalar_ab_from_greens(cfg_greens, grid, nc=Nc)
                as_cfg.append(a_val)
                bm_cfg.append(b_val)

        if not is_root:
            continue

        as_arr = np.asarray(as_cfg, dtype=np.float64)
        bm_arr = np.asarray(bm_cfg, dtype=np.float64)
        summary = summarize_shell_analysis(
            as_arr, bm_arr, grid, max_mode_fraction=max_mode_fraction
        )
        k_spatial, m_paper_cfg, kbin_labels = collect_mass_curve(
            as_arr, bm_arr, grid, max_mode_fraction=max_mode_fraction
        )
        m_jk = jk_ls_avg(jackknife(np.real(m_paper_cfg)))
        m_mean = np.asarray([float(x.mean) for x in m_jk], dtype=np.float64)
        m_sdev = np.asarray([float(x.sdev) for x in m_jk], dtype=np.float64)

        if exclude_k_latt:
            k_plot, m_mean_plot, m_sdev_plot = filter_k_bins(
                k_spatial, m_mean[None, :], m_sdev[None, :], exclude_k_latt=exclude_k_latt
            )
            m_mean_plot = m_mean_plot[0]
            m_sdev_plot = m_sdev_plot[0]
            keep = np.ones(k_spatial.shape[0], dtype=bool)
            for k_val in exclude_k_latt:
                keep &= np.abs(k_spatial - k_val) > 1e-4
            m_cfg_plot = np.real(m_paper_cfg)[:, keep]
            label_plot = kbin_labels[keep]
            k_plot = k_spatial[keep]
        else:
            k_plot, m_mean_plot, m_sdev_plot = k_spatial, m_mean, m_sdev
            m_cfg_plot = np.real(m_paper_cfg)
            label_plot = kbin_labels

        scan_results.append(
            dict(
                mass=bare_mass,
                As=as_arr,
                Bm=bm_arr,
                kbin_k_spatial=k_plot,
                kbin_M_cfg=m_cfg_plot,
                kbin_M_mean=m_mean_plot,
                kbin_M_sdev=m_sdev_plot,
                kbin_label=label_plot,
            )
        )
        med_as = float(np.nanmedian(summary["As_p4_spread"]))
        med_bm = float(np.nanmedian(summary["Bm_p4_spread"]))
        print(
            f"mass={bare_mass:+.6f}: median p4 spread As={med_as:.2f}, Bm={med_bm:.2f}, "
            f"shells={len(k_spatial)}"
        )
        if med_as > P4_SPREAD_WARN or med_bm > P4_SPREAD_WARN:
            print(
                f"  WARNING: p4 spread exceeds {P4_SPREAD_WARN:.0%}; "
                "Eq. (25) p4 average may be unreliable."
            )

    if not is_root:
        sys.exit(0)

    mass_arr = np.asarray([item["mass"] for item in scan_results], dtype=np.float64)
    a_fm = a_fm_override if a_fm_override is not None else load_a_fm(ensemble)

    os.makedirs(os.path.join(ROOT, "artifacts/data"), exist_ok=True)
    save_kwargs = dict(
        ensemble=np.asarray(ensemble),
        masses=mass_arr,
        reference_mass=np.asarray(reference_mass),
        momentum_list=grid.momentum_list,
        latt_size=grid.latt_size,
        kbin_k_spatial=scan_results[0]["kbin_k_spatial"],
        kbin_M_mean=np.stack([item["kbin_M_mean"] for item in scan_results]),
        kbin_M_sdev=np.stack([item["kbin_M_sdev"] for item in scan_results]),
        kbin_M_cfg=np.stack([item["kbin_M_cfg"] for item in scan_results]),
        kbin_label=scan_results[0]["kbin_label"],
        exclude_k_latt=np.asarray(exclude_k_latt),
        max_mode_fraction=np.asarray(max_mode_fraction),
        temporal_momenta=np.asarray(t_momenta),
        n_conf=np.asarray(n_conf),
        As=np.stack([item["As"] for item in scan_results]),
        Bm=np.stack([item["Bm"] for item in scan_results]),
    )
    if a_fm is not None:
        save_kwargs["a_fm"] = np.asarray(a_fm)
    out_data = data_path(ensemble)
    np.savez(out_data, **save_kwargs)
    print(f"Wrote {out_data}")

    write_all_plots(
        ensemble,
        mass_arr,
        grid,
        scan_results,
        reference_mass=reference_mass,
        a_fm=a_fm,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Staggered M(|k|) from CG+IPG gauges")
    parser.add_argument("--ensemble", default=None)
    parser.add_argument("--n-conf", type=int, default=None)
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Re-draw PDFs from artifacts/data/qprop_M_{ensemble}.npz",
    )
    args = parser.parse_args()
    if args.ensemble is not None:
        ensemble = args.ensemble
    if args.n_conf is not None:
        N_conf = args.n_conf

    if args.replot_only:
        replot_from_npz(ensemble, reference_mass=reference_mass)
        sys.exit(0)

    run_mass_scan(
        ensemble,
        list(mass_ls),
        N_conf,
        max_mode_fraction=max_mode_fraction,
        exclude_k_latt=exclude_k_latt,
        reference_mass=reference_mass,
        a_fm_override=None,
    )

# %%
