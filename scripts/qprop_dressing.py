# %%
"""Coarse-grid dressing diagnostics: per-momentum As/Bm/M index plots.

Shared measurement helpers are imported from ``scripts/qprop_M.py``.
Run directly: ``python scripts/qprop_dressing.py``
"""

from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gvar as gv
import matplotlib.pyplot as plt
import numpy as np
from pyquda import init
from pyquda_utils import core, io
from tqdm.auto import tqdm

from lametlat.utils.plot_settings import default_plot, errorb, fs_p, fs_small_p
from lametlat.utils.resampling import jackknife, jk_ls_avg

from scripts.qprop_M import (
    MomentumGrid,
    P4_SPREAD_WARN,
    build_coarse_momentum_list,
    build_dirac,
    clover_gamma_ops,
    default_temporal_momenta,
    scalar_ab_from_greens,
    gauge_path,
    greens_from_fft,
    invert_propagator,
    lattice_size_for_ensemble,
    staggered_eta_ops,
    summarize_shell_analysis,
    wilson_corrected_m,
)

# --- run parameters ---
fermion = "staggered"  # "staggered" | "clover"
bare_mass = -0.038888
ensemble = "S24T24_cg_ipg"
N_conf = 50
spatial_momenta = [[0, 0, 0], [2, 2, 2], [4, 4, 4], [6, 6, 6]]
max_mode_fraction = 0.25
source_position = [0, 0, 0, 0]
tol = 1e-8
maxiter = 10000
xi_0, nu = 1.0, 1.0
csw_r = 1.02868
csw_t = 1.02868
multigrid = None
Nc = 3


def plot_dressing_vs_index(
    dressing: dict[str, np.ndarray],
    grid: MomentumGrid,
    output_dir: str,
    fermion: str,
    ensemble: str,
) -> None:
    x_values = np.arange(len(grid.momentum_label))
    for dressing_name, values in dressing.items():
        if dressing_name == "At":
            continue
        values_re = np.real(values)
        values_im = np.imag(values)
        values_norm = np.abs(values)
        values_re_jk = jk_ls_avg(jackknife(values_re))
        values_im_jk = jk_ls_avg(jackknife(values_im))
        values_norm_jk = jk_ls_avg(jackknife(values_norm))

        fig, ax = default_plot()
        ax.errorbar(x_values, gv.mean(values_re_jk), yerr=gv.sdev(values_re_jk), label=r"$\mathrm{Re}$", **errorb)
        ax.errorbar(
            x_values + 0.15,
            gv.mean(values_im_jk),
            yerr=gv.sdev(values_im_jk),
            label=r"$\mathrm{Im}$",
            **errorb,
        )
        ax.errorbar(
            x_values + 0.30,
            gv.mean(values_norm_jk),
            yerr=gv.sdev(values_norm_jk),
            label=r"$\mathrm{Norm}$",
            **errorb,
        )
        ax.set_xticks(x_values)
        ax.set_xticklabels(grid.momentum_label, rotation=45, ha="right")
        ax.set_xlabel(r"$(p_x, p_y, p_z, p_t)$", **fs_p)
        ax.set_ylabel(rf"${dressing_name}(p)$", **fs_p)
        ax.legend(**fs_small_p)
        plt.tight_layout()
        out = os.path.join(output_dir, f"qprop_dressing_{fermion}_{ensemble}_{dressing_name}.pdf")
        fig.savefig(out, transparent=True)
        plt.close(fig)
        print(f"Wrote {out}")


if __name__ == "__main__":
    cache_dir = os.path.join(ROOT, ".cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    init([1, 1, 1, 1], resource_path=cache_dir)
    latt_size = lattice_size_for_ensemble(ensemble)
    latt_info = core.LatticeInfo(latt_size, -1, xi_0 / nu)
    is_root = latt_info.mpi_rank == 0
    eta_ops = staggered_eta_ops(latt_info) if fermion == "staggered" else None
    gamma_ops = clover_gamma_ops() if fermion == "clover" else None

    temporal_momenta = default_temporal_momenta(latt_size)
    momentum_array = build_coarse_momentum_list(spatial_momenta, temporal_momenta)
    grid = MomentumGrid.from_arrays(momentum_array, np.asarray(latt_size))

    if is_root:
        print(
            f"fermion={fermion}, bare_mass={bare_mass}, "
            f"{len(momentum_array)} 4-momenta "
            f"({len(spatial_momenta)} shells × {len(temporal_momenta)} p4), "
            f"measurement=fft, dressing=scalar_ab"
        )

    dirac = build_dirac(latt_info, fermion, bare_mass, tol, maxiter, xi_0, csw_r, csw_t, multigrid)
    dressing_names = ("As", "Bm", "M", "M_corr") if fermion == "clover" else ("As", "Bm", "M")
    dressing_arrays = {name: [] for name in dressing_names}

    for cfg in tqdm(range(N_conf), desc="Processing configurations", disable=not is_root):
        gauge = io.readNERSCGauge(gauge_path(ensemble, cfg))
        with dirac.useGauge(gauge):
            propag = invert_propagator(dirac, latt_info, fermion, source_position)
            cfg_greens = greens_from_fft(latt_info, propag, fermion, grid, eta_ops, gamma_ops)

        if is_root:
            a_val, b_val = scalar_ab_from_greens(cfg_greens, grid, nc=Nc)
            dressing_arrays["As"].append(a_val)
            dressing_arrays["Bm"].append(b_val)
            with np.errstate(divide="ignore", invalid="ignore"):
                dressing_arrays["M"].append(b_val / a_val)
                if fermion == "clover":
                    dressing_arrays["M_corr"].append(wilson_corrected_m(b_val, a_val, grid))

    if not is_root:
        sys.exit(0)

    dressing = {name: np.asarray(vals) for name, vals in dressing_arrays.items()}
    summary = summarize_shell_analysis(
        dressing["As"],
        dressing["Bm"],
        grid,
        max_mode_fraction=max_mode_fraction,
    )
    med_as = float(np.nanmedian(summary["As_p4_spread"]))
    med_bm = float(np.nanmedian(summary["Bm_p4_spread"]))
    print(
        "Median p4 relative spread over shells: "
        f"As={med_as:.3e}, Bm={med_bm:.3e}"
    )
    if med_as > P4_SPREAD_WARN or med_bm > P4_SPREAD_WARN:
        print(
            f"WARNING: p4 spread exceeds {P4_SPREAD_WARN:.0%}; "
            "Eq. (25) p4 average may be unreliable."
        )
    for i, label in enumerate(summary["shell_label"]):
        print(
            f"  {label} |k|={summary['k_spatial'][i]:.4f} "
            f"M_paper={summary['M_paper_real_mean'][i]:+.4f} "
            f"+/- {summary['M_paper_real_sdev'][i]:.4f}"
        )

    os.makedirs(os.path.join(ROOT, "artifacts/data"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "artifacts/plots"), exist_ok=True)
    data_path = os.path.join(ROOT, f"artifacts/data/qprop_dressing_{fermion}_{ensemble}.npz")
    cache_kwargs = dict(
        fermion=np.asarray(fermion),
        ensemble=np.asarray(ensemble),
        bare_mass=np.asarray(bare_mass),
        momentum_list=grid.momentum_list,
        momentum_label=np.asarray(grid.momentum_label),
        latt_size=grid.latt_size,
        spatial_momenta=np.asarray(spatial_momenta),
        temporal_momenta=np.asarray(temporal_momenta),
        dressing_method=np.asarray("scalar_ab"),
        wilson_spatial_term=grid.wilson_spatial_term,
        As=dressing["As"],
        Bm=dressing["Bm"],
        M=dressing["M"],
    )
    if fermion == "clover":
        cache_kwargs["M_corr"] = dressing["M_corr"]
    np.savez(data_path, **cache_kwargs)
    print(f"Wrote {data_path}")

    plot_dir = os.path.join(ROOT, "artifacts/plots")
    plot_dressing_vs_index(dressing, grid, plot_dir, fermion, ensemble)

# %%
