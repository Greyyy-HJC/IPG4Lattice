# %%
"""Staggered momentum-wall tdir quark propagator on CG+IPG ensembles.

One staggered inversion per spatial momentum; wall source with ``exp(-ip·x)`` phase.
Caches correlators to ``artifacts/data/qprop_mom_{ensemble}.npz`` and plots effective masses.
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

from lametlat.preprocess.read_raw import pt2_to_meff
from lametlat.utils.plot_settings import *
from lametlat.utils.resampling import *

from scripts.qprop_utils import (
    GAMMA_NAMES,
    QPROP_MOM_SPATIAL,
    gauge_path,
    lattice_size_for_ensemble,
    staggered_eta_ops,
    staggered_wall_corr_t_by_gamma,
)

# --- run parameters ---
ensemble = "S16T16_cg_ipg"
N_conf = 50
mass = -0.038888
tol = 1e-8
maxiter = 10000

momentum_list = [list(spatial) for spatial in QPROP_MOM_SPATIAL]
momentum_label = [f"({px},{py},{pz})" for px, py, pz in QPROP_MOM_SPATIAL]
momentum_array = np.asarray(momentum_list, dtype=np.float64)

cache_dir = os.path.join(ROOT, ".cache")
os.makedirs(cache_dir, exist_ok=True)
init([1, 1, 1, 1], resource_path=cache_dir)

latt_size = lattice_size_for_ensemble(ensemble)
latt_info = core.LatticeInfo(latt_size, -1, 1.0)
dirac = core.getStaggered(latt_info, mass, tol, maxiter)
is_root = latt_info.mpi_rank == 0

eta_ops = staggered_eta_ops(latt_info)
gamma_names = list(GAMMA_NAMES) + ["pDotg"]

wall_quark_corr_t_by_gamma = {name: [] for name in gamma_names}

for cfg in tqdm(range(N_conf), desc="Processing configurations", disable=not is_root):
    gauge = io.readNERSCGauge(gauge_path(ensemble, cfg))

    with dirac.useGauge(gauge):
        cfg_corr = {name: [] for name in GAMMA_NAMES}

        for spatial in momentum_list:
            corr_by_gamma = staggered_wall_corr_t_by_gamma(
                latt_info, dirac, spatial, eta_ops
            )
            if is_root:
                for gamma_name in GAMMA_NAMES:
                    cfg_corr[gamma_name].append(corr_by_gamma[gamma_name])

        if is_root:
            cfg_corr["pDotg"] = [
                momentum_array[idx, 0] * cfg_corr["gX"][idx]
                + momentum_array[idx, 1] * cfg_corr["gY"][idx]
                + momentum_array[idx, 2] * cfg_corr["gZ"][idx]
                for idx in range(len(momentum_list))
            ]
            for gamma_name in cfg_corr:
                wall_quark_corr_t_by_gamma[gamma_name].append(
                    np.stack(cfg_corr[gamma_name])
                )

if is_root:
    os.makedirs(os.path.join(ROOT, "artifacts/data"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "artifacts/plots"), exist_ok=True)
    np.savez(
        os.path.join(ROOT, f"artifacts/data/qprop_mom_{ensemble}.npz"),
        **{
            gamma_name: np.asarray(wall_quark_corr_t_by_gamma[gamma_name])
            for gamma_name in wall_quark_corr_t_by_gamma
        },
        momentum_list=momentum_array,
        momentum_label=np.asarray(momentum_label),
        latt_size=np.asarray(latt_size),
        fermion=np.asarray("staggered"),
        bare_mass=np.asarray(mass),
    )
    print(f"Cached to artifacts/data/qprop_mom_{ensemble}.npz")

    def safe_pt2_to_meff(pt2_array, gamma_name, mom_label):
        try:
            return pt2_to_meff(pt2_array, boundary="periodic")
        except ZeroDivisionError:
            print(
                f"[WARN] ZeroDivision in pt2_to_meff for {gamma_name}, p={mom_label}; "
                "filling with NaN."
            )
            n_t = len(pt2_array)
            return gv.gvar(np.full(n_t, np.nan), np.full(n_t, np.nan))

    for gamma_name in wall_quark_corr_t_by_gamma:
        point_quark_corr_t = np.asarray(wall_quark_corr_t_by_gamma[gamma_name])
        point_quark_corr_t_re = np.real(point_quark_corr_t)
        point_quark_corr_t_norm = np.abs(point_quark_corr_t)
        print("max |Im point_quark_corr_t|: ", np.max(np.abs(np.imag(point_quark_corr_t))))
        print("shape of point_quark_corr_t: ", np.shape(point_quark_corr_t))

        point_quark_corr_t_re_jk_avg = jk_ls_avg(jackknife(point_quark_corr_t_re))
        point_quark_corr_t_norm_jk_avg = jk_ls_avg(jackknife(point_quark_corr_t_norm))

        fig_re, ax_re = default_plot()
        fig_norm, ax_norm = default_plot()
        for idx, label in enumerate(momentum_label):
            point_meff_t_re = safe_pt2_to_meff(
                point_quark_corr_t_re_jk_avg[idx], gamma_name, label
            )
            point_meff_t_norm = safe_pt2_to_meff(
                point_quark_corr_t_norm_jk_avg[idx], gamma_name, label
            )

            ax_re.errorbar(
                np.arange(len(point_meff_t_re)),
                gv.mean(point_meff_t_re),
                yerr=gv.sdev(point_meff_t_re),
                label="tdir_" + label,
                **errorb,
            )
            ax_norm.errorbar(
                np.arange(len(point_meff_t_norm)),
                gv.mean(point_meff_t_norm),
                yerr=gv.sdev(point_meff_t_norm),
                label="tdir_" + label,
                **errorb,
            )

        ax_re.legend(ncol=2, **fs_small_p)
        ax_re.set_xlabel(r"$n_{\mathrm{sep}}$", **fs_p)
        ax_re.set_ylabel(r"$m_{\mathrm{eff}}^{\mathrm{Re}}$", **fs_p)
        fig_re.tight_layout()
        fig_re.savefig(
            os.path.join(
                ROOT,
                f"artifacts/plots/qprop_mom_tdir_meff_re_{ensemble}_{gamma_name}.pdf",
            ),
            transparent=True,
        )
        plt.close(fig_re)

        ax_norm.legend(ncol=2, **fs_small_p)
        ax_norm.set_xlabel(r"$n_{\mathrm{sep}}$", **fs_p)
        ax_norm.set_ylabel(r"$m_{\mathrm{eff}}^{\mathrm{Norm}}$", **fs_p)
        fig_norm.tight_layout()
        fig_norm.savefig(
            os.path.join(
                ROOT,
                f"artifacts/plots/qprop_mom_tdir_meff_norm_{ensemble}_{gamma_name}.pdf",
            ),
            transparent=True,
        )
        plt.close(fig_norm)

# %%
