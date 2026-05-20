#!/usr/bin/env python3
"""Compare staggered vs clover M(|k|) on the same CG+IPG ensemble (offline)."""

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

from lametlat.utils.plot_settings import default_plot, errorb, fs_p, fs_small_p
from lametlat.utils.resampling import jackknife, jk_ls_avg

from scripts.qprop_M import (
    MomentumGrid,
    bare_mass_to_MeV,
    collect_kbin_mass_curve,
    k_lattice_to_gev,
    load_a_fm,
    m_lattice_to_gev,
)


def load_mass_curve(path: str, max_mode_fraction: float = 0.25):
    data = np.load(path, allow_pickle=True)
    grid = MomentumGrid.from_arrays(data["momentum_list"], data["latt_size"])
    masses = data["masses"]
    if "As" in data and "Bm" in data:
        curves = []
        for i in range(len(masses)):
            k, m_cfg, labels = collect_kbin_mass_curve(
                data["As"][i],
                data["Bm"][i],
                grid,
                max_mode_fraction=max_mode_fraction,
                merge_by_k=False,
            )
            curves.append((k, m_cfg, labels))
        return data, masses, curves
    k = data["kbin_k_spatial"]
    m_cfg = data["kbin_M_cfg"]
    labels = data["kbin_label"]
    return data, masses, [(k, m_cfg[i], labels) for i in range(len(masses))]


def main() -> None:
    ensemble = "S24T24_cg_ipg"
    bare_mass = -0.038888
    max_mode_fraction = 0.25

    stag_path = os.path.join(ROOT, f"artifacts/data/qprop_M_staggered_{ensemble}.npz")
    clov_path = os.path.join(ROOT, f"artifacts/data/qprop_M_clover_{ensemble}.npz")
    if not os.path.isfile(stag_path):
        raise FileNotFoundError(f"Missing {stag_path}; run qprop_M.py with fermion=staggered")
    if not os.path.isfile(clov_path):
        raise FileNotFoundError(f"Missing {clov_path}; run qprop_M.py with fermion=clover")

    stag_data, stag_masses, stag_curves = load_mass_curve(stag_path, max_mode_fraction)
    clov_data, clov_masses, clov_curves = load_mass_curve(clov_path, max_mode_fraction)

    stag_i = int(np.argmin(np.abs(stag_masses - bare_mass)))
    clov_i = int(np.argmin(np.abs(clov_masses - bare_mass)))

    k_s, m_s, lab_s = stag_curves[stag_i]
    k_c, m_c, lab_c = clov_curves[clov_i]

    a_fm = float(stag_data["a_fm"]) if "a_fm" in stag_data else load_a_fm(ensemble)
    plot_dir = os.path.join(ROOT, "artifacts/plots")
    os.makedirs(plot_dir, exist_ok=True)

    m_s_jk = jk_ls_avg(jackknife(np.real(m_s)))
    m_c_jk = jk_ls_avg(jackknife(np.real(m_c)))

    fig, ax = default_plot()
    ax.errorbar(
        k_s,
        gv.mean(m_s_jk),
        yerr=gv.sdev(m_s_jk),
        **errorb,
        label=rf"staggered $am_0={stag_masses[stag_i]:.6g}$",
    )
    ax.errorbar(
        k_c,
        gv.mean(m_c_jk),
        yerr=gv.sdev(m_c_jk),
        **errorb,
        label=rf"clover $am_0={clov_masses[clov_i]:.6g}$",
    )
    ax.set_xlabel(r"$|k| = \sqrt{\sum_i \sin^2 p_i}$ (lattice units)", **fs_p)
    ax.set_ylabel(r"$M(|k|)$ (lattice units)", **fs_p)
    ax.legend(**fs_small_p)
    plt.tight_layout()
    out_latt = os.path.join(plot_dir, f"qprop_M_compare_{ensemble}_lattice.pdf")
    fig.savefig(out_latt, transparent=True)
    plt.close(fig)
    print(f"Wrote {out_latt}")

    if a_fm is not None:
        p_s = k_lattice_to_gev(k_s, a_fm)
        p_c = k_lattice_to_gev(k_c, a_fm)
        hbarc = 0.1973269804
        fig, ax = default_plot()
        ax.errorbar(
            p_s,
            gv.mean(m_s_jk) * hbarc / a_fm,
            yerr=gv.sdev(m_s_jk) * hbarc / a_fm,
            **errorb,
            label="staggered",
        )
        ax.errorbar(
            p_c,
            gv.mean(m_c_jk) * hbarc / a_fm,
            yerr=gv.sdev(m_c_jk) * hbarc / a_fm,
            **errorb,
            label="clover",
        )
        ax.set_xlabel(r"$|p|$ (GeV)", **fs_p)
        ax.set_ylabel(r"$M(|p|)$ (GeV)", **fs_p)
        ax.legend(**fs_small_p)
        plt.tight_layout()
        out_gev = os.path.join(plot_dir, f"qprop_M_compare_{ensemble}_gev.pdf")
        fig.savefig(out_gev, transparent=True)
        plt.close(fig)
        print(f"Wrote {out_gev} (a={a_fm:.4f} fm)")

    print("Staggered shells (first 5):", list(lab_s[:5]))
    print("Clover shells (first 5):", list(lab_c[:5]))
    if a_fm is not None:
        print(
            f"Bare mass estimate: staggered {bare_mass_to_MeV(stag_masses[stag_i], a_fm):.1f} MeV, "
            f"clover {bare_mass_to_MeV(clov_masses[clov_i], a_fm):.1f} MeV "
            f"(paper quenched reference ~14 MeV; not Asqtad-tuned)"
        )


if __name__ == "__main__":
    main()
