#!/usr/bin/env python3
"""Check E_eff(p) against sqrt(M(|k|)^2 + |k|^2) from cached staggered CG+IPG data.

Reads ``qprop_mom_{ensemble}.npz`` (time-direction effective masses) and
``qprop_M_{ensemble}.npz`` / legacy ``qprop_M_staggered_{ensemble}.npz`` (Eq.25 shell M).
Outputs PDF diagnostics only.
"""

from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gvar as gv
import matplotlib.pyplot as plt
import numpy as np

from lametlat.preprocess.read_raw import pt2_to_meff
from lametlat.utils.plot_settings import *
from lametlat.utils.resampling import jackknife, jk_ls_avg

from scripts.qprop_utils import MomentumGrid, plot_dir, resolve_data_path

DEFAULT_ENSEMBLE = "S24T24_cg_ipg"
DEFAULT_MASS = -0.038888
DEFAULT_GAMMA = "I"
DEFAULT_T_MIN = 6
DEFAULT_T_MAX = 14


def mom_data_path(ensemble: str) -> str:
    return os.path.join(ROOT, f"artifacts/data/qprop_mom_{ensemble}.npz")


def safe_pt2_to_meff(pt2_array):
    try:
        return pt2_to_meff(pt2_array, boundary="periodic")
    except ZeroDivisionError:
        n_t = len(pt2_array)
        return gv.gvar(np.full(n_t, np.nan), np.full(n_t, np.nan))


def k_spatial_from_momentum_list(momentum_list: np.ndarray, latt_size: np.ndarray) -> np.ndarray:
    grid = MomentumGrid.from_arrays(
        np.column_stack([momentum_list, np.zeros(len(momentum_list))]),
        latt_size,
    )
    return grid.k_spatial


def plateau_mean(meff_gv, t_min: int, t_max: int) -> tuple[float, float]:
    means = gv.mean(meff_gv)
    sdevs = gv.sdev(meff_gv)
    sl = slice(t_min, t_max + 1)
    m = means[sl]
    s = sdevs[sl]
    mask = np.isfinite(m) & np.isfinite(s) & (s > 0)
    if not np.any(mask):
        return np.nan, np.nan
    w = 1.0 / s[mask] ** 2
    val = float(np.sum(w * m[mask]) / np.sum(w))
    err = float(1.0 / np.sqrt(np.sum(w)))
    return val, err


def match_M_at_k(
    k_target: float,
    k_shells: np.ndarray,
    M_mean: np.ndarray,
    M_cfg: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, str]:
    if k_target <= 1e-8:
        return np.nan, np.nan, "p=0 (no Eq.25 shell)"
    idx = int(np.argmin(np.abs(k_shells - k_target)))
    m_jk = jk_ls_avg(jackknife(np.real(M_cfg[:, idx])))
    return float(m_jk.mean), float(m_jk.sdev), str(labels[idx])


def load_mass_index(masses: np.ndarray, bare_mass: float) -> int:
    return int(np.argmin(np.abs(masses - bare_mass)))


def run_dispersion_check(
    ensemble: str,
    *,
    gamma: str,
    bare_mass: float,
    t_min: int,
    t_max: int,
    corr_mode: str,
    M_cache: str | None,
) -> None:
    mom_path = mom_data_path(ensemble)
    M_path = M_cache if M_cache else resolve_data_path(ensemble)
    if not os.path.isfile(mom_path):
        raise FileNotFoundError(mom_path)
    if not os.path.isfile(M_path):
        raise FileNotFoundError(M_path)

    mom = np.load(mom_path, allow_pickle=True)
    Mdata = np.load(M_path, allow_pickle=True)

    if gamma not in mom.files:
        raise KeyError(f"Gamma {gamma!r} not in {mom_path}; keys={mom.files}")

    latt_size = np.asarray(mom["latt_size"], dtype=np.float64)
    momentum_list = np.asarray(mom["momentum_list"], dtype=np.float64)
    mom_labels = [str(x) for x in mom["momentum_label"]]
    k_pts = k_spatial_from_momentum_list(momentum_list, latt_size)

    masses = np.asarray(Mdata["masses"], dtype=np.float64)
    imass = load_mass_index(masses, bare_mass)
    k_shells = np.asarray(Mdata["kbin_k_spatial"], dtype=np.float64)
    M_mean = np.asarray(Mdata["kbin_M_mean"][imass], dtype=np.float64)
    M_cfg = np.asarray(Mdata["kbin_M_cfg"][imass], dtype=np.float64)
    shell_labels = Mdata["kbin_label"]

    corr = np.asarray(mom[gamma])
    if corr_mode == "norm":
        corr_use = np.abs(corr)
    elif corr_mode == "re":
        corr_use = np.real(corr)
    else:
        raise ValueError(f"Unknown corr_mode {corr_mode!r}")

    rows = []
    E_eff_list, E_eff_err_list = [], []
    E_pred_list, E_pred_err_list = [], []
    k_plot_list = []

    for idx, label in enumerate(mom_labels):
        jk_avg = jk_ls_avg(jackknife(corr_use[:, idx, :]))
        meff = safe_pt2_to_meff(jk_avg)
        e_eff, e_err = plateau_mean(meff, t_min, t_max)
        k_val = float(k_pts[idx])
        m_val, m_err, shell_lab = match_M_at_k(k_val, k_shells, M_mean, M_cfg, shell_labels)
        if np.isfinite(m_val):
            e_pred = float(np.sqrt(m_val**2 + k_val**2))
            e_pred_err = float(
                np.sqrt((m_val * m_err) ** 2) / max(e_pred**2, 1e-30)
            ) if e_pred > 0 else np.nan
        else:
            e_pred, e_pred_err = np.nan, np.nan
        delta = e_eff - e_pred if np.isfinite(e_pred) else np.nan
        rows.append((label, k_val, m_val, m_err, shell_lab, e_eff, e_err, e_pred, e_pred_err, delta))
        if k_val > 1e-8 and np.isfinite(e_eff) and np.isfinite(e_pred):
            k_plot_list.append(k_val)
            E_eff_list.append(e_eff)
            E_eff_err_list.append(e_err)
            E_pred_list.append(e_pred)
            E_pred_err_list.append(e_pred_err)

    print(f"ensemble={ensemble} gamma={gamma} mass={bare_mass} corr={corr_mode}")
    print(f"M cache: {M_path}")
    print(f"{'p':<12} {'|k|':>8} {'M(|k|)':>10} {'E_eff':>10} {'E_pred':>10} {'Δ':>10} shell")
    for row in rows:
        label, k_val, m_val, _, shell_lab, e_eff, _, e_pred, _, delta = row
        print(
            f"{label:<12} {k_val:8.4f} {m_val:10.4f} {e_eff:10.4f} "
            f"{e_pred:10.4f} {delta:10.4f}  {shell_lab}"
        )

    os.makedirs(plot_dir(), exist_ok=True)
    out_pdf = os.path.join(plot_dir(), f"qprop_dispersion_{ensemble}_{gamma}.pdf")

    k_plot = np.asarray(k_plot_list)
    E_eff = np.asarray(E_eff_list)
    E_eff_err = np.asarray(E_eff_err_list)
    E_pred = np.asarray(E_pred_list)
    E_pred_err = np.asarray(E_pred_err_list)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.errorbar(k_plot, E_eff, yerr=E_eff_err, fmt="o", label=r"$E_{\mathrm{eff}}$", **errorb)
    ax.errorbar(
        k_plot,
        E_pred,
        yerr=E_pred_err,
        fmt="s",
        label=r"$\sqrt{M^2+|k|^2}$",
        **errorb,
    )
    ax.set_xlabel(r"$|k| = \sqrt{\sum_i \sin^2 p_i}$", **fs_p)
    ax.set_ylabel(r"Energy (lattice units)", **fs_p)
    ax.legend(**fs_small_p)

    ax2 = axes[1]
    ax2.errorbar(E_pred, E_eff, xerr=E_pred_err, yerr=E_eff_err, fmt="o", **errorb)
    lim_lo = np.nanmin([E_pred, E_eff])
    lim_hi = np.nanmax([E_pred, E_eff])
    if np.isfinite(lim_lo) and np.isfinite(lim_hi):
        ax2.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.8, alpha=0.5)
    ax2.set_xlabel(r"$\sqrt{M^2+|k|^2}$", **fs_p)
    ax2.set_ylabel(r"$E_{\mathrm{eff}}$", **fs_p)

    fig.suptitle(f"{ensemble} / {gamma} / mass={bare_mass}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_pdf, transparent=True)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispersion check: meff vs sqrt(M^2+|k|^2)")
    parser.add_argument("--ensemble", default=DEFAULT_ENSEMBLE)
    parser.add_argument("--gamma", default=DEFAULT_GAMMA)
    parser.add_argument("--mass", type=float, default=DEFAULT_MASS)
    parser.add_argument("--t-min", type=int, default=DEFAULT_T_MIN)
    parser.add_argument("--t-max", type=int, default=DEFAULT_T_MAX)
    parser.add_argument("--corr", choices=("norm", "re"), default="norm")
    parser.add_argument("--M-cache", default=None, help="Override path to qprop_M npz")
    args = parser.parse_args()

    run_dispersion_check(
        args.ensemble,
        gamma=args.gamma,
        bare_mass=args.mass,
        t_min=args.t_min,
        t_max=args.t_max,
        corr_mode=args.corr,
        M_cache=args.M_cache,
    )


if __name__ == "__main__":
    main()
