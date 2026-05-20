"""Staggered M(|k|) utilities for CG+IPG quark propagators (1204.0716-style).

Pipeline (momentum space)
-------------------------
1. **Gauge** — read Coulomb + IPG NERSC configs.
2. **Wall inversion** — one inversion per spatial momentum shell with source phase
   ``exp(-i p·x)`` at ``t=0`` (better SNR than a single point source after IPG).
3. **G_Γ(p)** — contract staggered propagator with taste phases η_Γ, gather ``(t,z,y,x)``,
   4D FFT → complex ``G_Γ(p)`` at each listed 4-momentum.
4. **Dressing coefficients** (scalar Fig. 3 projection, not full Eq. 20 inversion)::

       A_s(p) = (k · Im Γ_spatial) / |k|² / N_c
       B_m(p) = Re G_I(p) / N_c

   where ``Γ_spatial = (Im gX, Im gY, Im gZ)`` and ``k = sin(p)`` on spatial components.
5. **M(|k|)** (Eq. 25) — group momenta by spatial shell (physical branch ``|n_i| ≤ L/4``),
   then per configuration::

       M(shell) = mean_{p4}(B_m) / mean_{p4}(A_s)

   Jackknife over configs → ``M(|k|)`` vs ``|k| = sqrt(Σ sin² p_i)``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import gvar as gv
import matplotlib.pyplot as plt
import numpy as np
from opt_einsum import contract
from pyquda_utils import core, io, source
from pyquda_utils.phase import MomentumPhase

from lametlat.utils.plot_settings import default_plot, errorb, fs_p, fs_small_p
from lametlat.utils.resampling import jackknife, jk_ls_avg

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

GAMMA_NAMES = ("I", "gX", "gY", "gZ", "gT")
HBARC_GEV_FM = 0.1973269804
P4_SPREAD_WARN = 0.30


# --- Momentum grid ---


@dataclass(frozen=True)
class MomentumGrid:
    momentum_list: np.ndarray
    momentum_label: List[str]
    latt_size: np.ndarray
    momentum_angles: np.ndarray
    k_mu: np.ndarray
    k_spatial: np.ndarray

    @classmethod
    def from_arrays(cls, momentum_list: np.ndarray, latt_size: np.ndarray) -> "MomentumGrid":
        momentum_list = np.asarray(momentum_list, dtype=np.float64)
        latt_size = np.asarray(latt_size, dtype=np.float64)
        momentum_angles = 2.0 * np.pi * momentum_list / latt_size
        k_mu = np.sin(momentum_angles)
        k_spatial = np.sqrt(np.sum(k_mu[:, :3] ** 2, axis=1))
        labels = [
            f"({int(px)},{int(py)},{int(pz)},{int(pt)})"
            for px, py, pz, pt in momentum_list
        ]
        return cls(
            momentum_list=momentum_list,
            momentum_label=labels,
            latt_size=latt_size,
            momentum_angles=momentum_angles,
            k_mu=k_mu,
            k_spatial=k_spatial,
        )


def default_temporal_momenta(latt_size: Sequence[int], step: int = 2) -> List[int]:
    """Temporal modes 0, step, ..., T/2 for Eq. (25) p4 averaging."""
    lt = int(latt_size[3])
    return list(range(0, lt // 2 + 1, step))


def signed_modes(extent: int) -> np.ndarray:
    return np.arange(-extent // 2 + 1, extent // 2 + 1, dtype=np.int32)


def shell_label_from_spatial_modes(nx: int, ny: int, nz: int) -> Tuple[int, int, int]:
    return tuple(sorted((abs(int(nx)), abs(int(ny)), abs(int(nz)))))


def near_body_diagonal_with_parity(nx: int, ny: int, nz: int) -> bool:
    nonzero = [int(c) for c in (nx, ny, nz) if int(c) != 0]
    return not nonzero or all(c > 0 for c in nonzero) or all(c < 0 for c in nonzero)


def cylinder_shell_labels(lx: int) -> set[Tuple[int, int, int]]:
    labels: set[Tuple[int, int, int]] = set()
    max_n = lx // 2
    for n in range(max_n + 1):
        labels.add((n, n, n))
        if n < max_n:
            labels.add((n, n, n + 1))
            labels.add((n, n + 1, n + 1))
    return labels


def build_momentum_list(
    latt_size: Sequence[int],
    temporal_momenta: Sequence[int],
    *,
    max_mode_fraction: float = 0.25,
) -> np.ndarray:
    lx, ly, lz, _lt = (int(v) for v in latt_size)
    max_modes = np.asarray(
        [int(np.floor(lx * max_mode_fraction)),
         int(np.floor(ly * max_mode_fraction)),
         int(np.floor(lz * max_mode_fraction))],
        dtype=np.int32,
    )
    cylinder = cylinder_shell_labels(lx)
    temporal = [int(pt) for pt in temporal_momenta]
    spatial_reps: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    for nx in signed_modes(lx):
        for ny in signed_modes(ly):
            for nz in signed_modes(lz):
                if not near_body_diagonal_with_parity(nx, ny, nz):
                    continue
                if np.any(np.abs([nx, ny, nz]) > max_modes):
                    continue
                label = shell_label_from_spatial_modes(nx, ny, nz)
                if label not in cylinder:
                    continue
                spatial = (int(nx), int(ny), int(nz))
                if label not in spatial_reps or spatial < spatial_reps[label]:
                    spatial_reps[label] = spatial
    spatial_list = sorted(spatial_reps.values())
    return np.asarray(
        [[nx, ny, nz, pt] for nx, ny, nz in spatial_list for pt in temporal],
        dtype=np.float64,
    )


def physical_branch_mask(
    momentum_list: np.ndarray,
    latt_size: np.ndarray,
    max_mode_fraction: float = 0.25,
) -> np.ndarray:
    max_mode = np.floor(latt_size * max_mode_fraction).astype(np.int64)
    mom = np.asarray(momentum_list, dtype=np.int64)
    return np.all(np.abs(mom[:, :3]) <= max_mode[None, :3], axis=1)


def orbit_sort_key(spatial: Sequence[int]) -> Tuple[int, int, int]:
    return tuple(sorted((abs(int(spatial[0])), abs(int(spatial[1])), abs(int(spatial[2])))))


def group_spatial_shells(
    grid: MomentumGrid,
    *,
    max_mode_fraction: float = 0.25,
) -> List[Dict[str, object]]:
    branch = physical_branch_mask(grid.momentum_list, grid.latt_size, max_mode_fraction)
    shells: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, spatial in enumerate(grid.momentum_list[:, :3]):
        if not branch[idx]:
            continue
        shells.setdefault(orbit_sort_key(spatial), []).append(idx)
    grouped = []
    for key in sorted(shells):
        idxs = shells[key]
        grouped.append(
            {
                "shell_key": key,
                "shell_label": str(key),
                "indices": idxs,
                "k_spatial": float(np.mean(grid.k_spatial[idxs])),
            }
        )
    return grouped


def temporal_indices_for_shell(grid: MomentumGrid, shell_indices: Iterable[int]) -> List[int]:
    spatial = grid.momentum_list[shell_indices[0], :3]
    pts = sorted({int(grid.momentum_list[i, 3]) for i in shell_indices})
    out = []
    for pt in pts:
        target = np.concatenate([spatial, [pt]])
        for idx in shell_indices:
            if np.all(grid.momentum_list[idx] == target):
                out.append(idx)
                break
    return out


# Fixed 4 spatial × 3 temporal modes for A_s / B_m / M(p) diagnostic plots.
COEFFICIENT_PLOT_SPATIAL: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),
    (2, 2, 2),
    (3, 3, 3),
    (4, 4, 4),
)
COEFFICIENT_PLOT_PT: Tuple[int, ...] = (0, 4, 8)

# Spatial reps for staggered momentum-wall tdir correlators (qprop_mom).
QPROP_MOM_SPATIAL: Tuple[Tuple[int, int, int], ...] = COEFFICIENT_PLOT_SPATIAL


def coefficient_plot_momentum_list() -> np.ndarray:
    """12 four-momenta: Cartesian product of COEFFICIENT_PLOT_SPATIAL × COEFFICIENT_PLOT_PT."""
    return np.asarray(
        [
            [px, py, pz, pt]
            for px, py, pz in COEFFICIENT_PLOT_SPATIAL
            for pt in COEFFICIENT_PLOT_PT
        ],
        dtype=np.float64,
    )


def coefficient_plot_indices(grid: MomentumGrid) -> np.ndarray:
    """Indices into the full momentum grid for the 12-point coefficient/M diagnostic subset."""
    targets = coefficient_plot_momentum_list()
    picks: List[int] = []
    for target in targets:
        shell_key = orbit_sort_key(target[:3])
        pt = int(target[3])
        found = None
        for idx in range(len(grid.momentum_list)):
            row = grid.momentum_list[idx]
            if int(row[3]) != pt:
                continue
            if orbit_sort_key(row[:3]) == shell_key:
                found = idx
                break
        if found is None:
            raise ValueError(
                f"Coefficient plot shell {shell_key} at pt={pt} "
                "not found in momentum_list; rerun measurement with default temporal grid."
            )
        picks.append(found)
    return np.asarray(picks, dtype=np.int64)


def select_coefficient_plot_indices(
    grid: MomentumGrid,
    *,
    max_mode_fraction: float = 0.25,
    max_points: int = 12,
) -> np.ndarray:
    """Return fixed 12-point diagnostic indices (legacy name kept for callers)."""
    del max_mode_fraction, max_points
    return coefficient_plot_indices(grid)


# --- Staggered greens (wall + FFT) ---


def staggered_sink_eta(latt_info, gamma_name: str) -> np.ndarray:
    coord_x = latt_info.coordinate(0)
    coord_y = latt_info.coordinate(1)
    coord_z = latt_info.coordinate(2)
    if gamma_name == "I":
        return np.ones_like(coord_x, dtype=np.float64)
    if gamma_name == "gX":
        return np.ones_like(coord_x, dtype=np.float64)
    if gamma_name == "gY":
        return (-1.0) ** coord_x
    if gamma_name == "gZ":
        return (-1.0) ** (coord_x + coord_y)
    if gamma_name == "gT":
        return (-1.0) ** (coord_x + coord_y + coord_z)
    raise KeyError(f"Unknown gamma insertion {gamma_name!r}")


def staggered_eta_ops(latt_info) -> List[Tuple[str, np.ndarray]]:
    return [(name, staggered_sink_eta(latt_info, name)) for name in GAMMA_NAMES]


def gather_staggered_eta_trace(latt_info, propag, eta_phase: np.ndarray) -> np.ndarray:
    contracted = contract("wtzyxaa,wtzyx->wtzyx", propag.data, eta_phase).get()
    gathered = core.gatherLattice(core.lexico(contracted, [0, 1, 2, 3, 4]), [0, 1, 2, 3])
    return np.asarray(gathered)


def fft_trace_to_momentum_list(
    trace_tzyx: np.ndarray,
    momentum_list: np.ndarray,
    latt_size: Sequence[int],
) -> np.ndarray:
    trace = np.asarray(trace_tzyx)
    volume = np.prod(trace.shape)
    fourier = np.fft.ifftn(trace, axes=(0, 1, 2, 3)) * volume
    lx, ly, lz, lt = (int(v) for v in latt_size)
    out = np.empty(len(momentum_list), dtype=np.complex128)
    for idx, mom in enumerate(momentum_list):
        out[idx] = fourier[int(mom[3]) % lt, int(mom[2]) % lz, int(mom[1]) % ly, int(mom[0]) % lx]
    return out


def spatial_rep_to_indices(grid: MomentumGrid) -> Dict[Tuple[int, int, int], List[int]]:
    out: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, spatial in enumerate(grid.momentum_list[:, :3]):
        key = tuple(int(v) for v in spatial)
        out.setdefault(key, []).append(idx)
    return out


def invert_wall_propagator(dirac, latt_info, source_phase):
    wall_source = source.staggeredPropagator(
        latt_info, "wall", 0, source_phase=np.conj(source_phase)
    )
    return core.invertStaggeredPropagator(dirac, wall_source)


def staggered_wall_corr_t_by_gamma(
    latt_info,
    dirac,
    spatial: Sequence[int],
    eta_ops: Sequence[Tuple[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Time-direction correlators C_Gamma(t, p) from one staggered momentum-wall inversion.

    Source phase ``exp(-ip·x)`` is applied in the inverter; sink momentum ``p`` enters via
    ``mom_phase`` on the spatial sum (same convention as clover ``qprop_mom``).
    """
    p_phase = MomentumPhase(latt_info).getPhases(
        [[int(spatial[0]), int(spatial[1]), int(spatial[2])]]
    )[0]
    propag = invert_wall_propagator(dirac, latt_info, p_phase)
    out: Dict[str, np.ndarray] = {}
    for gamma_name, eta_phase in eta_ops:
        corr_t = contract(
            "wtzyx,wtzyxaa,wtzyx->t",
            p_phase,
            propag.data,
            eta_phase,
        ).get()
        out[gamma_name] = np.asarray(
            core.gatherLattice(corr_t, [0, -1, -1, -1]),
            dtype=np.complex128,
        )
    return out


def greens_from_wall_fft(latt_info, dirac, grid: MomentumGrid, eta_ops) -> Dict[str, np.ndarray]:
    """One wall inversion per spatial rep; FFT extracts all p4 at that momentum."""
    mom_phase = MomentumPhase(latt_info)
    n_mom = len(grid.momentum_list)
    greens = {name: np.zeros(n_mom, dtype=np.complex128) for name, _ in eta_ops}
    for spatial, idx_list in spatial_rep_to_indices(grid).items():
        phase = mom_phase.getPhases([[spatial[0], spatial[1], spatial[2]]])[0]
        propag = invert_wall_propagator(dirac, latt_info, phase)
        sub_mom = grid.momentum_list[idx_list]
        for gamma_name, eta_phase in eta_ops:
            trace = gather_staggered_eta_trace(latt_info, propag, eta_phase)
            greens[gamma_name][idx_list] = fft_trace_to_momentum_list(
                trace, sub_mom, grid.latt_size
            )
    return greens


# --- Coefficients A_s, B_m ---


def scalar_ab_from_greens(
    greens_by_gamma: Mapping[str, np.ndarray],
    grid: MomentumGrid,
    *,
    nc: int = 3,
    k_tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """A_s and B_m from gamma-traced FFT greens (scalar projection)."""
    k_spatial = grid.k_mu[:, :3]
    k2 = np.sum(k_spatial**2, axis=1)
    gamma_imag = np.stack(
        [
            np.imag(greens_by_gamma["gX"]),
            np.imag(greens_by_gamma["gY"]),
            np.imag(greens_by_gamma["gZ"]),
        ],
        axis=-1,
    )
    b_val = np.real(greens_by_gamma["I"]) / nc
    a_val = np.full_like(b_val, np.nan, dtype=np.float64)
    numerator = np.sum(k_spatial * gamma_imag, axis=-1) / nc
    np.divide(numerator, k2, out=a_val, where=k2 > k_tol)
    return a_val, b_val


# --- M(|k|) from Eq. (25) ---


def p4_relative_spread(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    mean_abs = np.mean(np.abs(values))
    if mean_abs == 0:
        return 0.0
    return float((np.max(values) - np.min(values)) / mean_abs)


def shell_p4_spread(values_cfg_p4: np.ndarray) -> float:
    spreads = [p4_relative_spread(values_cfg_p4[cfg]) for cfg in range(values_cfg_p4.shape[0])]
    return float(np.nanmedian(spreads))


def paper_mass_from_dressing(
    as_cfg: np.ndarray,
    bm_cfg: np.ndarray,
    temporal_indices: Sequence[int],
) -> np.ndarray:
    n_conf = as_cfg.shape[0]
    out = np.full(n_conf, np.nan, dtype=np.complex128)
    for cfg in range(n_conf):
        as_slice = as_cfg[cfg, temporal_indices]
        bm_slice = bm_cfg[cfg, temporal_indices]
        as_mean = np.nanmean(as_slice)
        bm_mean = np.nanmean(bm_slice)
        if np.isfinite(as_mean) and abs(as_mean) > 0:
            out[cfg] = bm_mean / as_mean
    return out


def collect_mass_curve(
    as_cfg: np.ndarray,
    bm_cfg: np.ndarray,
    grid: MomentumGrid,
    *,
    max_mode_fraction: float = 0.25,
    k_tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-shell M(|k|): one point per spatial shell (no |k| bin merging)."""
    shells = group_spatial_shells(grid, max_mode_fraction=max_mode_fraction)
    k_vals, m_cfgs, labels = [], [], []
    for shell in shells:
        if shell["k_spatial"] <= k_tol:
            continue
        t_idxs = temporal_indices_for_shell(grid, shell["indices"])
        if not t_idxs:
            continue
        k_vals.append(shell["k_spatial"])
        m_cfgs.append(paper_mass_from_dressing(as_cfg, bm_cfg, t_idxs))
        labels.append(shell["shell_label"])
    if not k_vals:
        return np.array([]), np.empty((as_cfg.shape[0], 0)), np.array([])
    order = np.argsort(k_vals)
    return (
        np.asarray(k_vals, dtype=np.float64)[order],
        np.column_stack([m_cfgs[i] for i in order]),
        np.asarray([labels[i] for i in order], dtype=object),
    )


def summarize_shell_analysis(
    as_cfg: np.ndarray,
    bm_cfg: np.ndarray,
    grid: MomentumGrid,
    *,
    max_mode_fraction: float = 0.25,
    k_tol: float = 1e-12,
) -> Dict[str, np.ndarray]:
    shells = group_spatial_shells(grid, max_mode_fraction=max_mode_fraction)
    records = {
        "shell_label": [],
        "k_spatial": [],
        "As_p4_spread": [],
        "Bm_p4_spread": [],
        "M_paper_real_mean": [],
        "M_paper_real_sdev": [],
    }
    for shell in shells:
        idxs = shell["indices"]
        t_idxs = temporal_indices_for_shell(grid, idxs)
        if not t_idxs or float(np.mean(grid.k_spatial[idxs])) <= k_tol:
            continue
        m_paper = paper_mass_from_dressing(as_cfg, bm_cfg, t_idxs)
        records["shell_label"].append(shell["shell_label"])
        records["k_spatial"].append(shell["k_spatial"])
        records["As_p4_spread"].append(shell_p4_spread(np.real(as_cfg[:, t_idxs])))
        records["Bm_p4_spread"].append(shell_p4_spread(np.real(bm_cfg[:, t_idxs])))
        records["M_paper_real_mean"].append(float(np.nanmean(np.real(m_paper))))
        records["M_paper_real_sdev"].append(
            float(np.nanstd(np.real(m_paper)) / np.sqrt(len(m_paper)))
        )
    return {key: np.asarray(val) for key, val in records.items()}


def filter_k_bins(
    k_spatial: np.ndarray,
    m_mean: np.ndarray,
    m_sdev: np.ndarray,
    *,
    exclude_k_latt: Sequence[float],
    k_atol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = np.ones(k_spatial.shape[0], dtype=bool)
    for k_val in exclude_k_latt:
        keep &= np.abs(k_spatial - k_val) > k_atol
    return k_spatial[keep], m_mean[:, keep], m_sdev[:, keep]


# --- Ensemble paths ---


def gauge_path(ensemble: str, cfg: int) -> str:
    if ensemble == "S16T16":
        return os.path.join(ROOT, f"ensemble/S16T16/wilson_b6.{cfg}")
    if ensemble == "S16T16_cg":
        return os.path.join(ROOT, f"ensemble/S16T16_cg/gauge/wilson_b6.cg.1e-08.{cfg}")
    if ensemble == "S16T16_cg_ipg":
        return os.path.join(ROOT, f"ensemble/S16T16_cg_ipg/gauge/wilson_b6.cg.ipg.1e-08.{cfg}")
    if ensemble == "S24T24_cg_ipg":
        return os.path.join(ROOT, f"ensemble/S24T24_cg_ipg/gauge/wilson_b6.cg.ipg.1e-14.{cfg}")
    if ensemble == "S32T32_cg_ipg":
        return os.path.join(ROOT, f"ensemble/S32T32_cg_ipg/gauge/wilson_b5_95_fixed.{cfg}.ipg")
    raise ValueError(f"Unknown ensemble {ensemble!r}")


def lattice_size_for_ensemble(ensemble: str) -> list[int]:
    if ensemble in {"S16T16", "S16T16_cg", "S16T16_cg_ipg"}:
        return [16, 16, 16, 16]
    if ensemble == "S24T24_cg_ipg":
        return [24, 24, 24, 24]
    if ensemble == "S32T32_cg_ipg":
        return [32, 32, 32, 32]
    raise ValueError(f"Unknown ensemble {ensemble!r}")


def data_path(ensemble: str) -> str:
    return os.path.join(ROOT, f"artifacts/data/qprop_M_{ensemble}.npz")


def legacy_data_path(ensemble: str) -> str:
    return os.path.join(ROOT, f"artifacts/data/qprop_M_staggered_{ensemble}.npz")


def resolve_data_path(ensemble: str) -> str:
    path = data_path(ensemble)
    if os.path.isfile(path):
        return path
    legacy = legacy_data_path(ensemble)
    if os.path.isfile(legacy):
        return legacy
    return path


def plot_dir() -> str:
    return os.path.join(ROOT, "artifacts/plots")


def load_a_fm(ensemble: str) -> Optional[float]:
    path = os.path.join(ROOT, f"artifacts/data/static_potential_scale_{ensemble}.npz")
    if not os.path.isfile(path):
        return None
    return float(np.load(path)["a_fm"])


def k_lattice_to_gev(k_spatial: np.ndarray, a_fm: float) -> np.ndarray:
    return (k_spatial / a_fm) * HBARC_GEV_FM


def m_lattice_to_gev(m_lattice: np.ndarray, a_fm: float) -> np.ndarray:
    return m_lattice * HBARC_GEV_FM / a_fm


# --- Plotting ---


def plot_M_vs_k(k_spatial: np.ndarray, m_cfg: np.ndarray, output_path: str) -> None:
    m_jk = jk_ls_avg(jackknife(np.real(m_cfg)))
    fig, ax = default_plot()
    ax.errorbar(
        k_spatial,
        gv.mean(m_jk),
        yerr=gv.sdev(m_jk),
        **errorb,
        label=r"$M(|k|)$, Eq.~(25)",
    )
    ax.set_xlabel(r"$|k| = \sqrt{\sum_i \sin^2 p_i}$ (lattice units)", **fs_p)
    ax.set_ylabel(r"$M(|k|)$ (lattice units)", **fs_p)
    ax.legend(**fs_small_p)
    plt.tight_layout()
    fig.savefig(output_path, transparent=True)
    plt.close(fig)


def plot_M_mass_scan(
    output_path: str,
    masses: Sequence[float],
    k_spatial: np.ndarray,
    m_mean: np.ndarray,
    m_sdev: np.ndarray,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    fig, ax = default_plot()
    markers = ["o", "s", "^", "v", "D", "P", "X"]
    eb = {key: val for key, val in errorb.items() if key not in ("linestyle", "marker")}
    for idx, mass in enumerate(masses):
        if not np.any(np.isfinite(m_mean[idx])):
            continue
        ax.errorbar(
            k_spatial,
            m_mean[idx],
            yerr=m_sdev[idx],
            marker=markers[idx % len(markers)],
            linestyle="-",
            **eb,
            label=rf"$am_0={mass:.6g}$",
        )
    ax.set_xlabel(
        xlabel if xlabel is not None else r"$|k| = \sqrt{\sum_i \sin^2 p_i}$ (lattice units)",
        **fs_p,
    )
    ax.set_ylabel(ylabel if ylabel is not None else r"$M(|k|)$ (lattice units)", **fs_p)
    ax.legend(**fs_small_p)
    plt.tight_layout()
    fig.savefig(output_path, transparent=True)
    plt.close(fig)


def pointwise_mass_from_dressing(
    as_cfg: np.ndarray,
    bm_cfg: np.ndarray,
    indices: Sequence[int],
    *,
    k_tol: float = 1e-12,
) -> np.ndarray:
    """M(p) = B_m(p) / A_s(p) at each listed momentum index (no p4 shell average)."""
    as_slice = np.asarray(as_cfg[:, indices], dtype=np.complex128)
    bm_slice = np.asarray(bm_cfg[:, indices], dtype=np.complex128)
    out = np.full(as_slice.shape, np.nan, dtype=np.complex128)
    np.divide(bm_slice, as_slice, out=out, where=np.abs(as_slice) > k_tol)
    return out


def plot_coefficient_vs_momentum(
    coeff_name: str,
    values_cfg_mom: np.ndarray,
    grid: MomentumGrid,
    output_path: str,
    *,
    max_mode_fraction: float = 0.25,
    max_points: int = 12,
) -> None:
    """Jackknife plot of A_s or B_m vs selected 4-momenta (12-point diagnostic subset)."""
    del max_mode_fraction, max_points
    plot_idx = coefficient_plot_indices(grid)
    values_re = np.real(values_cfg_mom[:, plot_idx])
    values_im = np.imag(values_cfg_mom[:, plot_idx])
    values_norm = np.abs(values_cfg_mom[:, plot_idx])
    values_re_jk = jk_ls_avg(jackknife(values_re))
    values_im_jk = jk_ls_avg(jackknife(values_im))
    values_norm_jk = jk_ls_avg(jackknife(values_norm))
    labels = [grid.momentum_label[i] for i in plot_idx]
    x_values = np.arange(len(plot_idx))

    fig, ax = default_plot()
    ax.errorbar(
        x_values,
        gv.mean(values_re_jk),
        yerr=gv.sdev(values_re_jk),
        label=r"$\mathrm{Re}$",
        **errorb,
    )
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
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel(r"$(p_x, p_y, p_z, p_t)$", **fs_p)
    ax.set_ylabel(rf"${coeff_name}(p)$", **fs_p)
    ax.legend(**fs_small_p)
    plt.tight_layout()
    fig.savefig(output_path, transparent=True)
    plt.close(fig)


def plot_M_vs_momentum(
    as_cfg: np.ndarray,
    bm_cfg: np.ndarray,
    grid: MomentumGrid,
    output_path: str,
) -> None:
    """Jackknife plot of pointwise M(p) = B_m / A_s on the 12-point diagnostic grid."""
    plot_idx = coefficient_plot_indices(grid)
    m_cfg = pointwise_mass_from_dressing(as_cfg, bm_cfg, plot_idx)
    m_re = np.real(m_cfg)
    m_jk = jk_ls_avg(jackknife(m_re))
    labels = [grid.momentum_label[i] for i in plot_idx]
    x_values = np.arange(len(plot_idx))

    fig, ax = default_plot()
    ax.errorbar(
        x_values,
        gv.mean(m_jk),
        yerr=gv.sdev(m_jk),
        label=r"$M(p)=\mathrm{Re}(B_m/A_s)$",
        **errorb,
    )
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel(r"$(p_x, p_y, p_z, p_t)$", **fs_p)
    ax.set_ylabel(r"$M(p)$ (lattice units)", **fs_p)
    ax.legend(**fs_small_p)
    plt.tight_layout()
    fig.savefig(output_path, transparent=True)
    plt.close(fig)


def write_all_plots(
    ensemble: str,
    masses: np.ndarray,
    grid: MomentumGrid,
    scan_results: Sequence[dict],
    *,
    reference_mass: float,
    a_fm: Optional[float],
) -> None:
    os.makedirs(plot_dir(), exist_ok=True)
    k_ref = scan_results[0]["kbin_k_spatial"]
    m_mean_all = np.stack([item["kbin_M_mean"] for item in scan_results])
    m_sdev_all = np.stack([item["kbin_M_sdev"] for item in scan_results])

    plot_M = os.path.join(plot_dir(), f"qprop_M_{ensemble}.pdf")
    if len(masses) == 1:
        plot_M_vs_k(k_ref, scan_results[0]["kbin_M_cfg"], plot_M)
    else:
        plot_M_mass_scan(plot_M, masses, k_ref, m_mean_all, m_sdev_all)
    print(f"Wrote {plot_M}")

    if a_fm is not None:
        plot_gev = os.path.join(plot_dir(), f"qprop_M_{ensemble}_gev.pdf")
        plot_M_mass_scan(
            plot_gev,
            masses,
            k_lattice_to_gev(k_ref, a_fm),
            m_lattice_to_gev(m_mean_all, a_fm),
            m_lattice_to_gev(m_sdev_all, a_fm),
            xlabel=r"$|p|$ (GeV)",
            ylabel=r"$M(|p|)$ (GeV)",
        )
        print(f"Wrote {plot_gev}")

    ref_idx = int(np.argmin(np.abs(masses - reference_mass)))
    ref = scan_results[ref_idx]
    coeff_kwargs = {
        "max_mode_fraction": float(ref.get("max_mode_fraction", 0.25)),
        "max_points": 12,
    }
    plot_coefficient_vs_momentum(
        "A_s",
        ref["As"],
        grid,
        os.path.join(plot_dir(), f"qprop_M_{ensemble}_As.pdf"),
        **coeff_kwargs,
    )
    plot_coefficient_vs_momentum(
        "B_m",
        ref["Bm"],
        grid,
        os.path.join(plot_dir(), f"qprop_M_{ensemble}_Bm.pdf"),
        **coeff_kwargs,
    )
    plot_M_vs_momentum(
        ref["As"],
        ref["Bm"],
        grid,
        os.path.join(plot_dir(), f"qprop_M_{ensemble}_M.pdf"),
    )
    print(f"Wrote {plot_dir()}/qprop_M_{ensemble}_As.pdf")
    print(f"Wrote {plot_dir()}/qprop_M_{ensemble}_Bm.pdf")
    print(f"Wrote {plot_dir()}/qprop_M_{ensemble}_M.pdf")


def replot_from_npz(
    ensemble: str,
    *,
    reference_mass: float = -0.038888,
) -> None:
    """Re-draw M / As / Bm PDFs from saved npz without PyQUDA."""
    path = resolve_data_path(ensemble)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    grid = MomentumGrid.from_arrays(data["momentum_list"], data["latt_size"])
    masses = data["masses"]
    a_fm = float(data["a_fm"]) if "a_fm" in data else load_a_fm(ensemble)

    scan_results = []
    for i in range(len(masses)):
        k, m_cfg, _ = collect_mass_curve(
            data["As"][i],
            data["Bm"][i],
            grid,
            max_mode_fraction=float(data.get("max_mode_fraction", 0.25)),
        )
        m_jk = jk_ls_avg(jackknife(np.real(m_cfg)))
        scan_results.append(
            {
                "mass": float(masses[i]),
                "As": data["As"][i],
                "Bm": data["Bm"][i],
                "kbin_k_spatial": k,
                "kbin_M_cfg": np.real(m_cfg),
                "kbin_M_mean": np.asarray([float(x.mean) for x in m_jk]),
                "kbin_M_sdev": np.asarray([float(x.sdev) for x in m_jk]),
            }
        )

    write_all_plots(ensemble, masses, grid, scan_results, reference_mass=reference_mass, a_fm=a_fm)
