# %%
"""M(|k|) from FFT greens (scalar A/B projection, Eq. 25 p4 average).

Shared measurement helpers live in this module; ``qprop_dressing.py`` imports them.
Run directly: ``python scripts/qprop_M.py``
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gvar as gv
import matplotlib.pyplot as plt
import numpy as np
from opt_einsum import contract
from pyquda import init
from pyquda_utils import core, gamma, io, source
from tqdm.auto import tqdm

from lametlat.utils.plot_settings import default_plot, errorb, fs_p, fs_small_p
from lametlat.utils.resampling import jackknife, jk_ls_avg

GAMMA_NAMES = ("I", "gX", "gY", "gZ", "gT")


@dataclass(frozen=True)
class MomentumGrid:
    momentum_list: np.ndarray
    momentum_label: List[str]
    latt_size: np.ndarray
    momentum_angles: np.ndarray
    k_mu: np.ndarray
    k_spatial: np.ndarray
    wilson_spatial_term: np.ndarray

    @classmethod
    def from_arrays(cls, momentum_list: np.ndarray, latt_size: np.ndarray) -> "MomentumGrid":
        momentum_list = np.asarray(momentum_list, dtype=np.float64)
        latt_size = np.asarray(latt_size, dtype=np.float64)
        momentum_angles = 2.0 * np.pi * momentum_list / latt_size
        k_mu = np.sin(momentum_angles)
        k_spatial = np.sqrt(np.sum(k_mu[:, :3] ** 2, axis=1))
        wilson_spatial_term = np.sum(1.0 - np.cos(momentum_angles[:, :3]), axis=1)
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
            wilson_spatial_term=wilson_spatial_term,
        )


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


def clover_gamma_ops() -> List[Tuple[str, object]]:
    return [
        ("I", gamma.gamma(0)),
        ("gX", gamma.gamma(1)),
        ("gY", gamma.gamma(2)),
        ("gZ", gamma.gamma(4)),
        ("gT", gamma.gamma(8)),
    ]


def build_coarse_momentum_list(
    spatial_momenta: Sequence[Sequence[int]],
    temporal_momenta: Sequence[int],
) -> np.ndarray:
    return np.asarray(
        [[px, py, pz, pt] for px, py, pz in spatial_momenta for pt in temporal_momenta],
        dtype=np.float64,
    )


def signed_modes(extent: int) -> np.ndarray:
    return np.arange(-extent // 2 + 1, extent // 2 + 1, dtype=np.int32)


def shell_label_from_spatial_modes(nx: int, ny: int, nz: int) -> Tuple[int, int, int]:
    return tuple(sorted((abs(int(nx)), abs(int(ny)), abs(int(nz)))))


def near_body_diagonal_with_parity(nx: int, ny: int, nz: int) -> bool:
    nonzero = [int(component) for component in (nx, ny, nz) if int(component) != 0]
    return not nonzero or all(component > 0 for component in nonzero) or all(
        component < 0 for component in nonzero
    )


def cylinder_shell_labels(lx: int) -> set[Tuple[int, int, int]]:
    labels: set[Tuple[int, int, int]] = set()
    max_n = lx // 2
    for n in range(max_n + 1):
        labels.add((n, n, n))
        if n < max_n:
            labels.add((n, n, n + 1))
            labels.add((n, n + 1, n + 1))
    return labels


def build_dressing_momentum_list(
    latt_size: Sequence[int],
    temporal_momenta: Sequence[int],
    *,
    max_mode_fraction: Optional[float] = None,
    one_rep_per_shell: bool = True,
) -> np.ndarray:
    lx, ly, lz, _lt = (int(v) for v in latt_size)
    if max_mode_fraction is None:
        max_modes = None
    elif 0.0 < max_mode_fraction <= 0.5:
        max_modes = np.asarray(
            [
                int(np.floor(lx * max_mode_fraction)),
                int(np.floor(ly * max_mode_fraction)),
                int(np.floor(lz * max_mode_fraction)),
            ],
            dtype=np.int32,
        )
    else:
        raise ValueError("max_mode_fraction must be in (0, 0.5] or None")

    cylinder = cylinder_shell_labels(lx)
    temporal = [int(pt) for pt in temporal_momenta]
    spatial_reps: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    for nx in signed_modes(lx):
        for ny in signed_modes(ly):
            for nz in signed_modes(lz):
                if not near_body_diagonal_with_parity(nx, ny, nz):
                    continue
                if max_modes is not None and np.any(np.abs([nx, ny, nz]) > max_modes):
                    continue
                label = shell_label_from_spatial_modes(nx, ny, nz)
                if label not in cylinder:
                    continue
                spatial = (int(nx), int(ny), int(nz))
                if one_rep_per_shell:
                    if label not in spatial_reps or spatial < spatial_reps[label]:
                        spatial_reps[label] = spatial

    spatial_list = sorted(spatial_reps.values())
    return np.asarray(
        [[nx, ny, nz, pt] for nx, ny, nz in spatial_list for pt in temporal],
        dtype=np.float64,
    )


def orbit_sort_key(spatial: Sequence[int]) -> Tuple[int, int, int]:
    return tuple(sorted((abs(int(spatial[0])), abs(int(spatial[1])), abs(int(spatial[2])))))


def physical_branch_mask(
    momentum_list: np.ndarray,
    latt_size: np.ndarray,
    max_mode_fraction: float = 0.25,
) -> np.ndarray:
    max_mode = np.floor(latt_size * max_mode_fraction).astype(np.int64)
    mom = np.asarray(momentum_list, dtype=np.int64)
    return np.all(np.abs(mom[:, :3]) <= max_mode[None, :3], axis=1)


def group_spatial_shells(
    grid: MomentumGrid,
    *,
    max_mode_fraction: Optional[float] = 0.25,
) -> List[Dict[str, object]]:
    shells: Dict[Tuple[int, int, int], List[int]] = {}
    if max_mode_fraction is None:
        branch = np.ones(grid.momentum_list.shape[0], dtype=bool)
    else:
        branch = physical_branch_mask(grid.momentum_list, grid.latt_size, max_mode_fraction)
    for idx, spatial in enumerate(grid.momentum_list[:, :3]):
        if not branch[idx]:
            continue
        key = orbit_sort_key(spatial)
        shells.setdefault(key, []).append(idx)

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


def paper_mass_corr_from_dressing(
    as_cfg: np.ndarray,
    bm_cfg: np.ndarray,
    grid: MomentumGrid,
    temporal_indices: Sequence[int],
) -> np.ndarray:
    """Eq. (25) with Wilson spatial term subtracted from Bm before p4 average (clover)."""
    n_conf = as_cfg.shape[0]
    out = np.full(n_conf, np.nan, dtype=np.complex128)
    wilson = grid.wilson_spatial_term[temporal_indices]
    for cfg in range(n_conf):
        as_slice = as_cfg[cfg, temporal_indices]
        bm_slice = bm_cfg[cfg, temporal_indices]
        as_mean = np.nanmean(as_slice)
        bm_mean = np.nanmean(bm_slice - wilson)
        if np.isfinite(as_mean) and abs(as_mean) > 0:
            out[cfg] = bm_mean / as_mean
    return out


def collect_paper_mass_curve(
    as_cfg: np.ndarray,
    bm_cfg: np.ndarray,
    grid: MomentumGrid,
    *,
    max_mode_fraction: Optional[float] = 0.25,
    k_tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def collect_kbin_mass_curve(
    as_cfg: np.ndarray,
    bm_cfg: np.ndarray,
    grid: MomentumGrid,
    *,
    max_mode_fraction: Optional[float] = 0.25,
    k_tol: float = 1e-12,
    merge_by_k: bool = False,
    k_round: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-shell M(|k|) curve; optional legacy merge by round(|k|) (alias-prone)."""
    k_shell, m_cfg, labels = collect_paper_mass_curve(
        as_cfg, bm_cfg, grid, max_mode_fraction=max_mode_fraction, k_tol=k_tol
    )
    if k_shell.size == 0:
        return k_shell, np.empty((as_cfg.shape[0], 0)), np.array([])
    if not merge_by_k:
        return k_shell, m_cfg, labels

    rounded = np.round(k_shell, decimals=k_round)
    k_bins, m_bins, bin_labels = [], [], []
    for value in sorted(np.unique(rounded)):
        idx = np.flatnonzero(rounded == value)
        k_bins.append(float(np.mean(k_shell[idx])))
        m_bins.append(np.nanmean(np.real(m_cfg[:, idx]), axis=1))
        bin_labels.append(",".join(str(labels[i]) for i in idx))
    return (
        np.asarray(k_bins, dtype=np.float64),
        np.column_stack(m_bins),
        np.asarray(bin_labels, dtype=object),
    )


def collect_kbin_mass_corr_curve(
    as_cfg: np.ndarray,
    bm_cfg: np.ndarray,
    grid: MomentumGrid,
    *,
    max_mode_fraction: Optional[float] = 0.25,
    k_tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    shells = group_spatial_shells(grid, max_mode_fraction=max_mode_fraction)
    k_vals, m_cfgs, labels = [], [], []
    for shell in shells:
        if shell["k_spatial"] <= k_tol:
            continue
        t_idxs = temporal_indices_for_shell(grid, shell["indices"])
        if not t_idxs:
            continue
        k_vals.append(shell["k_spatial"])
        m_cfgs.append(paper_mass_corr_from_dressing(as_cfg, bm_cfg, grid, t_idxs))
        labels.append(shell["shell_label"])
    if not k_vals:
        return np.array([]), np.empty((as_cfg.shape[0], 0)), np.array([])
    order = np.argsort(k_vals)
    return (
        np.asarray(k_vals, dtype=np.float64)[order],
        np.column_stack([m_cfgs[i] for i in order]),
        np.asarray([labels[i] for i in order], dtype=object),
    )


def default_temporal_momenta(latt_size: Sequence[int], step: int = 2) -> List[int]:
    """Temporal modes 0, step, ..., T/2 for Eq. (25) p4 averaging."""
    lt = int(latt_size[3])
    return list(range(0, lt // 2 + 1, step))


P4_SPREAD_WARN = 0.30


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


def scalar_ab_from_greens(
    greens_by_gamma: Mapping[str, np.ndarray],
    grid: MomentumGrid,
    *,
    nc: int = 3,
    k_tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Real scalar dressing coefficients A and B from gamma-traced FFT greens.

    A = (k · Im Γ_spatial) / |k|² / N_c,  B = Re(I) / N_c.
    """
    k_spatial = grid.k_mu[:, :3]
    k2 = np.sum(k_spatial**2, axis=1)
    gamma_imag = np.stack(
        [np.imag(greens_by_gamma["gX"]), np.imag(greens_by_gamma["gY"]), np.imag(greens_by_gamma["gZ"])],
        axis=-1,
    )
    b_val = np.real(greens_by_gamma["I"]) / nc
    a_val = np.full_like(b_val, np.nan, dtype=np.float64)
    numerator = np.sum(k_spatial * gamma_imag, axis=-1) / nc
    np.divide(numerator, k2, out=a_val, where=k2 > k_tol)
    return a_val, b_val


def wilson_corrected_m(
    bm: np.ndarray,
    as_val: np.ndarray,
    grid: MomentumGrid,
) -> np.ndarray:
    """Clover mass after subtracting the Wilson spatial term from B."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (bm - grid.wilson_spatial_term) / as_val


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


def gather_staggered_eta_trace(latt_info, propag, eta_phase: np.ndarray) -> np.ndarray:
    contracted = contract("wtzyxaa,wtzyx->wtzyx", propag.data, eta_phase).get()
    gathered = core.gatherLattice(core.lexico(contracted, [0, 1, 2, 3, 4]), [0, 1, 2, 3])
    return np.asarray(gathered)


def gather_clover_gamma_trace(latt_info, propag, gamma_matrix) -> np.ndarray:
    contracted = contract("wtzyxijaa,ji->wtzyx", propag.data, gamma_matrix).get()
    gathered = core.gatherLattice(core.lexico(contracted, [0, 1, 2, 3, 4]), [0, 1, 2, 3])
    return np.asarray(gathered)


def staggered_greens_from_fft(latt_info, propag, eta_ops, grid: MomentumGrid) -> Dict[str, np.ndarray]:
    greens: Dict[str, np.ndarray] = {}
    for gamma_name, eta_phase in eta_ops:
        trace = gather_staggered_eta_trace(latt_info, propag, eta_phase)
        greens[gamma_name] = fft_trace_to_momentum_list(trace, grid.momentum_list, grid.latt_size)
    return greens


def clover_greens_from_fft(latt_info, propag, gamma_ops, grid: MomentumGrid) -> Dict[str, np.ndarray]:
    greens: Dict[str, np.ndarray] = {}
    for gamma_name, gamma_matrix in gamma_ops:
        trace = gather_clover_gamma_trace(latt_info, propag, gamma_matrix)
        greens[gamma_name] = fft_trace_to_momentum_list(trace, grid.momentum_list, grid.latt_size)
    return greens


def build_dirac(latt_info, fermion: str, mass: float, tol: float, maxiter: int, xi_0: float, csw_r, csw_t, multigrid):
    if fermion == "staggered":
        return core.getStaggered(latt_info, mass, tol, maxiter)
    if fermion == "clover":
        return core.getClover(latt_info, mass, tol, maxiter, xi_0, csw_r, csw_t, multigrid)
    raise ValueError(f"Unknown fermion {fermion!r}")


def invert_propagator(dirac, latt_info, fermion: str, source_position: Sequence[int]):
    if fermion == "staggered":
        point_source = source.staggeredPropagator(latt_info, "point", source_position)
        return core.invertStaggeredPropagator(dirac, point_source)
    point_source = source.propagator(latt_info, "point", source_position)
    return core.invertPropagator(dirac, point_source)


def invert_wall_propagator(dirac, latt_info, fermion: str, source_phase):
    if fermion == "staggered":
        wall_source = source.staggeredPropagator(
            latt_info, "wall", 0, source_phase=np.conj(source_phase)
        )
        return core.invertStaggeredPropagator(dirac, wall_source)
    wall_source = source.propagator(latt_info, "wall", 0, source_phase=np.conj(source_phase))
    return core.invertPropagator(dirac, wall_source)


def spatial_rep_to_indices(grid: MomentumGrid) -> Dict[Tuple[int, int, int], List[int]]:
    out: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, spatial in enumerate(grid.momentum_list[:, :3]):
        key = tuple(int(v) for v in spatial)
        out.setdefault(key, []).append(idx)
    return out


def greens_from_wall_fft(
    latt_info,
    dirac,
    fermion: str,
    grid: MomentumGrid,
    eta_ops,
    gamma_ops,
) -> Dict[str, np.ndarray]:
    """One wall inversion per spatial rep; FFT extracts all p4 at that momentum."""
    from pyquda_utils.phase import MomentumPhase

    mom_phase = MomentumPhase(latt_info)
    n_mom = len(grid.momentum_list)
    if fermion == "staggered":
        greens = {name: np.zeros(n_mom, dtype=np.complex128) for name, _ in eta_ops}
    else:
        greens = {name: np.zeros(n_mom, dtype=np.complex128) for name, _ in gamma_ops}

    for spatial, idx_list in spatial_rep_to_indices(grid).items():
        phase = mom_phase.getPhases([[spatial[0], spatial[1], spatial[2]]])[0]
        propag = invert_wall_propagator(dirac, latt_info, fermion, phase)
        sub_mom = grid.momentum_list[idx_list]
        if fermion == "staggered":
            for gamma_name, eta_phase in eta_ops:
                trace = gather_staggered_eta_trace(latt_info, propag, eta_phase)
                vals = fft_trace_to_momentum_list(trace, sub_mom, grid.latt_size)
                greens[gamma_name][idx_list] = vals
        else:
            for gamma_name, gamma_matrix in gamma_ops:
                trace = gather_clover_gamma_trace(latt_info, propag, gamma_matrix)
                vals = fft_trace_to_momentum_list(trace, sub_mom, grid.latt_size)
                greens[gamma_name][idx_list] = vals
    return greens


def greens_from_fft(
    latt_info,
    propag,
    fermion: str,
    grid: MomentumGrid,
    eta_ops,
    gamma_ops,
) -> Dict[str, np.ndarray]:
    if fermion == "staggered":
        return staggered_greens_from_fft(latt_info, propag, eta_ops, grid)
    return clover_greens_from_fft(latt_info, propag, gamma_ops, grid)


def measure_greens(
    latt_info,
    dirac,
    fermion: str,
    grid: MomentumGrid,
    eta_ops,
    gamma_ops,
    *,
    measurement: str,
    source_position: Sequence[int],
):
    if measurement == "wall_fft":
        return greens_from_wall_fft(latt_info, dirac, fermion, grid, eta_ops, gamma_ops)
    if measurement == "fft":
        propag = invert_propagator(dirac, latt_info, fermion, source_position)
        return greens_from_fft(latt_info, propag, fermion, grid, eta_ops, gamma_ops)
    raise ValueError(f"Unknown measurement {measurement!r}; use 'fft' or 'wall_fft'")


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


def summarize_shell_analysis(
    as_cfg: np.ndarray,
    bm_cfg: np.ndarray,
    grid: MomentumGrid,
    *,
    max_mode_fraction: Optional[float] = None,
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
        records["M_paper_real_sdev"].append(float(np.nanstd(np.real(m_paper)) / np.sqrt(len(m_paper))))
    return {key: np.asarray(val) for key, val in records.items()}


def plot_M_vs_k(k_spatial, m_cfg, output_path: str) -> None:
    m_jk = jk_ls_avg(jackknife(np.real(m_cfg)))
    fig, ax = default_plot()
    ax.errorbar(k_spatial, gv.mean(m_jk), yerr=gv.sdev(m_jk), **errorb, label=r"$M(|k|)$, Eq.~(25)")
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
    ylim: Optional[Tuple[float, float]] = None,
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
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(**fs_small_p)
    plt.tight_layout()
    fig.savefig(output_path, transparent=True)
    plt.close(fig)


def k_lattice_to_gev(k_spatial: np.ndarray, a_fm: float, hbarc_gev_fm: float = 0.1973269804) -> np.ndarray:
    """|p| [GeV] from lattice |k| = sqrt(sum sin^2 p_i) and scale a."""
    return (k_spatial / a_fm) * hbarc_gev_fm


def bare_mass_to_MeV(am_bare: float, a_fm: float, hbarc_gev_fm: float = 0.1973269804) -> float:
    """Convert bare lattice am to MeV (valence estimate; not Asqtad-tuned)."""
    return abs(am_bare) * hbarc_gev_fm / a_fm * 1000.0


def m_lattice_to_gev(m_lattice: np.ndarray, a_fm: float, hbarc_gev_fm: float = 0.1973269804) -> np.ndarray:
    """M(|k|) in lattice units to GeV using scale a."""
    return m_lattice * hbarc_gev_fm / a_fm


def load_a_fm(ensemble: str) -> Optional[float]:
    path = os.path.join(ROOT, f"artifacts/data/static_potential_scale_{ensemble}.npz")
    if not os.path.isfile(path):
        return None
    data = np.load(path)
    return float(data["a_fm"])


def replot_mass_scan_from_npz(
    data_path: str,
    *,
    plot_path: Optional[str] = None,
    plot_gev_path: Optional[str] = None,
) -> None:
    """Re-draw M(|k|) PDFs from a saved ``qprop_M_*.npz`` without re-running inversions."""
    data = np.load(data_path, allow_pickle=True)
    fermion = str(data["fermion"])
    ensemble = str(data["ensemble"])
    masses = data["masses"]
    k_ref = data["kbin_k_spatial"]
    m_mean = data["kbin_M_mean"]
    m_sdev = data["kbin_M_sdev"]
    plot_path = plot_path or os.path.join(ROOT, f"artifacts/plots/qprop_M_{fermion}_{ensemble}.pdf")
    plot_gev_path = plot_gev_path or os.path.join(
        ROOT, f"artifacts/plots/qprop_M_{fermion}_{ensemble}_gev.pdf"
    )
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plot_M_mass_scan(plot_path, masses, k_ref, m_mean, m_sdev)
    a_fm = float(data["a_fm"]) if "a_fm" in data else load_a_fm(ensemble)
    if a_fm is not None:
        hbarc = 0.1973269804
        plot_M_mass_scan(
            plot_gev_path,
            masses,
            k_lattice_to_gev(k_ref, a_fm),
            m_lattice_to_gev(m_mean, a_fm),
            m_lattice_to_gev(m_sdev, a_fm),
            xlabel=r"$|p|$ (GeV)",
            ylabel=r"$M(|p|)$ (GeV)",
        )
        print(f"Wrote {plot_gev_path}")
    print(f"Wrote {plot_path}")


def replot_m_corr_from_npz(
    data_path: str,
    *,
    plot_path: Optional[str] = None,
    plot_gev_path: Optional[str] = None,
    max_mode_fraction: float = 0.25,
) -> None:
    """Re-draw Wilson-corrected M_corr(|k|) from saved As/Bm (clover; no re-inversion)."""
    data = np.load(data_path, allow_pickle=True)
    if str(data["fermion"]) != "clover":
        raise ValueError("M_corr replot is only defined for clover fermions")
    grid = MomentumGrid.from_arrays(data["momentum_list"], data["latt_size"])
    masses = data["masses"]
    frac = float(data["max_mode_fraction"]) if "max_mode_fraction" in data else max_mode_fraction
    fermion = str(data["fermion"])
    ensemble = str(data["ensemble"])
    plot_path = plot_path or os.path.join(
        ROOT, f"artifacts/plots/qprop_M_{fermion}_{ensemble}_M_corr.pdf"
    )
    plot_gev_path = plot_gev_path or os.path.join(
        ROOT, f"artifacts/plots/qprop_M_{fermion}_{ensemble}_M_corr_gev.pdf"
    )
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    k_ref = None
    m_mean_all, m_sdev_all = [], []
    for i in range(len(masses)):
        k, m_cfg, _ = collect_kbin_mass_corr_curve(
            data["As"][i], data["Bm"][i], grid, max_mode_fraction=frac
        )
        if k_ref is None:
            k_ref = k
        m_jk = jk_ls_avg(jackknife(np.real(m_cfg)))
        m_mean_all.append(np.asarray([float(x.mean) for x in m_jk], dtype=np.float64))
        m_sdev_all.append(np.asarray([float(x.sdev) for x in m_jk], dtype=np.float64))
    m_mean_all = np.stack(m_mean_all)
    m_sdev_all = np.stack(m_sdev_all)
    plot_M_mass_scan(plot_path, masses, k_ref, m_mean_all, m_sdev_all)
    a_fm = float(data["a_fm"]) if "a_fm" in data else load_a_fm(ensemble)
    if a_fm is not None:
        plot_M_mass_scan(
            plot_gev_path,
            masses,
            k_lattice_to_gev(k_ref, a_fm),
            m_lattice_to_gev(m_mean_all, a_fm),
            m_lattice_to_gev(m_sdev_all, a_fm),
            xlabel=r"$|p|$ (GeV)",
            ylabel=r"$M_{\mathrm{corr}}(|p|)$ (GeV)",
        )
        print(f"Wrote {plot_gev_path}")
    print(f"Wrote {plot_path}")


# --- run parameters ---
fermion = "staggered"  # "staggered" | "clover"
mass_ls = [-0.08, -0.06, -0.038888, -0.02]
ensemble = "S24T24_cg_ipg"
N_conf = 50
exclude_k_latt = ()
max_mode_fraction = 0.25
merge_by_k = False
measurement = "wall_fft"  # "fft" | "wall_fft"
temporal_momenta: Optional[List[int]] = None  # None -> 0,2,...,T/2
source_position = [0, 0, 0, 0]
a_fm_override: Optional[float] = None  # None -> load from static_potential_scale npz if present
tol = 1e-8
maxiter = 10000
xi_0, nu = 1.0, 1.0
csw_r = 1.02868
csw_t = 1.02868
multigrid = None
Nc = 3


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="M(|k|) scan from CG+IPG gauges")
    parser.add_argument("--fermion", choices=("staggered", "clover"), default=None)
    parser.add_argument("--ensemble", default=None)
    parser.add_argument("--n-conf", type=int, default=None)
    parser.add_argument("--measurement", choices=("fft", "wall_fft"), default=None)
    parser.add_argument("--max-mode-fraction", type=float, default=None)
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Re-draw PDFs from existing artifacts/data/qprop_M_{fermion}_{ensemble}.npz",
    )
    parser.add_argument(
        "--replot-m-corr",
        action="store_true",
        help="Re-draw clover M_corr (Wilson term subtracted) from existing npz",
    )
    args = parser.parse_args()
    if args.fermion is not None:
        fermion = args.fermion
    if args.ensemble is not None:
        ensemble = args.ensemble
    if args.n_conf is not None:
        N_conf = args.n_conf
    if args.measurement is not None:
        measurement = args.measurement
    if args.max_mode_fraction is not None:
        max_mode_fraction = args.max_mode_fraction

    data_path = os.path.join(ROOT, f"artifacts/data/qprop_M_{fermion}_{ensemble}.npz")
    if args.replot_m_corr:
        if not os.path.isfile(data_path):
            raise FileNotFoundError(data_path)
        replot_m_corr_from_npz(data_path)
        sys.exit(0)
    if args.replot_only:
        if not os.path.isfile(data_path):
            raise FileNotFoundError(data_path)
        replot_mass_scan_from_npz(data_path)
        sys.exit(0)

    cache_dir = os.path.join(ROOT, ".cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    init([1, 1, 1, 1], resource_path=cache_dir)
    latt_size = lattice_size_for_ensemble(ensemble)
    latt_info = core.LatticeInfo(latt_size, -1, xi_0 / nu)
    is_root = latt_info.mpi_rank == 0
    eta_ops = staggered_eta_ops(latt_info) if fermion == "staggered" else None
    gamma_ops = clover_gamma_ops() if fermion == "clover" else None

    t_momenta = temporal_momenta if temporal_momenta is not None else default_temporal_momenta(latt_size)
    momentum_array = build_dressing_momentum_list(
        latt_size,
        t_momenta,
        max_mode_fraction=max_mode_fraction,
        one_rep_per_shell=True,
    )
    grid = MomentumGrid.from_arrays(momentum_array, np.asarray(latt_size))

    if is_root:
        print(
            f"fermion={fermion}, mass_ls={mass_ls}, {len(momentum_array)} 4-momenta, "
            f"measurement={measurement}, dressing=scalar_ab, "
            f"max_mode_fraction={max_mode_fraction}, temporal={t_momenta}"
        )
        if exclude_k_latt:
            print(f"Excluding |k| bins (lattice units): {exclude_k_latt}")

    scan_results = []
    for bare_mass in mass_ls:
        dirac = build_dirac(latt_info, fermion, bare_mass, tol, maxiter, xi_0, csw_r, csw_t, multigrid)
        as_cfg, bm_cfg = [], []
        for cfg in tqdm(range(N_conf), desc=f"mass {bare_mass:.6f}", disable=not is_root):
            gauge = io.readNERSCGauge(gauge_path(ensemble, cfg))
            with dirac.useGauge(gauge):
                cfg_greens = measure_greens(
                    latt_info,
                    dirac,
                    fermion,
                    grid,
                    eta_ops,
                    gamma_ops,
                    measurement=measurement,
                    source_position=source_position,
                )
            if is_root:
                a_val, b_val = scalar_ab_from_greens(cfg_greens, grid, nc=Nc)
                as_cfg.append(a_val)
                bm_cfg.append(b_val)

        if not is_root:
            continue

        as_arr = np.asarray(as_cfg, dtype=np.float64)
        bm_arr = np.asarray(bm_cfg, dtype=np.float64)
        summary = summarize_shell_analysis(as_arr, bm_arr, grid, max_mode_fraction=max_mode_fraction)
        k_spatial, m_paper_cfg, kbin_labels = collect_kbin_mass_curve(
            as_arr,
            bm_arr,
            grid,
            max_mode_fraction=max_mode_fraction,
            merge_by_k=merge_by_k,
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
                median_As_p4_spread=float(np.nanmedian(summary["As_p4_spread"])),
                median_Bm_p4_spread=float(np.nanmedian(summary["Bm_p4_spread"])),
            )
        )
        med_as = scan_results[-1]["median_As_p4_spread"]
        med_bm = scan_results[-1]["median_Bm_p4_spread"]
        print(
            f"mass={bare_mass:+.6f}: median p4 spread As={med_as:.2f}, Bm={med_bm:.2f}, "
            f"shells={len(k_spatial)}"
        )
        if med_as > P4_SPREAD_WARN or med_bm > P4_SPREAD_WARN:
            print(
                f"  WARNING: p4 spread exceeds {P4_SPREAD_WARN:.0%}; "
                "Eq. (25) p4 average may be unreliable."
            )
        for i, label in enumerate(summary["shell_label"][:5]):
            print(
                f"  shell {label} |k|={summary['k_spatial'][i]:.4f} "
                f"M={summary['M_paper_real_mean'][i]:+.4f} "
                f"As_spread={summary['As_p4_spread'][i]:.2f} "
                f"Bm_spread={summary['Bm_p4_spread'][i]:.2f}"
            )

    if not is_root:
        sys.exit(0)

    k_ref = scan_results[0]["kbin_k_spatial"]
    m_mean_all = np.stack([item["kbin_M_mean"] for item in scan_results])
    m_sdev_all = np.stack([item["kbin_M_sdev"] for item in scan_results])
    mass_arr = np.asarray([item["mass"] for item in scan_results], dtype=np.float64)

    os.makedirs(os.path.join(ROOT, "artifacts/data"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "artifacts/plots"), exist_ok=True)
    data_path = os.path.join(ROOT, f"artifacts/data/qprop_M_{fermion}_{ensemble}.npz")
    plot_path = os.path.join(ROOT, f"artifacts/plots/qprop_M_{fermion}_{ensemble}.pdf")

    a_fm = a_fm_override if a_fm_override is not None else load_a_fm(ensemble)
    save_kwargs = dict(
        fermion=np.asarray(fermion),
        ensemble=np.asarray(ensemble),
        masses=mass_arr,
        momentum_list=grid.momentum_list,
        latt_size=grid.latt_size,
        kbin_k_spatial=k_ref,
        kbin_M_mean=m_mean_all,
        kbin_M_sdev=m_sdev_all,
        kbin_M_cfg=np.stack([item["kbin_M_cfg"] for item in scan_results]),
        kbin_label=scan_results[0]["kbin_label"],
        exclude_k_latt=np.asarray(exclude_k_latt),
        max_mode_fraction=np.asarray(max_mode_fraction),
        measurement=np.asarray(measurement),
        temporal_momenta=np.asarray(t_momenta),
        merge_by_k=np.asarray(merge_by_k),
        As=np.stack([item["As"] for item in scan_results]),
        Bm=np.stack([item["Bm"] for item in scan_results]),
    )
    if a_fm is not None:
        save_kwargs["a_fm"] = np.asarray(a_fm)
        save_kwargs["kbin_p_gev"] = k_lattice_to_gev(k_ref, a_fm)
    np.savez(data_path, **save_kwargs)

    if len(mass_ls) == 1:
        plot_M_vs_k(k_ref, scan_results[0]["kbin_M_cfg"], plot_path)
    else:
        plot_M_mass_scan(plot_path, mass_arr, k_ref, m_mean_all, m_sdev_all)
        if a_fm is not None:
            p_gev = k_lattice_to_gev(k_ref, a_fm)
            plot_gev_path = os.path.join(
                ROOT, f"artifacts/plots/qprop_M_{fermion}_{ensemble}_gev.pdf"
            )
            plot_M_mass_scan(
                plot_gev_path,
                mass_arr,
                p_gev,
                m_mean_all * (0.1973269804 / a_fm),
                m_sdev_all * (0.1973269804 / a_fm),
                xlabel=r"$|p|$ (GeV)",
                ylabel=r"$M(|p|)$ (GeV)",
            )
            print(f"Wrote {plot_gev_path}")

    print(f"Wrote {data_path}")
    print(f"Wrote {plot_path}")

# %%
