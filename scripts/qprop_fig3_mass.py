#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib")))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


DEFAULT_MASSES = (-0.080000, -0.060000, -0.038888, -0.020000, 0.000000)
DEFAULT_LATT_SIZE = (16, 16, 16, 16)
HBARC_GEV_FM = 0.1973269804
NC = 3
contract = None
init = None
core = None
gamma = None
io = None
source = None


def load_runtime_deps() -> None:
    global contract, init, core, gamma, io, source
    if core is not None:
        return
    from opt_einsum import contract as opt_contract
    from pyquda import init as pyquda_init
    from pyquda_utils import core as pyquda_core
    from pyquda_utils import gamma as pyquda_gamma
    from pyquda_utils import io as pyquda_io
    from pyquda_utils import source as pyquda_source

    contract = opt_contract
    init = pyquda_init
    core = pyquda_core
    gamma = pyquda_gamma
    io = pyquda_io
    source = pyquda_source


@dataclass(frozen=True)
class MomentumBasis:
    modes: np.ndarray
    momentum_list: np.ndarray
    k_spatial: np.ndarray
    k_norm: np.ndarray
    wilson_spatial: np.ndarray
    wilson_4d: np.ndarray
    shell_id: np.ndarray
    mask_cylinder: np.ndarray
    analysis_shells: list[tuple[int, int, int]]
    shell_indices: list[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh Fig. 3-style Coulomb-gauge quark mass-function measurement "
            "from CG+IPG gauges. Existing .npz files are never read."
        )
    )
    parser.add_argument("--ensemble", default="S16T16_cg_ipg", help="Label used in output filenames.")
    parser.add_argument(
        "--gauge-dir",
        type=Path,
        default=Path("ensemble/S16T16_cg_ipg/gauge"),
        help="Directory containing CG+IPG NERSC gauge files.",
    )
    parser.add_argument(
        "--glob",
        default="wilson_b6.cg.ipg.1e-08.*",
        help="Gauge filename glob. Defaults to the 50 1e-08 S16T16_cg_ipg configurations.",
    )
    parser.add_argument("--n-conf", type=int, default=50, help="Number of matched configurations to process.")
    parser.add_argument(
        "--latt-size",
        default="auto",
        help="Lattice size as Lx,Ly,Lz,Lt, or auto to infer from names like S32T32.",
    )
    parser.add_argument(
        "--a-fm",
        type=float,
        default=None,
        help="Optional lattice spacing in fm. If set, plot |k| and M in GeV.",
    )
    parser.add_argument(
        "--max-k-bins",
        type=int,
        default=12,
        help="Maximum number of low-|k| bins to plot/analyze after cylinder-cut binning.",
    )
    parser.add_argument(
        "--skip-k-bins",
        type=int,
        default=0,
        help="Number of lowest-|k| bins to skip before applying --max-k-bins.",
    )
    parser.add_argument(
        "--min-k-latt",
        type=float,
        default=None,
        help="Optional lower cut on lattice-unit |k| bins.",
    )
    parser.add_argument(
        "--max-k-latt",
        type=float,
        default=None,
        help="Optional upper cut on lattice-unit |k| bins.",
    )
    parser.add_argument(
        "--min-k-gev",
        type=float,
        default=None,
        help="Optional lower cut on physical |k| bins; requires --a-fm.",
    )
    parser.add_argument(
        "--max-k-gev",
        type=float,
        default=None,
        help="Optional upper cut on physical |k| bins; requires --a-fm.",
    )
    parser.add_argument(
        "--masses",
        default=",".join(f"{mass:.6f}" for mass in DEFAULT_MASSES),
        help="Comma-separated bare clover mass parameters to scan.",
    )
    parser.add_argument("--tag", default="mass_scan", help="Output filename tag.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--tol", type=float, default=1e-8, help="Dirac solver tolerance.")
    parser.add_argument("--maxiter", type=int, default=10000, help="Dirac solver maximum iterations.")
    parser.add_argument("--csw-r", type=float, default=1.02868, help="Spatial clover coefficient.")
    parser.add_argument("--csw-t", type=float, default=1.02868, help="Temporal clover coefficient.")
    parser.add_argument(
        "--plot-raw",
        action="store_true",
        help="Plot raw B/A instead of subtracting the Wilson/clover tree-level scalar offset.",
    )
    parser.add_argument(
        "--max-mode-fraction",
        type=float,
        default=0.25,
        help=(
            "Keep only spatial Fourier modes with |n_i| <= fraction*L_i before binning. "
            "The default keeps the physical branch and avoids averaging sin(p)-degenerate doubler modes."
        ),
    )
    parser.add_argument(
        "--resource-path",
        type=Path,
        default=Path(".cache"),
        help="PyQUDA resource/tune-cache path.",
    )
    argv = sys.argv[1:]
    if "--masses" in argv:
        idx = argv.index("--masses")
        if idx + 1 < len(argv):
            argv = argv[:idx] + [f"--masses={argv[idx + 1]}"] + argv[idx + 2 :]
    return parser.parse_args(argv)


def parse_masses(raw: str) -> list[float]:
    masses = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not masses:
        raise ValueError("--masses must contain at least one value")
    return masses


def parse_latt_size(raw: str, ensemble: str, gauge_dir: Path) -> tuple[int, int, int, int]:
    if raw.lower() != "auto":
        parts = [int(item.strip()) for item in raw.split(",") if item.strip()]
        if len(parts) != 4:
            raise ValueError("--latt-size must be auto or four comma-separated integers")
        return tuple(parts)

    text = f"{ensemble} {gauge_dir}"
    match = re.search(r"S(\d+)T(\d+)", text)
    if match is None:
        return DEFAULT_LATT_SIZE
    spatial = int(match.group(1))
    temporal = int(match.group(2))
    return (spatial, spatial, spatial, temporal)


def extract_cfg_index(path: Path) -> int:
    for token in reversed(path.name.split(".")):
        if token.isdigit():
            return int(token)
    match = re.search(r"\.(\d+)$", path.name)
    if match is None:
        raise ValueError(f"Could not extract configuration index from {path}")
    return int(match.group(1))


def list_gauge_files(gauge_dir: Path, glob_pattern: str, n_conf: int | None) -> list[Path]:
    files = sorted(
        (path for path in gauge_dir.glob(glob_pattern) if path.is_file()),
        key=lambda path: (extract_cfg_index(path), path.name),
    )
    if n_conf is not None:
        files = files[:n_conf]
    return files


def output_paths(ensemble: str, tag: str) -> tuple[Path, Path]:
    stem = f"qprop_fig3_mass_{ensemble}_{tag}"
    return (
        Path("artifacts/data") / f"{stem}.npz",
        Path("artifacts/plots") / f"{stem}.pdf",
    )


def ensure_outputs_available(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = "\n  ".join(str(path) for path in existing)
        raise SystemExit(f"Refusing to overwrite existing output(s). Pass --overwrite:\n  {joined}")


def signed_modes(extent: int) -> np.ndarray:
    return np.arange(-extent // 2 + 1, extent // 2 + 1, dtype=np.int32)


def shell_label_from_spatial_modes(nx: int, ny: int, nz: int) -> tuple[int, int, int]:
    return tuple(sorted((abs(int(nx)), abs(int(ny)), abs(int(nz)))))


def near_body_diagonal_with_parity(nx: int, ny: int, nz: int) -> bool:
    nonzero = [int(component) for component in (nx, ny, nz) if int(component) != 0]
    return not nonzero or all(component > 0 for component in nonzero) or all(component < 0 for component in nonzero)


def build_momentum_basis(latt_size: Sequence[int], max_mode_fraction: float) -> MomentumBasis:
    lx, ly, lz, lt = latt_size
    modes_x = signed_modes(lx)
    modes_y = signed_modes(ly)
    modes_z = signed_modes(lz)
    modes_t = signed_modes(lt)

    nx, ny, nz, nt = np.meshgrid(modes_x, modes_y, modes_z, modes_t, indexing="ij")
    momentum_list = np.stack([nx, ny, nz, nt], axis=-1).reshape(-1, 4)

    px = 2.0 * np.pi * momentum_list[:, 0] / lx
    py = 2.0 * np.pi * momentum_list[:, 1] / ly
    pz = 2.0 * np.pi * momentum_list[:, 2] / lz
    pt = 2.0 * np.pi * momentum_list[:, 3] / lt
    k_spatial = np.stack([np.sin(px), np.sin(py), np.sin(pz)], axis=-1)
    k_norm = np.linalg.norm(k_spatial, axis=1)
    wilson_spatial = (
        1.0
        - np.cos(px)
        + 1.0
        - np.cos(py)
        + 1.0
        - np.cos(pz)
    )
    wilson_4d = wilson_spatial + 1.0 - np.cos(pt)

    shell_id = np.empty(momentum_list.shape[0], dtype=object)
    for idx, mom in enumerate(momentum_list):
        shell_id[idx] = shell_label_from_spatial_modes(*mom[:3])

    cylinder_shells: set[tuple[int, int, int]] = set()
    max_n = lx // 2
    for n in range(max_n + 1):
        cylinder_shells.add((n, n, n))
        if n < max_n:
            cylinder_shells.add((n, n, n + 1))
            cylinder_shells.add((n, n + 1, n + 1))

    if not (0.0 < max_mode_fraction <= 0.5):
        raise ValueError("--max-mode-fraction must be in (0, 0.5]")
    max_modes = np.asarray(
        [
            int(np.floor(lx * max_mode_fraction)),
            int(np.floor(ly * max_mode_fraction)),
            int(np.floor(lz * max_mode_fraction)),
        ],
        dtype=np.int32,
    )
    mask_physical_branch = np.all(np.abs(momentum_list[:, :3]) <= max_modes[None, :], axis=1)

    mask_cylinder = np.asarray(
        [
            tuple(label) in cylinder_shells and near_body_diagonal_with_parity(*mom[:3]) and keep_branch
            for label, mom, keep_branch in zip(shell_id, momentum_list, mask_physical_branch)
        ],
        dtype=bool,
    )

    analysis_shells = sorted(
        {
            tuple(label)
            for label, keep, norm in zip(shell_id, mask_cylinder, k_norm)
            if keep and norm > 1e-12
        },
        key=lambda label: (np.linalg.norm(2.0 * np.pi * np.asarray(label, dtype=np.float64) / lx), label),
    )
    shell_indices = [
        np.asarray(
            [idx for idx, label in enumerate(shell_id) if mask_cylinder[idx] and label == shell],
            dtype=np.int64,
        )
        for shell in analysis_shells
    ]

    return MomentumBasis(
        modes=modes_x,
        momentum_list=momentum_list,
        k_spatial=k_spatial,
        k_norm=k_norm,
        wilson_spatial=wilson_spatial,
        wilson_4d=wilson_4d,
        shell_id=shell_id,
        mask_cylinder=mask_cylinder,
        analysis_shells=analysis_shells,
        shell_indices=shell_indices,
    )


def fft_trace_to_momentum(trace_tzyx: np.ndarray, modes: np.ndarray) -> np.ndarray:
    volume = np.prod(trace_tzyx.shape)
    fourier = np.fft.ifftn(trace_tzyx, axes=(0, 1, 2, 3)) * volume
    reorder = [np.mod(modes, extent) for extent in trace_tzyx.shape]
    ordered = fourier[np.ix_(reorder[0], reorder[1], reorder[2], reorder[3])]
    return np.transpose(ordered, (3, 2, 1, 0)).reshape(-1)


def gather_trace(point_propag, gamma_matrix) -> np.ndarray:
    contracted = contract("wtzyxijaa,ji->wtzyx", point_propag.data, gamma_matrix).get()
    gathered = core.gatherLattice(core.lexico(contracted, [0, 1, 2, 3, 4]), [0, 1, 2, 3])
    return np.asarray(gathered)


def jackknife_ratio(numerator: np.ndarray, denominator: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_cfg = numerator.shape[0]
    full = np.sum(numerator, axis=0) / np.sum(denominator, axis=0)
    if n_cfg < 2:
        return full, np.full_like(full, np.nan, dtype=np.float64)

    total_num = np.sum(numerator, axis=0)
    total_den = np.sum(denominator, axis=0)
    samples = (total_num[None, :] - numerator) / (total_den[None, :] - denominator)
    mean = np.mean(samples, axis=0)
    err = np.sqrt((n_cfg - 1) * np.mean((samples - mean[None, :]) ** 2, axis=0))
    return full, err


def jackknife_mean(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_cfg = values.shape[0]
    full = np.mean(values, axis=0)
    if n_cfg < 2:
        return full, np.full_like(full, np.nan, dtype=np.float64)

    total = np.sum(values, axis=0)
    samples = (total[None, :] - values) / (n_cfg - 1)
    mean = np.mean(samples, axis=0)
    err = np.sqrt((n_cfg - 1) * np.mean((samples - mean[None, :]) ** 2, axis=0))
    return full, err


def build_k_bins(shell_k_norm: np.ndarray) -> list[np.ndarray]:
    rounded = np.round(shell_k_norm, decimals=12)
    return [
        np.flatnonzero(rounded == value)
        for value in sorted(np.unique(rounded))
    ]


def weighted_cfg_average(values: np.ndarray, bin_indices: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    averaged = []
    for indices in bin_indices:
        bin_weights = weights[indices].astype(np.float64)
        averaged.append(np.average(values[:, indices], axis=1, weights=bin_weights))
    return np.stack(averaged, axis=1)


def select_k_bins(
    kbin_indices: list[np.ndarray],
    shell_k_norm: np.ndarray,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    selected = list(kbin_indices)
    if args.min_k_gev is not None:
        if args.a_fm is None:
            raise ValueError("--min-k-gev requires --a-fm")
        min_k_latt_from_gev = args.min_k_gev * args.a_fm / HBARC_GEV_FM
        selected = [indices for indices in selected if float(np.mean(shell_k_norm[indices])) >= min_k_latt_from_gev]
    if args.max_k_gev is not None:
        if args.a_fm is None:
            raise ValueError("--max-k-gev requires --a-fm")
        max_k_latt_from_gev = args.max_k_gev * args.a_fm / HBARC_GEV_FM
        selected = [indices for indices in selected if float(np.mean(shell_k_norm[indices])) <= max_k_latt_from_gev]
    if args.min_k_latt is not None:
        selected = [indices for indices in selected if float(np.mean(shell_k_norm[indices])) >= args.min_k_latt]
    if args.max_k_latt is not None:
        selected = [indices for indices in selected if float(np.mean(shell_k_norm[indices])) <= args.max_k_latt]
    if args.skip_k_bins:
        selected = selected[args.skip_k_bins :]
    if args.max_k_bins is not None:
        selected = selected[: args.max_k_bins]
    if not selected:
        raise ValueError("No |k| bins selected; relax --max-k-bins/--max-k-latt/--max-k-gev")
    return selected


def build_dirac(latt_size: Sequence[int], mass: float, args: argparse.Namespace):
    xi_0, nu = 1.0, 1.0
    latt_info = core.LatticeInfo(list(latt_size), -1, xi_0 / nu)
    dirac = core.getClover(
        latt_info,
        mass,
        args.tol,
        args.maxiter,
        xi_0,
        args.csw_r,
        args.csw_t,
        None,
    )
    return latt_info, dirac


def measure_mass(
    gauge_files: Sequence[Path],
    mass: float,
    basis: MomentumBasis,
    latt_size: Sequence[int],
    args: argparse.Namespace,
) -> dict[str, np.ndarray | float]:
    latt_info, dirac = build_dirac(latt_size, mass, args)
    is_root = latt_info.mpi_rank == 0

    gamma_i = gamma.gamma(0)
    gamma_x = gamma.gamma(1)
    gamma_y = gamma.gamma(2)
    gamma_z = gamma.gamma(4)

    shell_a = []
    shell_b = []
    shell_b_corr = []
    shell_tree = []
    max_imag_identity = 0.0
    max_real_spatial_gamma = 0.0

    for gauge_path in tqdm(gauge_files, desc=f"mass {mass:.6f}", disable=not is_root):
        gauge = io.readNERSCGauge(str(gauge_path))
        with dirac.useGauge(gauge):
            point_source = source.propagator(latt_info, "point", [0, 0, 0, 0])
            point_propag = core.invertPropagator(dirac, point_source)

            traces = [
                gather_trace(point_propag, gamma_i),
                gather_trace(point_propag, gamma_x),
                gather_trace(point_propag, gamma_y),
                gather_trace(point_propag, gamma_z),
            ]

        mom_identity, mom_gx, mom_gy, mom_gz = [
            fft_trace_to_momentum(trace, basis.modes) / NC for trace in traces
        ]
        max_imag_identity = max(max_imag_identity, float(np.max(np.abs(np.imag(mom_identity)))))
        max_real_spatial_gamma = max(
            max_real_spatial_gamma,
            float(
                max(
                    np.max(np.abs(np.real(mom_gx))),
                    np.max(np.abs(np.real(mom_gy))),
                    np.max(np.abs(np.real(mom_gz))),
                )
            ),
        )

        b_point = np.real(mom_identity)
        gamma_imag = np.stack([np.imag(mom_gx), np.imag(mom_gy), np.imag(mom_gz)], axis=1)
        k2_spatial = np.sum(basis.k_spatial**2, axis=1)
        a_point = np.full(k2_spatial.shape, np.nan, dtype=np.float64)
        np.divide(
            np.sum(basis.k_spatial * gamma_imag, axis=1),
            k2_spatial,
            out=a_point,
            where=k2_spatial > 1e-12,
        )

        cfg_a = []
        cfg_b = []
        cfg_b_corr = []
        cfg_tree = []
        for indices in basis.shell_indices:
            valid = np.isfinite(a_point[indices])
            valid_indices = indices[valid]
            b_corr = b_point[valid_indices] - basis.wilson_4d[valid_indices] * a_point[valid_indices]
            cfg_a.append(float(np.mean(a_point[valid_indices])))
            cfg_b.append(float(np.mean(b_point[valid_indices])))
            cfg_b_corr.append(float(np.mean(b_corr)))
            cfg_tree.append(float(np.mean(basis.wilson_4d[valid_indices])))
        shell_a.append(cfg_a)
        shell_b.append(cfg_b)
        shell_b_corr.append(cfg_b_corr)
        shell_tree.append(cfg_tree)

    shell_a_arr = np.asarray(shell_a, dtype=np.float64)
    shell_b_arr = np.asarray(shell_b, dtype=np.float64)
    shell_b_corr_arr = np.asarray(shell_b_corr, dtype=np.float64)
    shell_tree_arr = np.asarray(shell_tree, dtype=np.float64)
    shell_m_raw_cfg = shell_b_arr / shell_a_arr
    shell_m_cfg = shell_b_corr_arr / shell_a_arr
    raw_mean, raw_sdev = jackknife_mean(shell_m_raw_cfg)
    corr_mean, corr_sdev = jackknife_mean(shell_m_cfg)

    return {
        "mass": mass,
        "shell_A_cfg": shell_a_arr,
        "shell_B_cfg": shell_b_arr,
        "shell_B_corr_cfg": shell_b_corr_arr,
        "shell_tree_cfg": shell_tree_arr,
        "shell_M_raw_cfg": shell_m_raw_cfg,
        "shell_M_cfg": shell_m_cfg,
        "shell_M_raw_mean": raw_mean,
        "shell_M_raw_sdev": raw_sdev,
        "shell_M_mean": corr_mean,
        "shell_M_sdev": corr_sdev,
        "max_imag_identity": max_imag_identity,
        "max_real_spatial_gamma": max_real_spatial_gamma,
    }


def plot_results(
    output_pdf: Path,
    masses: Sequence[float],
    x_values: np.ndarray,
    y_mean: np.ndarray,
    y_sdev: np.ndarray,
    plot_raw: bool,
    a_fm: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    markers = ["o", "s", "^", "v", "D", "P", "X"]
    for idx, mass in enumerate(masses):
        yerr = None
        if np.any(np.isfinite(y_sdev[idx])):
            yerr = np.nan_to_num(y_sdev[idx], nan=0.0)
        ax.errorbar(
            x_values,
            y_mean[idx],
            yerr=yerr,
            marker=markers[idx % len(markers)],
            linestyle="-",
            linewidth=1.1,
            markersize=4.2,
            capsize=2.0,
            label=rf"$am_0={mass:.6g}$",
        )

    if a_fm is None:
        ax.set_xlabel(r"$|k| = \sqrt{\sum_i \sin^2(p_i)}$")
        ylabel = r"$B/A$" if plot_raw else r"$aM(|k|)$"
    else:
        ax.set_xlabel(r"$|k|$ [GeV]")
        ylabel = r"$M(|k|)$ [GeV]" if not plot_raw else r"$B/A$ [GeV]"
    ax.set_ylabel(ylabel)
    ax.set_title(r"CG+IPG quark mass function")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, transparent=True)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    masses = parse_masses(args.masses)
    data_path, plot_pdf = output_paths(args.ensemble, args.tag)
    ensure_outputs_available((data_path, plot_pdf), args.overwrite)

    gauge_dir = args.gauge_dir.resolve()
    latt_size = parse_latt_size(args.latt_size, args.ensemble, gauge_dir)
    gauge_files = list_gauge_files(gauge_dir, args.glob, args.n_conf)
    if not gauge_files:
        raise SystemExit(f"No gauge files matched {args.glob} in {gauge_dir}")

    load_runtime_deps()
    args.resource_path.mkdir(parents=True, exist_ok=True)
    init([1, 1, 1, 1], resource_path=str(args.resource_path))

    basis = build_momentum_basis(latt_size, args.max_mode_fraction)
    shell_k_norm = np.asarray(
        [
            np.mean(basis.k_norm[indices][np.isfinite(basis.k_norm[indices])])
            for indices in basis.shell_indices
        ],
        dtype=np.float64,
    )
    shell_count_selected = np.asarray([len(indices) for indices in basis.shell_indices], dtype=np.int32)
    kbin_indices_all = build_k_bins(shell_k_norm)
    kbin_indices = select_k_bins(kbin_indices_all, shell_k_norm, args)
    kbin_k_norm = np.asarray(
        [float(np.average(shell_k_norm[indices], weights=shell_count_selected[indices])) for indices in kbin_indices],
        dtype=np.float64,
    )
    kbin_count_selected = np.asarray(
        [int(np.sum(shell_count_selected[indices])) for indices in kbin_indices],
        dtype=np.int32,
    )
    kbin_count_shells = np.asarray([len(indices) for indices in kbin_indices], dtype=np.int32)

    print(f"Using {len(gauge_files)} configurations from {gauge_dir}")
    print(f"Lattice size: {latt_size}")
    print(f"Mass scan: {', '.join(f'{mass:.6f}' for mass in masses)}")
    print(
        f"Built {basis.momentum_list.shape[0]} 4-momenta "
        f"({int(np.sum(basis.mask_cylinder))} survive the cylinder cut)"
    )
    print(f"Spatial physical-branch mode cut: |n_i| <= {args.max_mode_fraction:g} L_i")
    print(
        f"Built {len(basis.analysis_shells)} analysis shell averages "
        f"and selected {len(kbin_indices)} of {len(kbin_indices_all)} |k| bins for M(|k|)"
    )
    if args.a_fm is None:
        print("All axes are lattice units; no lattice spacing or physical quark mass is assumed.")
    else:
        print(f"Using a = {args.a_fm:.6g} fm for plot axes in GeV.")

    measurements = [measure_mass(gauge_files, mass, basis, latt_size, args) for mass in masses]

    raw_mean = np.stack([item["shell_M_raw_mean"] for item in measurements])
    raw_sdev = np.stack([item["shell_M_raw_sdev"] for item in measurements])
    corr_mean = np.stack([item["shell_M_mean"] for item in measurements])
    corr_sdev = np.stack([item["shell_M_sdev"] for item in measurements])

    kbin_a_cfg = np.stack(
        [weighted_cfg_average(item["shell_A_cfg"], kbin_indices, shell_count_selected) for item in measurements]
    )
    kbin_b_cfg = np.stack(
        [weighted_cfg_average(item["shell_B_cfg"], kbin_indices, shell_count_selected) for item in measurements]
    )
    kbin_b_corr_cfg = np.stack(
        [weighted_cfg_average(item["shell_B_corr_cfg"], kbin_indices, shell_count_selected) for item in measurements]
    )
    kbin_tree_cfg = np.stack(
        [weighted_cfg_average(item["shell_tree_cfg"], kbin_indices, shell_count_selected) for item in measurements]
    )
    kbin_m_raw_cfg = kbin_b_cfg / kbin_a_cfg
    kbin_m_cfg = kbin_b_corr_cfg / kbin_a_cfg

    kbin_raw_mean = []
    kbin_raw_sdev = []
    kbin_corr_mean = []
    kbin_corr_sdev = []
    for idx in range(len(masses)):
        mean, sdev = jackknife_mean(kbin_m_raw_cfg[idx])
        kbin_raw_mean.append(mean)
        kbin_raw_sdev.append(sdev)
        mean, sdev = jackknife_mean(kbin_m_cfg[idx])
        kbin_corr_mean.append(mean)
        kbin_corr_sdev.append(sdev)
    kbin_raw_mean = np.stack(kbin_raw_mean)
    kbin_raw_sdev = np.stack(kbin_raw_sdev)
    kbin_corr_mean = np.stack(kbin_corr_mean)
    kbin_corr_sdev = np.stack(kbin_corr_sdev)

    plotted_mean = kbin_raw_mean if args.plot_raw else kbin_corr_mean
    plotted_sdev = kbin_raw_sdev if args.plot_raw else kbin_corr_sdev
    if args.a_fm is None:
        plot_x = kbin_k_norm
        plot_y = plotted_mean
        plot_yerr = plotted_sdev
    else:
        conversion = HBARC_GEV_FM / args.a_fm
        plot_x = kbin_k_norm * conversion
        plot_y = plotted_mean * conversion
        plot_yerr = plotted_sdev * conversion

    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        data_path,
        ensemble=args.ensemble,
        tag=args.tag,
        gauge_dir=str(gauge_dir),
        gauge_files=np.asarray([str(path) for path in gauge_files]),
        cfg_indices=np.asarray([extract_cfg_index(path) for path in gauge_files], dtype=np.int32),
        masses=np.asarray(masses, dtype=np.float64),
        latt_size=np.asarray(latt_size, dtype=np.int32),
        a_fm=np.nan if args.a_fm is None else args.a_fm,
        momentum_list=basis.momentum_list,
        k_spatial=basis.k_spatial,
        k_norm=basis.k_norm,
        wilson_spatial=basis.wilson_spatial,
        wilson_4d=basis.wilson_4d,
        mask_cylinder=basis.mask_cylinder,
        shell_label=np.asarray([str(shell) for shell in basis.analysis_shells]),
        shell_k_norm=shell_k_norm,
        shell_count_selected=shell_count_selected,
        kbin_k_norm=kbin_k_norm,
        kbin_count_selected=kbin_count_selected,
        kbin_count_shells=kbin_count_shells,
        kbin_count_available=np.asarray(len(kbin_indices_all), dtype=np.int32),
        shell_A_cfg=np.stack([item["shell_A_cfg"] for item in measurements]),
        shell_B_cfg=np.stack([item["shell_B_cfg"] for item in measurements]),
        shell_B_corr_cfg=np.stack([item["shell_B_corr_cfg"] for item in measurements]),
        shell_tree_cfg=np.stack([item["shell_tree_cfg"] for item in measurements]),
        shell_M_raw_cfg=np.stack([item["shell_M_raw_cfg"] for item in measurements]),
        shell_M_cfg=np.stack([item["shell_M_cfg"] for item in measurements]),
        shell_M_raw_mean=raw_mean,
        shell_M_raw_sdev=raw_sdev,
        shell_M_mean=corr_mean,
        shell_M_sdev=corr_sdev,
        kbin_A_cfg=kbin_a_cfg,
        kbin_B_cfg=kbin_b_cfg,
        kbin_B_corr_cfg=kbin_b_corr_cfg,
        kbin_tree_cfg=kbin_tree_cfg,
        kbin_M_raw_cfg=kbin_m_raw_cfg,
        kbin_M_cfg=kbin_m_cfg,
        kbin_M_raw_mean=kbin_raw_mean,
        kbin_M_raw_sdev=kbin_raw_sdev,
        kbin_M_mean=kbin_corr_mean,
        kbin_M_sdev=kbin_corr_sdev,
        plotted_quantity="raw_B_over_A" if args.plot_raw else "wilson_clover_tree_scalar_corrected",
        plot_x=plot_x,
        plot_y=plot_y,
        plot_y_sdev=plot_yerr,
        plot_units="lattice" if args.a_fm is None else "GeV",
        max_imag_identity=np.asarray([item["max_imag_identity"] for item in measurements]),
        max_real_spatial_gamma=np.asarray([item["max_real_spatial_gamma"] for item in measurements]),
    )

    plot_results(plot_pdf, masses, plot_x, plot_y, plot_yerr, args.plot_raw, args.a_fm)

    print(f"max |Im Tr[S]| by mass: {np.asarray([item['max_imag_identity'] for item in measurements])}")
    print(
        "max |Re Tr[gamma_i S]| by mass: "
        f"{np.asarray([item['max_real_spatial_gamma'] for item in measurements])}"
    )
    print(f"Saved fresh run data to {data_path}")
    print(f"Saved {plot_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
