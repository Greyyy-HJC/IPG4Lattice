import argparse
import os
from pathlib import Path

import gvar as gv
import matplotlib
matplotlib.use("Agg")
import numpy as np
from opt_einsum import contract
from pyquda import init
from pyquda_utils import core, gamma, io, source
from pyquda_utils.phase import MomentumPhase
from tqdm.auto import tqdm

from lametlat.utils.plot_settings import *
from lametlat.utils.resampling import jackknife, jk_ls_avg


import matplotlib.pyplot as plt


ENSEMBLE = "S16T16_cg_ipg"
N_CONF = 50
CFG_GLOB = "wilson_b6.cg.ipg.1e-08.*"
CYLINDER_RADIUS = 1.1
PHASE_CHUNK_SIZE = 128
MAX_N2 = None

# Lattice parameters
XI_0, NU = 1.0, 1.0
MASS = -0.038888  # kappa = 0.12623
CSW_R = 1.02868
CSW_T = 1.02868
MULTIGRID = None  # [[4, 4, 4, 4], [2, 2, 2, 8]]

ENSEMBLE_META = {
    "S16T16": {
        "latt_size": [16, 16, 16, 16],
        "gauge_dir": "ensemble/S16T16",
        "cfg_glob": "wilson_b6.*",
    },
    "S16T16_cg": {
        "latt_size": [16, 16, 16, 16],
        "gauge_dir": "ensemble/S16T16_cg/gauge",
        "cfg_glob": "wilson_b6.cg.1e-08.*",
    },
    "S16T16_cg_ipg": {
        "latt_size": [16, 16, 16, 16],
        "gauge_dir": "ensemble/S16T16_cg_ipg/gauge",
        "cfg_glob": CFG_GLOB,
    },
    "S32T32_cg_ipg": {
        "latt_size": [32, 32, 32, 32],
        "gauge_dir": "ensemble/S32T32_cg_ipg/gauge",
        "cfg_glob": "wilson_b5_95_fixed.*.ipg",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure a Fig.3-like Wilson-corrected static mass function M(|k|) from Fourier-space quark propagator traces."
    )
    parser.add_argument(
        "--ensemble",
        default=ENSEMBLE,
        choices=sorted(ENSEMBLE_META.keys()),
        help="Ensemble name.",
    )
    parser.add_argument(
        "--n-conf",
        default=N_CONF,
        type=int,
        help="Maximum number of configurations to process.",
    )
    parser.add_argument(
        "--max-n2",
        default=MAX_N2,
        type=int,
        help="Optional cutoff on n_x^2 + n_y^2 + n_z^2 for the spatial momentum grid.",
    )
    parser.add_argument(
        "--no-cylinder-cut",
        action="store_true",
        help="Disable the cylinder cut when forming shell averages.",
    )
    parser.add_argument(
        "--phase-chunk-size",
        default=PHASE_CHUNK_SIZE,
        type=int,
        help="How many 4-momenta to Fourier transform at once.",
    )
    return parser.parse_args()


def cfg_sort_key(path: Path):
    tokens = path.name.split(".")
    try:
        return tuple(int(tok) if tok.isdigit() else tok for tok in tokens)
    except ValueError:
        return tuple(tokens)


def canonical_momentum_indices(length):
    if length % 2 != 0:
        raise ValueError(f"Expected an even lattice extent, got {length}")
    return list(range(-(length // 2) + 1, length // 2 + 1))


def cylinder_distance(vec):
    vec = np.asarray(vec, dtype=np.float64)
    if np.allclose(vec, 0.0):
        return 0.0
    axis = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    proj = np.dot(vec, axis)
    perp_sq = np.dot(vec, vec) - proj**2
    return np.sqrt(max(perp_sq, 0.0))


def orbit_key_from_spatial_momentum(spatial_momentum):
    return tuple(sorted(abs(int(v)) for v in spatial_momentum))


def build_momentum_grid(latt_size, max_n2):
    lx, ly, lz, lt = latt_size
    spatial_indices = canonical_momentum_indices(lx)
    temporal_indices = canonical_momentum_indices(lt)

    momentum_list = []
    orbit_keys = []
    shell_keys = []
    kvec_norm = []
    k4_values = []
    k_spatial = []
    pvec_norm = []
    p4_values = []
    mask_cylinder = []
    wilson_spatial = []
    wilson_full = []

    for nx in spatial_indices:
        for ny in spatial_indices:
            for nz in spatial_indices:
                n2 = nx**2 + ny**2 + nz**2
                if max_n2 is not None and n2 > max_n2:
                    continue

                spatial_vec = np.array([nx, ny, nz], dtype=np.int32)
                orbit_key = orbit_key_from_spatial_momentum(spatial_vec)
                kx = np.sin(2.0 * np.pi * nx / lx)
                ky = np.sin(2.0 * np.pi * ny / ly)
                kz = np.sin(2.0 * np.pi * nz / lz)
                spatial_k = np.array([kx, ky, kz], dtype=np.float64)
                spatial_norm = np.linalg.norm(spatial_k)
                px = 2.0 * np.pi * nx / lx
                py = 2.0 * np.pi * ny / ly
                pz = 2.0 * np.pi * nz / lz
                spatial_p = np.array([px, py, pz], dtype=np.float64)
                spatial_p_norm = np.linalg.norm(spatial_p)
                shell_key = str(orbit_key)
                keep_cylinder = cylinder_distance(spatial_vec) <= CYLINDER_RADIUS
                spatial_wilson = np.sum(1.0 - np.cos(spatial_p))

                for nt in temporal_indices:
                    kt = np.sin(2.0 * np.pi * nt / lt)
                    pt = 2.0 * np.pi * nt / lt
                    momentum_list.append([nx, ny, nz, nt])
                    orbit_keys.append(orbit_key)
                    shell_keys.append(shell_key)
                    kvec_norm.append(spatial_norm)
                    k4_values.append(kt)
                    k_spatial.append(spatial_k)
                    pvec_norm.append(spatial_p_norm)
                    p4_values.append(pt)
                    mask_cylinder.append(keep_cylinder)
                    wilson_spatial.append(spatial_wilson)
                    wilson_full.append(spatial_wilson + (1.0 - np.cos(pt)))

    momentum_array = np.asarray(momentum_list, dtype=np.int32)
    orbit_labels = np.asarray([str(key) for key in orbit_keys])
    shell_labels = np.asarray(shell_keys)
    kvec_norm = np.asarray(kvec_norm, dtype=np.float64)
    k4_values = np.asarray(k4_values, dtype=np.float64)
    k_spatial = np.asarray(k_spatial, dtype=np.float64)
    pvec_norm = np.asarray(pvec_norm, dtype=np.float64)
    p4_values = np.asarray(p4_values, dtype=np.float64)
    mask_cylinder = np.asarray(mask_cylinder, dtype=bool)
    wilson_spatial = np.asarray(wilson_spatial, dtype=np.float64)
    wilson_full = np.asarray(wilson_full, dtype=np.float64)
    spatial_n2 = np.sum(momentum_array[:, :3] ** 2, axis=1)

    orbit_map = {}
    for idx, (orbit_label, nt) in enumerate(zip(orbit_labels, momentum_array[:, 3])):
        orbit_map.setdefault((orbit_label, int(nt)), []).append(idx)

    unique_shell_labels = np.unique(shell_labels)
    shell_to_indices = {}
    for shell_label in unique_shell_labels:
        shell_to_indices[shell_label] = np.where(shell_labels == shell_label)[0]

    return {
        "momentum_array": momentum_array,
        "momentum_label": np.asarray([f"({px},{py},{pz},{pt})" for px, py, pz, pt in momentum_array]),
        "orbit_labels": orbit_labels,
        "shell_labels": shell_labels,
        "kvec_norm": kvec_norm,
        "k4_values": k4_values,
        "k_spatial": k_spatial,
        "pvec_norm": pvec_norm,
        "p4_values": p4_values,
        "mask_cylinder": mask_cylinder,
        "spatial_n2": spatial_n2,
        "wilson_spatial": wilson_spatial,
        "wilson_full": wilson_full,
        "orbit_map": orbit_map,
        "shell_to_indices": shell_to_indices,
    }


def jackknife_avg(data):
    data = np.asarray(data)
    if data.shape[0] == 1:
        return gv.gvar(data[0], np.zeros_like(data[0], dtype=np.float64))
    return jk_ls_avg(jackknife(data))


def jackknife_ratio(num, den):
    num = np.asarray(num)
    den = np.asarray(den)
    if num.shape[0] == 1:
        mean = np.asarray(num[0] / den[0], dtype=np.float64)
        return gv.gvar(mean, np.zeros_like(mean, dtype=np.float64))
    return jk_ls_avg(jackknife(num) / jackknife(den))


def build_cfg_file_list(ensemble, n_conf):
    meta = ENSEMBLE_META[ensemble]
    gauge_dir = Path(meta["gauge_dir"])
    if not gauge_dir.exists():
        raise FileNotFoundError(f"Gauge directory not found: {gauge_dir}")

    cfg_files = sorted(gauge_dir.glob(meta["cfg_glob"]), key=cfg_sort_key)
    if len(cfg_files) == 0:
        raise FileNotFoundError(f"No gauge files matched {meta['cfg_glob']} in {gauge_dir}")
    return cfg_files[:n_conf]


def compute_as_from_traces(greens_by_gamma, momentum_meta):
    g_spatial = np.stack(
        [
            np.asarray(greens_by_gamma["gX"]),
            np.asarray(greens_by_gamma["gY"]),
            np.asarray(greens_by_gamma["gZ"]),
        ],
        axis=-1,
    )
    k_spatial = momentum_meta["k_spatial"][None, :, :]
    valid = np.abs(k_spatial) > 1e-12

    as_components = np.divide(
        np.imag(g_spatial),
        k_spatial,
        out=np.full(g_spatial.shape, np.nan, dtype=np.float64),
        where=valid,
    )

    valid_count = np.sum(np.isfinite(as_components), axis=-1)
    a_s = np.divide(
        np.nansum(as_components, axis=-1),
        valid_count,
        out=np.full(as_components.shape[:2], np.nan, dtype=np.float64),
        where=valid_count > 0,
    )

    b_massive = np.real(np.asarray(greens_by_gamma["I"]))
    return a_s, b_massive, as_components


def summarize_orbit_spread(a_s, b_massive, momentum_meta):
    orbit_a_rel = []
    orbit_b_rel = []
    orbit_labels = np.unique(momentum_meta["orbit_labels"])
    temporal_modes = np.unique(momentum_meta["momentum_array"][:, 3])

    for orbit_label in orbit_labels:
        orbit_a_by_p4 = []
        orbit_b_by_p4 = []

        for nt in temporal_modes:
            orbit_member_indices = np.where(
                (momentum_meta["orbit_labels"] == orbit_label)
                & (momentum_meta["momentum_array"][:, 3] == nt)
            )[0]
            if len(orbit_member_indices) == 0:
                continue

            orbit_a_block = a_s[:, orbit_member_indices]
            orbit_b_block = b_massive[:, orbit_member_indices]
            if np.isfinite(orbit_a_block).any():
                orbit_a_by_p4.append(np.nanmean(orbit_a_block))
            if np.isfinite(orbit_b_block).any():
                orbit_b_by_p4.append(np.nanmean(orbit_b_block))

        if len(orbit_a_by_p4) >= 2:
            orbit_a_by_p4 = np.asarray(orbit_a_by_p4, dtype=np.float64)
            mean_abs_a = abs(np.nanmean(orbit_a_by_p4))
            if mean_abs_a > 1e-12:
                orbit_a_rel.append(np.nanstd(orbit_a_by_p4) / mean_abs_a)

        if len(orbit_b_by_p4) >= 2:
            orbit_b_by_p4 = np.asarray(orbit_b_by_p4, dtype=np.float64)
            mean_abs_b = abs(np.nanmean(orbit_b_by_p4))
            if mean_abs_b > 1e-12:
                orbit_b_rel.append(np.nanstd(orbit_b_by_p4) / mean_abs_b)

    a_spread = float(np.nanmedian(orbit_a_rel)) if orbit_a_rel else np.nan
    b_spread = float(np.nanmedian(orbit_b_rel)) if orbit_b_rel else np.nan
    return a_spread, b_spread


def shell_analysis(a_s, b_massive, momentum_meta, use_cylinder_cut):
    shell_results = []
    selected_global_mask = (
        momentum_meta["mask_cylinder"] if use_cylinder_cut else np.ones_like(momentum_meta["mask_cylinder"], dtype=bool)
    )

    for shell_key, shell_indices in momentum_meta["shell_to_indices"].items():
        shell_indices = np.asarray(shell_indices, dtype=np.int64)
        selected_indices = shell_indices[selected_global_mask[shell_indices]]
        if len(selected_indices) == 0:
            continue

        selected_a = a_s[:, selected_indices]
        if not np.isfinite(selected_a).any():
            continue

        a_cfg = np.nanmean(selected_a, axis=1)
        b_cfg = np.nanmean(b_massive[:, selected_indices], axis=1)
        finite_mask = np.isfinite(a_cfg) & np.isfinite(b_cfg) & (np.abs(a_cfg) > 1e-12)
        if not finite_mask.any():
            continue

        selected_count = int(len(selected_indices))
        total_count = int(len(shell_indices))
        p_norm = float(np.nanmean(momentum_meta["pvec_norm"][selected_indices]))
        k_norm = float(np.nanmean(momentum_meta["kvec_norm"][selected_indices]))
        tree_level_shift = float(np.nanmean(momentum_meta["wilson_full"][selected_indices]))
        ratio_raw_gvar = jackknife_ratio(b_cfg[finite_mask], a_cfg[finite_mask])
        # For Wilson/clover quarks the scalar channel carries the additive
        # Wilson term. Remove its tree-level piece when forming a Fig.3-like
        # running mass so the momentum dependence is comparable to the paper.
        ratio_gvar = ratio_raw_gvar - tree_level_shift
        a_gvar = jackknife_avg(a_cfg[finite_mask])
        b_gvar = jackknife_avg(b_cfg[finite_mask])

        shell_results.append(
            {
                "shell_key": shell_key,
                "p_norm": p_norm,
                "k_norm": k_norm,
                "tree_level_shift": tree_level_shift,
                "count_total": total_count,
                "count_selected": selected_count,
                "a_cfg": a_cfg,
                "b_cfg": b_cfg,
                "m_raw_gvar": ratio_raw_gvar,
                "m_gvar": ratio_gvar,
                "a_gvar": a_gvar,
                "b_gvar": b_gvar,
            }
        )

    shell_results.sort(key=lambda item: item["k_norm"])
    return shell_results


def kbin_analysis(shell_results, round_decimals=12):
    kbin_results = []
    grouped = {}

    for item in shell_results:
        k_key = round(float(item["k_norm"]), round_decimals)
        grouped.setdefault(k_key, []).append(item)

    for _, items in sorted(grouped.items()):
        weights = np.asarray([item["count_selected"] for item in items], dtype=np.float64)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            continue

        a_cfg_stack = np.stack([item["a_cfg"] for item in items], axis=0)
        b_cfg_stack = np.stack([item["b_cfg"] for item in items], axis=0)
        a_cfg = np.average(a_cfg_stack, axis=0, weights=weights)
        b_cfg = np.average(b_cfg_stack, axis=0, weights=weights)
        finite_mask = np.isfinite(a_cfg) & np.isfinite(b_cfg) & (np.abs(a_cfg) > 1e-12)
        if not finite_mask.any():
            continue

        p_norm = float(np.average([item["p_norm"] for item in items], weights=weights))
        k_norm = float(np.average([item["k_norm"] for item in items], weights=weights))
        tree_level_shift = float(np.average([item["tree_level_shift"] for item in items], weights=weights))
        ratio_raw_gvar = jackknife_ratio(b_cfg[finite_mask], a_cfg[finite_mask])
        ratio_gvar = ratio_raw_gvar - tree_level_shift
        a_gvar = jackknife_avg(a_cfg[finite_mask])
        b_gvar = jackknife_avg(b_cfg[finite_mask])

        kbin_results.append(
            {
                "k_key": k_key,
                "p_norm": p_norm,
                "k_norm": k_norm,
                "tree_level_shift": tree_level_shift,
                "count_selected": int(weight_sum),
                "count_shells": len(items),
                "shell_labels": [item["shell_key"] for item in items],
                "a_cfg": a_cfg,
                "b_cfg": b_cfg,
                "m_raw_gvar": ratio_raw_gvar,
                "m_gvar": ratio_gvar,
                "a_gvar": a_gvar,
                "b_gvar": b_gvar,
            }
        )

    return kbin_results


def save_outputs(
    ensemble,
    momentum_meta,
    greens_by_gamma,
    a_s,
    b_massive,
    as_components,
    shell_results,
    use_cylinder_cut,
):
    os.makedirs("artifacts/data", exist_ok=True)
    os.makedirs("artifacts/plots", exist_ok=True)
    kbin_results = kbin_analysis(shell_results)

    if shell_results:
        shell_p = np.asarray([item["p_norm"] for item in shell_results], dtype=np.float64)
        shell_k = np.asarray([item["k_norm"] for item in shell_results], dtype=np.float64)
        shell_m_raw_mean = np.asarray([gv.mean(item["m_raw_gvar"]) for item in shell_results], dtype=np.float64)
        shell_m_raw_sdev = np.asarray([gv.sdev(item["m_raw_gvar"]) for item in shell_results], dtype=np.float64)
        shell_m_mean = np.asarray([gv.mean(item["m_gvar"]) for item in shell_results], dtype=np.float64)
        shell_m_sdev = np.asarray([gv.sdev(item["m_gvar"]) for item in shell_results], dtype=np.float64)
        shell_a_mean = np.asarray([gv.mean(item["a_gvar"]) for item in shell_results], dtype=np.float64)
        shell_a_sdev = np.asarray([gv.sdev(item["a_gvar"]) for item in shell_results], dtype=np.float64)
        shell_b_mean = np.asarray([gv.mean(item["b_gvar"]) for item in shell_results], dtype=np.float64)
        shell_b_sdev = np.asarray([gv.sdev(item["b_gvar"]) for item in shell_results], dtype=np.float64)
        shell_tree_level_shift = np.asarray([item["tree_level_shift"] for item in shell_results], dtype=np.float64)
        shell_count_total = np.asarray([item["count_total"] for item in shell_results], dtype=np.int32)
        shell_count_selected = np.asarray([item["count_selected"] for item in shell_results], dtype=np.int32)
        shell_labels = np.asarray([item["shell_key"] for item in shell_results], dtype=str)
    else:
        shell_p = np.asarray([], dtype=np.float64)
        shell_k = np.asarray([], dtype=np.float64)
        shell_m_raw_mean = np.asarray([], dtype=np.float64)
        shell_m_raw_sdev = np.asarray([], dtype=np.float64)
        shell_m_mean = np.asarray([], dtype=np.float64)
        shell_m_sdev = np.asarray([], dtype=np.float64)
        shell_a_mean = np.asarray([], dtype=np.float64)
        shell_a_sdev = np.asarray([], dtype=np.float64)
        shell_b_mean = np.asarray([], dtype=np.float64)
        shell_b_sdev = np.asarray([], dtype=np.float64)
        shell_tree_level_shift = np.asarray([], dtype=np.float64)
        shell_count_total = np.asarray([], dtype=np.int32)
        shell_count_selected = np.asarray([], dtype=np.int32)
        shell_labels = np.asarray([], dtype=str)

    if kbin_results:
        kbin_p = np.asarray([item["p_norm"] for item in kbin_results], dtype=np.float64)
        kbin_k = np.asarray([item["k_norm"] for item in kbin_results], dtype=np.float64)
        kbin_m_raw_mean = np.asarray([gv.mean(item["m_raw_gvar"]) for item in kbin_results], dtype=np.float64)
        kbin_m_raw_sdev = np.asarray([gv.sdev(item["m_raw_gvar"]) for item in kbin_results], dtype=np.float64)
        kbin_m_mean = np.asarray([gv.mean(item["m_gvar"]) for item in kbin_results], dtype=np.float64)
        kbin_m_sdev = np.asarray([gv.sdev(item["m_gvar"]) for item in kbin_results], dtype=np.float64)
        kbin_a_mean = np.asarray([gv.mean(item["a_gvar"]) for item in kbin_results], dtype=np.float64)
        kbin_a_sdev = np.asarray([gv.sdev(item["a_gvar"]) for item in kbin_results], dtype=np.float64)
        kbin_b_mean = np.asarray([gv.mean(item["b_gvar"]) for item in kbin_results], dtype=np.float64)
        kbin_b_sdev = np.asarray([gv.sdev(item["b_gvar"]) for item in kbin_results], dtype=np.float64)
        kbin_tree_level_shift = np.asarray([item["tree_level_shift"] for item in kbin_results], dtype=np.float64)
        kbin_count_selected = np.asarray([item["count_selected"] for item in kbin_results], dtype=np.int32)
        kbin_count_shells = np.asarray([item["count_shells"] for item in kbin_results], dtype=np.int32)
        kbin_labels = np.asarray([str(item["shell_labels"]) for item in kbin_results], dtype=str)
    else:
        kbin_p = np.asarray([], dtype=np.float64)
        kbin_k = np.asarray([], dtype=np.float64)
        kbin_m_raw_mean = np.asarray([], dtype=np.float64)
        kbin_m_raw_sdev = np.asarray([], dtype=np.float64)
        kbin_m_mean = np.asarray([], dtype=np.float64)
        kbin_m_sdev = np.asarray([], dtype=np.float64)
        kbin_a_mean = np.asarray([], dtype=np.float64)
        kbin_a_sdev = np.asarray([], dtype=np.float64)
        kbin_b_mean = np.asarray([], dtype=np.float64)
        kbin_b_sdev = np.asarray([], dtype=np.float64)
        kbin_tree_level_shift = np.asarray([], dtype=np.float64)
        kbin_count_selected = np.asarray([], dtype=np.int32)
        kbin_count_shells = np.asarray([], dtype=np.int32)
        kbin_labels = np.asarray([], dtype=str)

    cache_dict = {
        "momentum_list": momentum_meta["momentum_array"],
        "momentum_label": momentum_meta["momentum_label"],
        "orbit_id": momentum_meta["orbit_labels"],
        "shell_id": momentum_meta["shell_labels"],
        "kvec_norm": momentum_meta["kvec_norm"],
        "k4_momentum": momentum_meta["k4_values"],
        "k_spatial": momentum_meta["k_spatial"],
        "pvec_norm": momentum_meta["pvec_norm"],
        "p4_momentum": momentum_meta["p4_values"],
        "mask_cylinder": momentum_meta["mask_cylinder"],
        "wilson_spatial": momentum_meta["wilson_spatial"],
        "wilson_full": momentum_meta["wilson_full"],
        "A_s_point": a_s,
        "B_point": b_massive,
        "A_s_component": as_components,
        "analysis_used_cylinder_cut": np.asarray(use_cylinder_cut, dtype=bool),
        "shell_p_norm": shell_p,
        "shell_k_norm": shell_k,
        "shell_M_bare_mean": shell_m_raw_mean,
        "shell_M_bare_sdev": shell_m_raw_sdev,
        "shell_M_raw_mean": shell_m_raw_mean,
        "shell_M_raw_sdev": shell_m_raw_sdev,
        "shell_M_wilson_corrected_mean": shell_m_mean,
        "shell_M_wilson_corrected_sdev": shell_m_sdev,
        "shell_M_mean": shell_m_mean,
        "shell_M_sdev": shell_m_sdev,
        "shell_A_mean": shell_a_mean,
        "shell_A_sdev": shell_a_sdev,
        "shell_B_mean": shell_b_mean,
        "shell_B_sdev": shell_b_sdev,
        "shell_tree_level_shift": shell_tree_level_shift,
        "shell_count_total": shell_count_total,
        "shell_count_selected": shell_count_selected,
        "shell_label": shell_labels,
        "kbin_p_norm": kbin_p,
        "kbin_k_norm": kbin_k,
        "kbin_M_bare_mean": kbin_m_raw_mean,
        "kbin_M_bare_sdev": kbin_m_raw_sdev,
        "kbin_M_mean": kbin_m_mean,
        "kbin_M_sdev": kbin_m_sdev,
        "kbin_A_mean": kbin_a_mean,
        "kbin_A_sdev": kbin_a_sdev,
        "kbin_B_mean": kbin_b_mean,
        "kbin_B_sdev": kbin_b_sdev,
        "kbin_tree_level_shift": kbin_tree_level_shift,
        "kbin_count_selected": kbin_count_selected,
        "kbin_count_shells": kbin_count_shells,
        "kbin_label": kbin_labels,
    }
    for gamma_name, greens in greens_by_gamma.items():
        cache_dict[gamma_name] = np.asarray(greens)

    cache_path = f"artifacts/data/qprop_fig3_mass_{ensemble}.npz"
    np.savez(cache_path, **cache_dict)
    print(f"Cached to {cache_path}")
    if len(kbin_k) > 0:
        print(
            "Wilson-corrected M(|k|) range after |k|-binning = "
            f"[{kbin_m_mean.min():.4f}, {kbin_m_mean.max():.4f}] in a^-1"
        )

    fig, ax = default_plot()
    if len(kbin_k) > 0:
        ax.errorbar(
            kbin_k,
            kbin_m_mean,
            yerr=kbin_m_sdev,
            marker="o",
            label=ensemble + (" + cylinder cut" if use_cylinder_cut else ""),
            **errorb,
        )
    ax.set_xlabel(r"$|k|$", **fs_p)
    ax.set_ylabel(r"$M(|k|)$ [$a^{-1}$]", **fs_p)
    if len(kbin_k) > 0:
        ax.legend(**fs_small_p)
    fig.tight_layout()
    plot_path = f"artifacts/plots/qprop_fig3_mass_{ensemble}.pdf"
    fig.savefig(plot_path, transparent=True)
    plt.close(fig)
    print(f"Saved {plot_path}")


def maybe_compare_with_existing_greens(ensemble, momentum_meta, greens_by_gamma):
    cache_path = Path(f"artifacts/data/qprop_greens_{ensemble}.npz")
    if not cache_path.exists():
        return

    cache = np.load(cache_path)
    if "I" not in cache:
        return

    existing_labels = np.asarray(cache["momentum_label"], dtype=str)
    target_label = "(0,0,0,0)"
    old_match = np.where(existing_labels == target_label)[0]
    new_match = np.where(momentum_meta["momentum_label"] == target_label)[0]
    if len(old_match) == 0 or len(new_match) == 0:
        return

    old_zero = np.asarray(cache["I"])[:, old_match[0]]
    new_zero = np.asarray(greens_by_gamma["I"])[:, new_match[0]]
    n_common = min(len(old_zero), len(new_zero))
    diff = new_zero[:n_common] - old_zero[:n_common]
    print(
        "Zero-mode cross-check with existing qprop_greens cache:",
        jackknife_avg(np.real(diff)),
    )


def main():
    args = parse_args()

    if not os.path.exists(".cache"):
        os.makedirs(".cache")
        print("Created .cache directory for PyQUDA resources")

    ensemble_meta = ENSEMBLE_META[args.ensemble]
    latt_size = ensemble_meta["latt_size"]

    cfg_files = build_cfg_file_list(args.ensemble, args.n_conf)
    print(f"Using {len(cfg_files)} configurations from {ensemble_meta['gauge_dir']}")

    momentum_meta = build_momentum_grid(latt_size, args.max_n2)
    momentum_list = momentum_meta["momentum_array"].tolist()
    print(
        f"Built {len(momentum_list)} 4-momenta"
        f" ({momentum_meta['mask_cylinder'].sum()} survive the cylinder cut)"
    )

    init([1, 1, 1, 1], resource_path=".cache")
    latt_info = core.LatticeInfo(latt_size, -1, XI_0 / NU)
    dirac = core.getClover(latt_info, MASS, 1e-8, 10000, XI_0, CSW_R, CSW_T, MULTIGRID)
    is_root = latt_info.mpi_rank == 0

    gamma_ops = [
        ("I", gamma.gamma(0)),
        ("gX", gamma.gamma(1)),
        ("gY", gamma.gamma(2)),
        ("gZ", gamma.gamma(4)),
        ("gT", gamma.gamma(8)),
    ]
    mom_phase = MomentumPhase(latt_info)
    point_quark_greens_by_gamma = {name: [] for name, _ in gamma_ops}

    for cfg_path in tqdm(cfg_files, desc="Processing configurations", disable=not is_root):
        gauge = io.readNERSCGauge(str(cfg_path))

        with dirac.useGauge(gauge):
            point_source = source.propagator(latt_info, "point", [0, 0, 0, 0])
            point_propag = core.invertPropagator(dirac, point_source)

            if is_root:
                cfg_greens_by_gamma = {name: [] for name, _ in gamma_ops}

            for start in range(0, len(momentum_list), args.phase_chunk_size):
                momentum_chunk = momentum_list[start : start + args.phase_chunk_size]
                momentum_phases = mom_phase.getPhases(momentum_chunk, x0=[0, 0, 0, 0])

                for gamma_name, gamma_matrix in gamma_ops:
                    point_quark_green = contract(
                        "pwtzyx,wtzyxijaa,ji->p",
                        momentum_phases,
                        point_propag.data,
                        gamma_matrix,
                    ).get()
                    if is_root:
                        cfg_greens_by_gamma[gamma_name].append(point_quark_green)

            if is_root:
                for gamma_name in cfg_greens_by_gamma:
                    point_quark_greens_by_gamma[gamma_name].append(
                        np.concatenate(cfg_greens_by_gamma[gamma_name], axis=0)
                    )

    if not is_root:
        return

    for gamma_name in point_quark_greens_by_gamma:
        point_quark_greens_by_gamma[gamma_name] = np.asarray(point_quark_greens_by_gamma[gamma_name])

    a_s, b_massive, as_components = compute_as_from_traces(point_quark_greens_by_gamma, momentum_meta)
    use_cylinder_cut = not args.no_cylinder_cut
    shell_results = shell_analysis(a_s, b_massive, momentum_meta, use_cylinder_cut=use_cylinder_cut)

    print("max |Im Tr[S]| =", np.max(np.abs(np.imag(point_quark_greens_by_gamma["I"]))))
    print(
        "max |Re Tr[gamma_i S]| =",
        max(
            np.max(np.abs(np.real(point_quark_greens_by_gamma["gX"]))),
            np.max(np.abs(np.real(point_quark_greens_by_gamma["gY"]))),
            np.max(np.abs(np.real(point_quark_greens_by_gamma["gZ"]))),
        ),
    )
    a_spread, b_spread = summarize_orbit_spread(a_s, b_massive, momentum_meta)
    print(f"median relative p4-spread of A_s over cubic/parity orbits = {a_spread:.3e}")
    print(f"median relative p4-spread of B over cubic/parity orbits = {b_spread:.3e}")
    print(f"Built {len(shell_results)} shell averages for M(|p|)")

    maybe_compare_with_existing_greens(args.ensemble, momentum_meta, point_quark_greens_by_gamma)
    save_outputs(
        args.ensemble,
        momentum_meta,
        point_quark_greens_by_gamma,
        a_s,
        b_massive,
        as_components,
        shell_results,
        use_cylinder_cut=use_cylinder_cut,
    )


if __name__ == "__main__":
    main()
