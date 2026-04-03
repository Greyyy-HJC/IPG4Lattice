#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ipg_utils import (
    cfg_subset_from_args,
    default_ipg_name,
    determinant_error,
    ensure_pyquda_initialized,
    extract_cfg_index,
    gauge_max_abs_diff,
    list_gauge_files,
    make_ipg_gauge,
    parse_cfg_list,
    projected_temporal_links,
    projected_temporal_spread,
    read_nersc_gauge,
    target_deviation,
    target_residual_error,
    transform_determinant_error,
    transform_unitarity_error,
    unitary_error,
    write_nersc_gauge,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CG+IPG ensembles from the Appendix A definition.")
    parser.add_argument("--cg-dir", required=True, type=Path, help="Directory containing original CG-fixed gauges.")
    parser.add_argument("--ipg-dir", required=True, type=Path, help="Directory containing CG+IPG gauges.")
    parser.add_argument("--glob", default="wilson_b6.cg.*", help="Filename glob for the CG input ensemble.")
    parser.add_argument("--n-conf", type=int, default=None, help="Limit validation to the first N matched configurations.")
    parser.add_argument("--cfg-list", default=None, help="Comma-separated configuration indices to validate.")
    parser.add_argument(
        "--projection-method",
        default="cabibbo-marinari",
        help="SU(3) projection method used for the Appendix A temporal-link projection.",
    )
    parser.add_argument(
        "--spread-tol",
        type=float,
        default=1e-10,
        help="Allowed spread of projected temporal links after IPG.",
    )
    parser.add_argument(
        "--boundary-tol",
        type=float,
        default=1e-10,
        help="Allowed periodic-closure error in the recursive g(t) construction.",
    )
    parser.add_argument(
        "--reconstruct-tol",
        type=float,
        default=1e-10,
        help="Allowed max absolute link difference between the written IPG gauge and the reconstructed one.",
    )
    parser.add_argument(
        "--repair-spread",
        action="store_true",
        help="If post_spread exceeds tolerance, rebuild and overwrite the failing CG+IPG gauge.",
    )
    parser.add_argument(
        "--repair-max-iters",
        type=int,
        default=3,
        help="Maximum number of repeated IPG passes used when repairing a large post_spread.",
    )
    return parser.parse_args()


def build_ipg_reference(gauge, projection_method: str, spread_tol: float, max_iters: int):
    if max_iters < 1:
        raise ValueError("max_iters must be at least 1")

    current = gauge
    result = None
    cumulative_transform = None
    iterations = 0
    for iterations in range(1, max_iters + 1):
        result = make_ipg_gauge(current, projection_method=projection_method)
        if cumulative_transform is None:
            cumulative_transform = result.transform.copy()
        else:
            cumulative_transform = result.transform @ cumulative_transform
        if result.transformed_spread <= spread_tol:
            break
        current = result.gauge

    assert result is not None
    assert cumulative_transform is not None
    return result, iterations, cumulative_transform


def main() -> int:
    args = parse_args()
    ensure_pyquda_initialized()
    ipg_dir = args.ipg_dir.resolve()
    output_root = ipg_dir.parent if ipg_dir.name == "gauge" else ipg_dir
    transform_dir = output_root / "ipg_transform"

    cg_files = list_gauge_files(args.cg_dir.resolve(), args.glob)
    cfg_list = parse_cfg_list(args.cfg_list)
    cg_files = cfg_subset_from_args(cg_files, cfg_list, args.n_conf)
    if not cg_files:
        raise SystemExit("No CG configurations selected for validation")

    global_metrics = {
        "pre_spread": 0.0,
        "post_spread": 0.0,
        "boundary": 0.0,
        "residual": 0.0,
        "target_dev": 0.0,
        "reconstruct": 0.0,
        "target_unitarity": 0.0,
        "target_det": 0.0,
        "polyakov_unitarity": 0.0,
        "polyakov_det": 0.0,
        "transform_unitarity": 0.0,
        "transform_det": 0.0,
        "logm_err": 0.0,
    }
    failures = []
    repaired = []

    for cg_path in cg_files:
        cfg = extract_cfg_index(cg_path)
        ipg_path = ipg_dir / default_ipg_name(cg_path.name)
        if not ipg_path.exists():
            raise SystemExit(f"Missing CG+IPG gauge for cfg {cfg}: {ipg_path}")

        print("\n" + "=" * 20)
        print(f"[cfg {cfg}] reading pair")
        cg_gauge = read_nersc_gauge(cg_path)
        ipg_gauge = read_nersc_gauge(ipg_path)

        cg_lexico = cg_gauge.lexico().copy()
        ipg_lexico = ipg_gauge.lexico().copy()
        result = make_ipg_gauge(cg_gauge, projection_method=args.projection_method)
        reference_result, reference_iters, reference_transform = build_ipg_reference(
            cg_gauge,
            projection_method=args.projection_method,
            spread_tol=args.spread_tol,
            max_iters=args.repair_max_iters,
        )

        pre_spread = projected_temporal_spread(cg_lexico, projection_method=args.projection_method)
        post_spread = projected_temporal_spread(ipg_lexico, projection_method=args.projection_method)
        post_projected = projected_temporal_links(ipg_lexico, projection_method=args.projection_method)

        residual = target_residual_error(result.projected_temporal_links, result.transform, result.target_matrix)
        target_dev = target_deviation(post_projected, reference_result.target_matrix)
        reconstruct = gauge_max_abs_diff(reference_result.gauge.lexico(), ipg_lexico)
        target_unitarity = unitary_error(result.target_matrix)
        target_det = determinant_error(result.target_matrix)
        polyakov_unitarity = unitary_error(result.polyakov_matrix)
        polyakov_det = determinant_error(result.polyakov_matrix)
        transform_unitarity = transform_unitarity_error(result.transform)
        transform_det = transform_determinant_error(result.transform)

        print("\n" + "-" * 20)
        print(
            f"[cfg {cfg}] pre_spread={pre_spread:.3e} post_spread={post_spread:.3e} "
            f"boundary={result.boundary_error:.3e} residual={residual:.3e} "
            f"target_dev={target_dev:.3e} reconstruct={reconstruct:.3e}"
        )
        print("\n" + "-" * 20)
        print(
            f"[cfg {cfg}] target_u={target_unitarity:.3e} target_det={target_det:.3e} "
            f"polyakov_u={polyakov_unitarity:.3e} polyakov_det={polyakov_det:.3e} "
            f"transform_u={transform_unitarity:.3e} transform_det={transform_det:.3e} "
            f"logm_err={result.logm_error:.3e} ref_passes={reference_iters}"
        )

        if args.repair_spread and post_spread > args.spread_tol:
            print("\n" + "-" * 20)
            print(
                f"[cfg {cfg}] post_spread={post_spread:.3e} exceeds tolerance {args.spread_tol:.3e}; "
                f"rewriting from CG with up to {args.repair_max_iters} IPG pass(es)"
            )
            write_nersc_gauge(ipg_path, reference_result.gauge)
            transform_dir.mkdir(parents=True, exist_ok=True)
            np.save(transform_dir / f"{cg_path.name}.npy", reference_transform)
            ipg_gauge = read_nersc_gauge(ipg_path)
            ipg_lexico = ipg_gauge.lexico().copy()
            post_spread = projected_temporal_spread(ipg_lexico, projection_method=args.projection_method)
            post_projected = projected_temporal_links(ipg_lexico, projection_method=args.projection_method)
            target_dev = target_deviation(post_projected, reference_result.target_matrix)
            reconstruct = gauge_max_abs_diff(reference_result.gauge.lexico(), ipg_lexico)
            repaired.append((cfg, reference_iters, post_spread))
            print("\n" + "-" * 20)
            print(
                f"[cfg {cfg}] repaired post_spread={post_spread:.3e} "
                f"reconstruct={reconstruct:.3e} passes={reference_iters}"
            )

        metrics = {
            "pre_spread": pre_spread,
            "post_spread": post_spread,
            "boundary": result.boundary_error,
            "residual": residual,
            "target_dev": target_dev,
            "reconstruct": reconstruct,
            "target_unitarity": target_unitarity,
            "target_det": target_det,
            "polyakov_unitarity": polyakov_unitarity,
            "polyakov_det": polyakov_det,
            "transform_unitarity": transform_unitarity,
            "transform_det": transform_det,
            "logm_err": result.logm_error,
        }

        for key, value in metrics.items():
            global_metrics[key] = max(global_metrics[key], value)

        if (
            post_spread > args.spread_tol
            or result.boundary_error > args.boundary_tol
            or residual > args.boundary_tol
            or reconstruct > args.reconstruct_tol
        ):
            failures.append((cfg, post_spread, result.boundary_error, residual, reconstruct))

    print("\n" + "-" * 20)
    print("Global maxima:")
    print(
        "  "
        + " ".join(
            f"{key}={value:.3e}"
            for key, value in (
                ("pre_spread", global_metrics["pre_spread"]),
                ("post_spread", global_metrics["post_spread"]),
                ("boundary", global_metrics["boundary"]),
                ("residual", global_metrics["residual"]),
                ("target_dev", global_metrics["target_dev"]),
                ("reconstruct", global_metrics["reconstruct"]),
                ("target_u", global_metrics["target_unitarity"]),
                ("target_det", global_metrics["target_det"]),
                ("polyakov_u", global_metrics["polyakov_unitarity"]),
                ("polyakov_det", global_metrics["polyakov_det"]),
                ("transform_u", global_metrics["transform_unitarity"]),
                ("transform_det", global_metrics["transform_det"]),
                ("logm_err", global_metrics["logm_err"]),
            )
        )
    )
    if failures:
        for cfg, post_spread, boundary, residual, reconstruct in failures:
            print(
                f"FAIL cfg={cfg} post_spread={post_spread:.3e} "
                f"boundary={boundary:.3e} residual={residual:.3e} reconstruct={reconstruct:.3e}"
            )
        return 1

    if repaired:
        for cfg, passes, post_spread in repaired:
            print(f"REPAIRED cfg={cfg} passes={passes} post_spread={post_spread:.3e}")

    print("\nAll configurations passed validation.\n")
    print("\n" + "=" * 20)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
