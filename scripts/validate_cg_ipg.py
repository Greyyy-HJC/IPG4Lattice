#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from ipg_utils import (
    PropagatorParams,
    build_dirac,
    cfg_subset_from_args,
    compare_observables,
    current_spatial_line_observable_with_dirac,
    default_ipg_name,
    ensure_pyquda_initialized,
    extract_cfg_index,
    list_gauge_files,
    parse_cfg_list,
    read_nersc_gauge,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate that CG and CG+IPG propagator slices agree.")
    parser.add_argument("--cg-dir", required=True, type=Path, help="Directory containing original CG-fixed gauges.")
    parser.add_argument("--ipg-dir", required=True, type=Path, help="Directory containing CG+IPG gauges.")
    parser.add_argument("--glob", default="wilson_b6.cg.*", help="Filename glob for the CG input ensemble.")
    parser.add_argument("--n-conf", type=int, default=None, help="Limit validation to the first N matched configurations.")
    parser.add_argument("--cfg-list", default=None, help="Comma-separated configuration indices to validate.")
    parser.add_argument("--mass", type=float, default=PropagatorParams.mass, help="Clover mass used in the propagator.")
    parser.add_argument("--tol", type=float, default=1e-8, help="Allowed max absolute difference.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_pyquda_initialized()

    params = PropagatorParams(mass=args.mass)
    latt_info, dirac = build_dirac(params)
    cg_files = list_gauge_files(args.cg_dir.resolve(), args.glob)
    cfg_list = parse_cfg_list(args.cfg_list)
    cg_files = cfg_subset_from_args(cg_files, cfg_list, args.n_conf)
    if not cg_files:
        raise SystemExit("No CG configurations selected for validation")

    global_max_abs = 0.0
    global_rel = 0.0
    failures = []

    for cg_path in cg_files:
        cfg = extract_cfg_index(cg_path)
        ipg_path = args.ipg_dir.resolve() / default_ipg_name(cg_path.name)
        if not ipg_path.exists():
            raise SystemExit(f"Missing CG+IPG gauge for cfg {cfg}: {ipg_path}")

        print(f"[cfg {cfg}] reading pair")
        cg_gauge = read_nersc_gauge(cg_path)
        ipg_gauge = read_nersc_gauge(ipg_path)

        cg_obs = current_spatial_line_observable_with_dirac(cg_gauge, latt_info, dirac)
        ipg_obs = current_spatial_line_observable_with_dirac(ipg_gauge, latt_info, dirac)
        max_abs, rel = compare_observables(cg_obs, ipg_obs)

        global_max_abs = max(global_max_abs, max_abs)
        global_rel = max(global_rel, rel)
        print(f"[cfg {cfg}] max_abs={max_abs:.3e} rel={rel:.3e}")
        if max_abs > args.tol:
            failures.append((cfg, max_abs, rel))

    print(f"Global max_abs={global_max_abs:.3e} global_rel={global_rel:.3e}")
    if failures:
        for cfg, max_abs, rel in failures:
            print(f"FAIL cfg={cfg} max_abs={max_abs:.3e} rel={rel:.3e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
