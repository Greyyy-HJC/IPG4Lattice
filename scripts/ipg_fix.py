#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ipg_utils import (
    default_ipg_name,
    ensure_pyquda_initialized,
    extract_cfg_index,
    list_gauge_files,
    make_ipg_gauge,
    read_nersc_gauge,
    write_nersc_gauge,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Integrated Polyakov gauge on top of CG-fixed NERSC ensembles.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing CG-fixed NERSC gauge files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write CG+IPG NERSC gauge files.")
    parser.add_argument("--glob", default="wilson_b6.cg.*", help="Filename glob for input gauge files.")
    parser.add_argument("--cfg-start", type=int, default=None, help="Lowest configuration index to include.")
    parser.add_argument("--cfg-stop", type=int, default=None, help="Highest configuration index to include.")
    parser.add_argument("--save-transform", action="store_true", help="Save g(t) as .npy under ../ipg_transform.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument(
        "--projection-method",
        default="cabibbo-marinari",
        choices=["cabibbo-marinari", "polar"],
        help="Method used to project averaged temporal links back to SU(3).",
    )
    parser.add_argument(
        "--verify-readback",
        action="store_true",
        help="Re-read every written gauge file to verify NERSC I/O.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_pyquda_initialized()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_root = output_dir.parent if output_dir.name == "gauge" else output_dir
    transform_dir = output_root / "ipg_transform"

    gauge_files = list_gauge_files(input_dir, args.glob, args.cfg_start, args.cfg_stop)
    if not gauge_files:
        raise SystemExit(f"No input gauge files matched {args.glob} in {input_dir}")

    print(f"Found {len(gauge_files)} input configurations in {input_dir}")
    for gauge_path in gauge_files:
        cfg = extract_cfg_index(gauge_path)
        output_path = output_dir / default_ipg_name(gauge_path.name)
        if output_path.exists() and not args.overwrite:
            print(f"[cfg {cfg}] skip existing {output_path.name}")
            continue

        print(f"[cfg {cfg}] reading {gauge_path.name}")
        gauge = read_nersc_gauge(gauge_path)
        result = make_ipg_gauge(gauge, projection_method=args.projection_method)

        print(
            f"[cfg {cfg}] spread={result.transformed_spread:.3e} "
            f"boundary={result.boundary_error:.3e} logm_err={result.logm_error:.3e}"
        )
        write_nersc_gauge(output_path, result.gauge)

        if args.save_transform:
            transform_dir.mkdir(parents=True, exist_ok=True)
            np.save(transform_dir / f"{gauge_path.name}.npy", result.transform)

        if args.verify_readback:
            _ = read_nersc_gauge(output_path)
            print(f"[cfg {cfg}] verified readback {output_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
