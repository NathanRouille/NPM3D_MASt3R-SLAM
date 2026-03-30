from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np


def compute_ate(traj_est_path, traj_ref_path, traj_format = "tum", align = True, correct_scale = True):
    """Compute Absolute Trajectory Error (ATE / APE-translation)"""
    try:
        from evo.tools import file_interface
        from evo.core import sync, metrics
        from evo.core.trajectory import PoseTrajectory3D
    except ImportError:
        raise ImportError("Please install evo:  pip install evo")

    traj_est_path = Path(traj_est_path)
    traj_ref_path = Path(traj_ref_path)

    if not traj_est_path.exists():
        raise FileNotFoundError(f"Estimated trajectory not found: {traj_est_path}")
    if not traj_ref_path.exists():
        raise FileNotFoundError(f"Reference trajectory not found: {traj_ref_path}")

    # Load trajectories
    if traj_format == "tum":
        traj_est = file_interface.read_tum_trajectory_file(str(traj_est_path))
        traj_ref = file_interface.read_tum_trajectory_file(str(traj_ref_path))
    elif traj_format == "euroc":
        traj_est = file_interface.read_euroc_csv_trajectory(str(traj_est_path))
        traj_ref = file_interface.read_euroc_csv_trajectory(str(traj_ref_path))
    else:
        raise ValueError(f"Unknown trajectory format: {traj_format}")

    # Temporal association
    max_diff = 0.02  # seconds
    traj_ref_sync, traj_est_sync = sync.associate_trajectories(
        traj_ref, traj_est, max_diff=max_diff
    )

    # Alignment (Umeyama + optional scale)
    if align:
        traj_est_sync.align(traj_ref_sync, correct_scale=correct_scale, correct_only_scale=False)

    # APE with translation part
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((traj_ref_sync, traj_est_sync))
    stats = ape_metric.get_all_statistics()

    return {
        "rmse": stats["rmse"],
        "mean": stats["mean"],
        "median": stats["median"],
        "std": stats["std"],
        "min": stats["min"],
        "max": stats["max"],
        "n_poses": len(traj_est_sync.timestamps),
    }


def find_trajectory(output_dir):
    output_dir = Path(output_dir)
    candidates = [
        output_dir / "traj_est.txt",
        output_dir / "trajectory.txt",
        output_dir / "stamped_traj_estimate.txt",
        output_dir / "results" / "traj_est.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="Compute ATE for a single run")
    parser.add_argument("--est", required=True, help="Estimated trajectory file")
    parser.add_argument("--ref", required=True, help="Ground-truth trajectory file")
    parser.add_argument("--format", default="tum", choices=["tum", "euroc"],
                        help="Trajectory file format")
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument("--no-scale-correction", action="store_true")
    args = parser.parse_args()

    result = compute_ate(args.est, args.ref, traj_format=args.format, align=not args.no_align, correct_scale=not args.no_scale_correction)

    print(f"ATE results  ({args.est})")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"{k:8s}: {v:.4f} m")
        else:
            print(f"{k:8s}: {v}")

if __name__ == "__main__":
    main()
