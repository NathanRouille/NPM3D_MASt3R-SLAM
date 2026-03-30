from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from compute_ate import compute_ate

# Default TUM sequences
DEFAULT_SEQUENCES = [
    "rgbd_dataset_freiburg1_desk",
    "rgbd_dataset_freiburg1_room",
    "rgbd_dataset_freiburg2_xyz",
    "rgbd_dataset_freiburg3_long_office_household",
]

FUSION_MODES = ["recent", "first", "best_score", "weighted_pointmap"] # best_score is the same as median in the paper and my report

# Map dataset folder name to short name for reporting
SEQ_SHORT = {
    "rgbd_dataset_freiburg1_desk": "fr1/desk",
    "rgbd_dataset_freiburg1_room": "fr1/room",
    "rgbd_dataset_freiburg2_xyz": "fr2/xyz",
    "rgbd_dataset_freiburg3_long_office_household": "fr3/office",
}


def make_fusion_config(base_cfg_path, filtering_mode, tmp_dir):
    """Write a temporary config that inherits eval_no_calib and overrides filtering_mode and error_formulation"""
    tmp_path = tmp_dir / f"fusion_{filtering_mode}.yaml"
    cfg = {
        "inherit": base_cfg_path,
        "tracking": {
            "filtering_mode": filtering_mode,
            "error_formulation": "ray",  # keep error formulation fixed
            **({"filtering_score": "median"} if filtering_mode == "best_score" else {}),
        },
        "local_opt": {
            "error_formulation": "ray",
        },
    }
    with open(tmp_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return tmp_path


def run_slam(dataset_path, config_path, run_name, python_bin = "python"):
    cmd = [
        python_bin,
        "main.py",
        "--dataset", str(dataset_path),
        "--config", str(config_path),
        "--save-as", run_name,
        "--no-viz",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"main.py exited with code {result.returncode}")
        return False
    return True


def find_trajectory(run_name, dataset_path):
    traj = Path("logs") / run_name / f"{Path(dataset_path).stem}.txt"
    if traj.exists():
        return traj
    return None


def main():
    parser = argparse.ArgumentParser(description="Run MASt3R-SLAM fusion ablation (Option B)")
    parser.add_argument("--dataset_root", default="datasets/tum",
                        help="Root directory containing TUM sequence folders")
    parser.add_argument("--gt_root", default="groundtruths/tum",
                        help="Root directory containing ground-truth files")
    parser.add_argument("--save_root", default="results/fusion_ablation",
                        help="Output root directory")
    parser.add_argument("--base_config", default="config/eval_no_calib.yaml",
                        help="Base config to inherit from")
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES,
                        help="Sequence folder names under dataset_root")
    parser.add_argument("--modes", nargs="+", default=FUSION_MODES,
                        help="Fusion modes to evaluate")
    parser.add_argument("--traj_filename", default="traj_est.txt",
                        help="Trajectory filename saved by main.py")
    parser.add_argument("--python", default="python",
                        help="Python interpreter to use")
    parser.add_argument("--skip_run", action="store_true",
                        help="Skip running SLAM; only aggregate existing results")
    args = parser.parse_args()

    save_root = Path(args.save_root)
    dataset_root = Path(args.dataset_root)
    gt_root = Path(args.gt_root)
    save_root.mkdir(parents=True, exist_ok=True)

    results: dict = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        for mode in args.modes:
            results[mode] = {}
            config_path = make_fusion_config(args.base_config, mode, tmp_dir)
            print(f"Fusion mode: {mode}")

            for seq in args.sequences:
                dataset_path = dataset_root / seq
                save_path = save_root / mode / seq
                short_name = SEQ_SHORT.get(seq, seq)

                if not dataset_path.exists():
                    print(f"Dataset not found: {dataset_path}")
                    results[mode][short_name] = None
                    continue

                if not args.skip_run:
                    run_name = mode
                    success = run_slam(
                        dataset_path, config_path, run_name, args.python
                    )
                    if not success:
                        results[mode][short_name] = None
                        continue

                # Find trajectory & compute ATE
                traj_est = find_trajectory(run_name, dataset_path)
                if traj_est is None:
                    print(f"Trajectory not found in {save_path}")
                    results[mode][short_name] = None
                    continue

                # Find ground truth
                gt_file = dataset_path / "groundtruth.txt"
                if not gt_file.exists():
                    print(f"Ground-truth not found for {seq}")
                    results[mode][short_name] = None
                    continue

                try:
                    sys.path.insert(0, str(Path(__file__).parent))
                    ate = compute_ate(traj_est, gt_file, traj_format="tum", align=True, correct_scale=True)
                    rmse = ate["rmse"]
                    print(f"  {short_name:20s} ATE RMSE = {rmse:.4f} m")
                    results[mode][short_name] = rmse
                except Exception as exc:
                    print(f"ATE computation failed for {seq}: {exc}")
                    results[mode][short_name] = None

    # Save aggregated results
    out_json = save_root / "results.json"
    cleaned = {m: {s: v for s, v in seqs.items() if v is not None} for m, seqs in results.items()}
    with open(out_json, "w") as f:
        json.dump(cleaned, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # Summary
    print(f"{'Mode':<24}", end="")
    all_seqs = list({s for mode_r in results.values() for s in mode_r})
    for s in all_seqs:
        short = SEQ_SHORT.get(s, s)
        print(f"  {short:>12}", end="")
    for mode, mode_r in results.items():
        print(f"{mode:<24}", end="")
        for s in all_seqs:
            v = mode_r.get(s)
            print(f"{v:>12.4f}" if v is not None else f"{'N/A':>12}", end="")


if __name__ == "__main__":
    main()
