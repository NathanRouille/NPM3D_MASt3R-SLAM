from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from compute_ate import compute_ate

DEFAULT_SEQUENCES = [
    "rgbd_dataset_freiburg1_desk",
    "rgbd_dataset_freiburg1_room",
    "rgbd_dataset_freiburg2_xyz",
    "rgbd_dataset_freiburg3_long_office_household",
]

SEQ_SHORT = {
    "rgbd_dataset_freiburg1_desk": "fr1/desk",
    "rgbd_dataset_freiburg1_room": "fr1/room",
    "rgbd_dataset_freiburg2_xyz": "fr2/xyz",
    "rgbd_dataset_freiburg3_long_office_household": "fr3/office",
}

ERROR_FORMULATIONS = ["ray", "point"]


def make_error_config(base_cfg_path, error_formulation, tmp_dir, point_subsample = 4):
    tmp_path = tmp_dir / f"error_{error_formulation}.yaml"
    cfg: dict = {
        "inherit": base_cfg_path,
        "tracking": {
            "filtering_mode": "weighted_pointmap",
            "error_formulation": error_formulation,
        },
        "local_opt": {
            "error_formulation": error_formulation,
        },
    }
    if error_formulation == "point":
        cfg["local_opt"]["point_subsample"] = point_subsample
    with open(tmp_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return tmp_path


def run_slam(dataset_path, config_path, run_name, python_bin = "python"):
    cmd = [
        python_bin,
        "main.py",
        "--dataset", str(dataset_path),
        "--config",  str(config_path),
        "--save-as", run_name,
        "--no-viz",
    ]
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
    parser = argparse.ArgumentParser(description="Run MASt3R-SLAM error formulation ablation (Option C)")
    parser.add_argument("--dataset_root", default="datasets/tum")
    parser.add_argument("--gt_root", default="groundtruths/tum")
    parser.add_argument("--save_root", default="results/error_ablation")
    parser.add_argument("--base_config", default="config/eval_no_calib.yaml")
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--formulations", nargs="+", default=ERROR_FORMULATIONS)
    parser.add_argument("--point_subsample", type=int, default=4, help="Pixel subsampling for the Python point-error backend")
    parser.add_argument("--traj_filename", default="traj_est.txt")
    parser.add_argument("--python", default="python")
    parser.add_argument("--skip_run", action="store_true")
    args = parser.parse_args()

    save_root = Path(args.save_root)
    dataset_root = Path(args.dataset_root)
    gt_root = Path(args.gt_root)
    save_root.mkdir(parents=True, exist_ok=True)

    results: dict = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        for formulation in args.formulations:
            results[formulation] = {}
            config_path = make_error_config(
                args.base_config, formulation, tmp_dir, args.point_subsample
            )
            print(f"Error formulation: {formulation}")

            for seq in args.sequences:
                dataset_path = dataset_root / seq
                save_path = save_root / formulation / seq
                short_name = SEQ_SHORT.get(seq, seq)

                if not dataset_path.exists():
                    print(f"Dataset not found: {dataset_path}")
                    results[formulation][short_name] = None
                    continue

                if not args.skip_run:
                    run_name = formulation
                    success = run_slam(
                        dataset_path, config_path, run_name, args.python
                    )
                    if not success:
                        results[formulation][short_name] = None
                        continue

                traj_est = find_trajectory(run_name, dataset_path)
                if traj_est is None:
                    print(f"Trajectory not found in {save_path}")
                    results[formulation][short_name] = None
                    continue

                # Ground truth
                gt_file = dataset_path / "groundtruth.txt"
                if not gt_file.exists():
                    print(f"Ground-truth not found for {seq}")
                    results[formulation][short_name] = None
                    continue

                try:
                    sys.path.insert(0, str(Path(__file__).parent))
                    ate = compute_ate(traj_est, gt_file, traj_format="tum", align=True, correct_scale=True)
                    rmse = ate["rmse"]
                    print(f"{short_name:20s}  ATE RMSE = {rmse:.4f} m")
                    results[formulation][short_name] = rmse
                except Exception as exc:
                    print(f"ATE computation failed for {seq}: {exc}")
                    results[formulation][short_name] = None

    out_json = save_root / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # Summary
    print(f"Ray vs Point Error (ATE RMSE [m])")
    all_seqs = sorted({s for r in results.values() for s in r})
    header = f"{'Formulation':>20}"
    for s in all_seqs:
        header += f"{s:>12}"
    print(header)
    for form, form_r in results.items():
        row = f"{form:>20}"
        for s in all_seqs:
            v = form_r.get(s)
            row += f"{v:>12.4f}" if v is not None else f"{'N/A':>12}"
        print(row)


if __name__ == "__main__":
    main()
