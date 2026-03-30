from __future__ import annotations
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import yaml
from compute_ate import compute_ate


DEFAULT_7SCENES   = ["heads", "redkitchen"]
DEFAULT_TUM       = [
    "rgbd_dataset_freiburg1_desk",
    "rgbd_dataset_freiburg1_room",
]
TUM_SHORT = {
    "rgbd_dataset_freiburg1_desk": "fr1/desk",
    "rgbd_dataset_freiburg1_room": "fr1/room",
}
SEQ_LENGTH = {
    "heads": 1000, "chess": 2000, "fire": 2000,
    "office": 2500, "pumpkin": 2000, "redkitchen": 2500,
    "fr1/desk": 600, "fr1/room": 1350,
}
CONDITIONS = ["with_lc", "without_lc"]
CONFIG_7SCENES = "config/eval_calib.yaml"
CONFIG_TUM = "config/eval_no_calib.yaml"


def make_lc_config(base_cfg, with_lc, tmp_dir):
    suffix = "on" if with_lc else "off"
    tag = Path(base_cfg).stem
    p = tmp_dir / f"lc_{suffix}_{tag}.yaml"
    yaml.dump({"inherit": base_cfg, "retrieval": {"k": 3 if with_lc else 0}}, open(p, "w"), default_flow_style=False)
    return p

def run_slam(dataset_path, config_path, save_as, python_bin="python"):
    cmd = [python_bin, "main.py",
           "--dataset", str(dataset_path),
           "--config",  str(config_path),
           "--save-as", save_as, "--no-viz"]
    print(f"Running: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=False, text=True)
    if r.returncode != 0:
        print(f"exited with code {r.returncode}")
        return False
    return True

def compute_ate_for(traj_est, gt_file):
    sys.path.insert(0, str(Path(__file__).parent)) 
    try:
        return compute_ate(traj_est, gt_file, traj_format="tum", align=True, correct_scale=True)["rmse"]
    except Exception as e:
        print(f"ATE: {e}")
        return None


def main():
    p = argparse.ArgumentParser(description="Option D — Loop closure ablation (7-Scenes + TUM)")
    p.add_argument("--scenes_root", default="datasets/7-scenes")
    p.add_argument("--tum_root", default="datasets/tum")
    p.add_argument("--save_root", default="results/lc_ablation")
    p.add_argument("--scenes", nargs="+", default=DEFAULT_7SCENES,
                   help="7-Scenes scene names")
    p.add_argument("--tum_sequences", nargs="+", default=DEFAULT_TUM,
                   help="TUM sequence folder names")
    p.add_argument("--python", default="python")
    p.add_argument("--skip_run", action="store_true")
    args = p.parse_args()

    Path(args.save_root).mkdir(parents=True, exist_ok=True)
    results = {c: {} for c in CONDITIONS}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        for condition in CONDITIONS:
            with_lc = (condition == "with_lc")
            print(f"{'Loop closure ON' if with_lc else 'Loop closure OFF'}")

            cfg_7s  = make_lc_config(CONFIG_7SCENES, with_lc, tmp)
            cfg_tum = make_lc_config(CONFIG_TUM,     with_lc, tmp)

            # 7-Scenes
            for scene in args.scenes:
                dset = Path(args.scenes_root) / scene
                save_as = f"7-scenes/calib/{condition}/{scene}"
                key = scene

                if not dset.exists():
                    print(f"skip {dset}")
                    results[condition][key] = None
                    continue

                if not args.skip_run:
                    run_slam(dset, cfg_7s, save_as, args.python)

                traj = Path("logs") / "7-scenes" / "calib" / condition / scene / f"{scene}.txt"
                gt   = Path("groundtruths/7-scenes") / f"{scene}.txt"

                if not traj.exists():
                    print(f"Trajectory not found for {scene}")
                    results[condition][key] = None
                    continue
                if not gt.exists():
                    print(f"GT not found for {scene}")
                    results[condition][key] = None
                    continue

                rmse = compute_ate_for(traj, gt)
                n = SEQ_LENGTH.get(scene, "?")
                print(f"7-scenes {scene:15s} (~{n} fr)  ATE = " + (f"{rmse:.4f} m" if rmse else "N/A"))
                results[condition][key] = rmse

            # TUM
            for seq in args.tum_sequences:
                dset = Path(args.tum_root) / seq
                short = TUM_SHORT.get(seq, seq)
                save_as = f"tum/{condition}/{seq}"
                key = short

                if not dset.exists():
                    print(f"skip {dset}")
                    results[condition][key] = None
                    continue

                if not args.skip_run:
                    run_slam(dset, cfg_tum, save_as, args.python)

                traj = Path("logs") / "tum" / condition / seq / f"{seq}.txt"
                gt = dset / "groundtruth.txt"

                if not traj.exists():
                    print(f"Trajectory not found for {short}")
                    results[condition][key] = None
                    continue
                if not gt.exists():
                    print(f"GT not found for {short}")
                    results[condition][key] = None
                    continue

                rmse = compute_ate_for(traj, gt)
                n = SEQ_LENGTH.get(short, "?")
                print(f"TUM {short:15s} (~{n} fr)  ATE = " + (f"{rmse:.4f} m" if rmse else "N/A"))
                results[condition][key] = rmse

    out = Path(args.save_root) / "results.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nResults saved to {out}")

    # Summary
    all_keys = list({k for r in results.values() for k in r})
    scenes_keys = [k for k in all_keys if k in args.scenes]
    tum_keys = [TUM_SHORT.get(s, s) for s in args.tum_sequences if TUM_SHORT.get(s, s) in all_keys]
    ordered = scenes_keys + tum_keys

    print("Loop Closure Ablation (ATE RMSE [m])")
    print(f"{'Sequence':<18}  {'Dataset':>8}  {'~frames':>8}  "
          f"{'with_lc':>10}  {'without_lc':>12}  {'Δ (on−off)':>12}")
    for key in ordered:
        dataset = "7-Scenes" if key in args.scenes else "TUM"
        n = SEQ_LENGTH.get(key, "?")
        v = {c: results[c].get(key) for c in CONDITIONS}
        row = f"{key:<18}  {dataset:>8}  {str(n):>8}"
        for c in CONDITIONS:
            row += f"{v[c]:>12.4f}" if v[c] is not None else f"{'N/A':>12}"
        if all(v[c] is not None for c in CONDITIONS):
            row += f"{v['with_lc'] - v['without_lc']:>+12.4f}"
        else:
            row += f"{'N/A':>12}"
        print(row)
    print("Δ < 0 means loop closure improves ATE")


if __name__ == "__main__":
    main()
