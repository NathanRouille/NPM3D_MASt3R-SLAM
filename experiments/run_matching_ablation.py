from __future__ import annotations
import argparse, json, subprocess, sys, tempfile, time
from pathlib import Path
import yaml
from compute_ate import compute_ate

DEFAULT_TUM    = ["rgbd_dataset_freiburg1_desk", "rgbd_dataset_freiburg1_room"]
DEFAULT_SCENES = []   # 7-Scenes optional — slow with kdtree/bruteforce

TUM_SHORT = {
    "rgbd_dataset_freiburg1_desk": "fr1/desk",
    "rgbd_dataset_freiburg1_room": "fr1/room",
}

METHODS = ["iterative_proj", "kdtree", "bruteforce"]

CONFIG_TUM = "config/eval_no_calib.yaml"
CONFIG_SCENES = "config/eval_calib.yaml"


#  Config generation
def make_method_config(base_cfg, method, tmp_dir):
    tag = Path(base_cfg).stem
    p = tmp_dir / f"match_{method}_{tag}.yaml"
    yaml.dump({
        "inherit": base_cfg,
        "matching": {
            "method": method,
            "bruteforce_chunk": 4096,
        },
    }, open(p, "w"), default_flow_style=False)
    return p



def run_slam(dataset_path, config_path, save_as, python_bin="python"):
    cmd = [python_bin, "main.py",
           "--dataset", str(dataset_path),
           "--config",  str(config_path),
           "--save-as", save_as, "--no-viz"]
    print(f"Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        print(f"exited with code {r.returncode}")
        return False, elapsed
    return True, elapsed


def compute_ate_for(traj_est, gt_file):
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        return compute_ate(traj_est, gt_file, traj_format="tum", align=True, correct_scale=True)["rmse"]
    except Exception as e:
        print(f"ATE: {e}")
        return None


def main():
    p = argparse.ArgumentParser(description="Option F — Matching method ablation")
    p.add_argument("--tum_root", default="datasets/tum")
    p.add_argument("--scenes_root", default="datasets/7-scenes")
    p.add_argument("--tum_sequences", nargs="+", default=DEFAULT_TUM)
    p.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--save_root", default="results/matching_ablation")
    p.add_argument("--python", default="python")
    p.add_argument("--skip_run", action="store_true")
    args = p.parse_args()

    Path(args.save_root).mkdir(parents=True, exist_ok=True)

    results: dict = {m: {} for m in args.methods}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        for method in args.methods:
            print(f"Matching method: {method}")

            cfg_tum = make_method_config(CONFIG_TUM, method, tmp)
            cfg_scenes = make_method_config(CONFIG_SCENES, method, tmp)

            # TUM
            for seq in args.tum_sequences:
                dset = Path(args.tum_root) / seq
                short = TUM_SHORT.get(seq, seq)
                save_as = f"matching/{method}/tum/{seq}"

                if not dset.exists():
                    print(f"skip {dset}")
                    results[method][short] = None
                    continue

                wall_time = None
                if not args.skip_run:
                    ok, wall_time = run_slam(dset, cfg_tum, save_as, args.python)
                    if not ok:
                        results[method][short] = None
                        continue

                traj = Path("logs") / save_as / f"{seq}.txt"
                gt = dset / "groundtruth.txt"

                if not traj.exists():
                    print(f"Trajectory not found for {short}")
                    results[method][short] = None
                    continue

                rmse = compute_ate_for(traj, gt)
                print(f"TUM {short:15s}  ATE = " + (f"{rmse:.4f} m" if rmse else "N/A") + (f"  wall = {wall_time:.0f}s" if wall_time else ""))
                results[method][short] = {
                    "ate": rmse,
                    "wall_time_s": wall_time,
                }

            # 7-Scenes
            for scene in args.scenes:
                dset = Path(args.scenes_root) / scene
                save_as = f"matching/{method}/7scenes/{scene}"

                if not dset.exists():
                    print(f"skip {dset}")
                    results[method][scene] = None
                    continue

                wall_time = None
                if not args.skip_run:
                    ok, wall_time = run_slam(dset, cfg_scenes, save_as, args.python)
                    if not ok:
                        results[method][scene] = None
                        continue

                traj = Path("logs") / save_as / f"{scene}.txt"
                gt = Path("groundtruths/7-scenes") / f"{scene}.txt"

                if not traj.exists():
                    print(f"Trajectory not found for {scene}")
                    results[method][scene] = None
                    continue

                rmse = compute_ate_for(traj, gt)
                print(f"  [7S]  {scene:15s}  ATE = " + (f"{rmse:.4f} m" if rmse else "N/A") + (f"  wall = {wall_time:.0f}s" if wall_time else ""))
                results[method][scene] = {
                    "ate": rmse,
                    "wall_time_s": wall_time,
                }

    out = Path(args.save_root) / "results.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nResults saved to {out}")

    # Summary
    all_seqs = list({k for r in results.values() for k in r})
    tum_seqs = [TUM_SHORT.get(s, s) for s in args.tum_sequences if TUM_SHORT.get(s, s) in all_seqs]
    scene_seqs = [s for s in args.scenes if s in all_seqs]
    ordered = tum_seqs + scene_seqs

    print("Matching Ablation — ATE RMSE [m]  |  Wall time [s]")
    header = f"{'Sequence':<18}"
    for m in args.methods:
        header += f"{m[:14]:>14}"
    print(header)
    for seq in ordered:
        row = f"{seq:<18}"
        for m in args.methods:
            v = results[m].get(seq)
            if v is None:
                row += f"{'N/A':>14}"
            elif isinstance(v, dict) and v.get("ate") is not None:
                ate = v["ate"]
                wt = v.get("wall_time_s")
                cell = f"{ate:.4f}" + (f"/{wt:.0f}s" if wt else "")
                row += f"{cell:>14}"
            else:
                row += f"{'N/A':>14}"
        print(row)
    print("Format: ATE[m] / wall_time[s]")


if __name__ == "__main__":
    main()
