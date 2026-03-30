## Ablation Experiments

This repository extends the original MASt3R-SLAM with four ablation experiments.
Before running any experiment, make sure you have followed the full installation
procedure described below, as all required dependencies are included in the project build.

### Pointmap Fusion Ablation
Compares four pointmap fusion strategies: `recent`, `first`, `median`, `weighted_pointmap`.
```
python experiments/run_fusion_ablation.py \
    --dataset_root datasets/tum \
    --save_root    results/fusion_ablation
```

### Ray Error vs. Point Error
Compares the paper's ray error formulation against a direct 3D point error.
```
python experiments/run_error_ablation.py \
    --dataset_root datasets/tum \
    --save_root    results/error_ablation
```

### Loop Closure Ablation
Evaluates the impact of loop closure on short vs. long sequences.
```
python experiments/run_loopclosure_ablation.py \
    --scenes_root  datasets/7-scenes \
    --tum_root     datasets/tum \
    --save_root    results/lc_ablation \
    --scenes heads redkitchen \
    --tum_sequences rgbd_dataset_freiburg1_desk rgbd_dataset_freiburg1_room
```

### Matching Method Comparison
Compares iterative projective matching against k-d tree and brute-force alternatives.
```
python experiments/run_matching_ablation.py \
    --tum_root  datasets/tum \
    --save_root results/matching_ablation
```
> **Note:** Brute-force matching is memory-intensive. On GPUs with less than 24 GB VRAM,
> set `matching.bruteforce_subsample: 8` and `matching.bruteforce_chunk: 256` in `config/base.yaml`.

### Modified source files
The following files from the original repository were modified:
- `mast3r_slam/frame.py` : bug fix (added `score` field default to `Frame` dataclass)
- `mast3r_slam/tracker.py` : added point error formulation (`opt_pose_point_sim3`)
- `mast3r_slam/global_opt.py` : added Python Gauss-Newton backend for point error (`solve_GN_points`)
- `mast3r_slam/matching.py` : added k-d tree and brute-force matching methods
- `config/base.yaml` : added `error_formulation`, `method`, and related hyperparameters

[comment]: <> (# MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors)

<p align="center">
  <h1 align="center">MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors</h1>
  <p align="center">
    <a href="https://rmurai.co.uk/"><strong>Riku Murai*</strong></a>
    ·
    <a href="https://edexheim.github.io/"><strong>Eric Dexheimer*</strong></a>
    ·
    <a href="https://www.doc.ic.ac.uk/~ajd/"><strong>Andrew J. Davison</strong></a>
  </p>
  <p align="center">(* Equal Contribution)</p>

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center"><a href="https://arxiv.org/abs/2412.12392">Paper</a> | <a href="https://youtu.be/wozt71NBFTQ">Video</a> | <a href="https://edexheim.github.io/mast3r-slam/">Project Page</a></h3>
  <div align="center"></div>

<p align="center">
    <img src="./media/teaser.gif" alt="teaser" width="100%">
</p>
<br>

# Getting Started
## Installation
```
conda create -n mast3r-slam python=3.11
conda activate mast3r-slam
```
Check the system's CUDA version with nvcc
```
nvcc --version
```
Install pytorch with **matching** CUDA version following:
```
# CUDA 11.8
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

Clone the repo and install the dependencies.
```
git clone https://github.com/rmurai0610/MASt3R-SLAM.git --recursive
cd MASt3R-SLAM/

# if you've clone the repo without --recursive run
# git submodule update --init --recursive

pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .
 

# Optionally install torchcodec for faster mp4 loading
pip install torchcodec==0.1
```

Setup the checkpoints for MASt3R and retrieval.  The license for the checkpoints and more information on the datasets used is written [here](https://github.com/naver/mast3r/blob/mast3r_sfm/CHECKPOINTS_NOTICE).
```
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

## WSL Users
We have primarily tested on Ubuntu.  If you are using WSL, please checkout to the windows branch and follow the above installation.
```
git checkout windows
```
This disables multiprocessing which causes an issue with shared memory as discussed [here](https://github.com/rmurai0610/MASt3R-SLAM/issues/21).

## Examples
```
bash ./scripts/download_tum.sh
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml
```
## Live Demo
Connect a realsense camera to the PC and run
```
python main.py --dataset realsense --config config/base.yaml
```
## Running on a video
Our system can process either MP4 videos or folders containing RGB images.
```
python main.py --dataset <path/to/video>.mp4 --config config/base.yaml
python main.py --dataset <path/to/folder> --config config/base.yaml
```
If the calibration parameters are known, you can specify them in intrinsics.yaml
```
python main.py --dataset <path/to/video>.mp4 --config config/base.yaml --calib config/intrinsics.yaml
python main.py --dataset <path/to/folder> --config config/base.yaml --calib config/intrinsics.yaml
```

## Downloading Dataset
### TUM-RGBD Dataset
```
bash ./scripts/download_tum.sh
```

### 7-Scenes Dataset
```
bash ./scripts/download_7_scenes.sh
```

### EuRoC Dataset
```
bash ./scripts/download_euroc.sh
```
### ETH3D SLAM Dataset
```
bash ./scripts/download_eth3d.sh
```

## Running Evaluations
All evaluation script will run our system in a single-threaded, headless mode.
We can run evaluations with/without calibration:
### TUM-RGBD Dataset
```
bash ./scripts/eval_tum.sh 
bash ./scripts/eval_tum.sh --no-calib
```

### 7-Scenes Dataset
```
bash ./scripts/eval_7_scenes.sh 
bash ./scripts/eval_7_scenes.sh --no-calib
```

### EuRoC Dataset
```
bash ./scripts/eval_euroc.sh 
bash ./scripts/eval_euroc.sh --no-calib
```
### ETH3D SLAM Dataset
```
bash ./scripts/eval_eth3d.sh 
```

## Reproducibility
There might be minor differences between the released version and the results in the paper after developing this multi-processing version. 
We run all our experiments on an RTX 4090, and the performance may differ when running with a different GPU.

## Acknowledgement
We sincerely thank the developers and contributors of the many open-source projects that our code is built upon.
- [MASt3R](https://github.com/naver/mast3r)
- [MASt3R-SfM](https://github.com/naver/mast3r/tree/mast3r_sfm)
- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)
- [ModernGL](https://github.com/moderngl/moderngl)

# Citation
If you found this code/work to be useful in your own research, please considering citing the following:

```bibtex
@inproceedings{murai2024_mast3rslam,
  title={{MASt3R-SLAM}: Real-Time Dense {SLAM} with {3D} Reconstruction Priors},
  author={Murai, Riku and Dexheimer, Eric and Davison, Andrew J.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025},
}
```
