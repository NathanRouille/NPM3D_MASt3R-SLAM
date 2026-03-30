import time
import torch
import torch.nn.functional as F
import numpy as np
import mast3r_slam.image as img_utils
from mast3r_slam.config import config
import mast3r_slam_backends
from scipy.spatial import KDTree


#  dispatches based on config["matching"]["method"]
def match(X11, X21, D11, D21, idx_1_to_2_init=None):
    """Dispatch to the configured matching method"""
    method = config["matching"].get("method", "iterative_proj")
    if method == "kdtree":
        return match_kdtree(X11, X21)
    elif method == "bruteforce":
        return match_bruteforce(D11, D21)
    else:  # default: iterative_proj
        return match_iterative_proj(X11, X21, D11, D21, idx_1_to_2_init)


def pixel_to_lin(p1, w):
    return p1[..., 0] + (w * p1[..., 1])


def lin_to_pixel(idx_1_to_2, w):
    u = idx_1_to_2 % w
    v = idx_1_to_2 // w
    return torch.stack((u, v), dim=-1)


def prep_for_iter_proj(X11, X21, idx_1_to_2_init):
    b, h, w, _ = X11.shape
    device = X11.device

    # Ray image
    rays_img = F.normalize(X11, dim=-1)
    rays_img = rays_img.permute(0, 3, 1, 2)  # (b,c,h,w)
    gx_img, gy_img = img_utils.img_gradient(rays_img)
    rays_with_grad_img = torch.cat((rays_img, gx_img, gy_img), dim=1)
    rays_with_grad_img = rays_with_grad_img.permute(
        0, 2, 3, 1
    ).contiguous()  # (b,h,w,c)

    # 3D points to project
    X21_vec = X21.view(b, -1, 3)
    pts3d_norm = F.normalize(X21_vec, dim=-1)

    # Initial guesses of projections
    if idx_1_to_2_init is None:
        # Reset to identity mapping
        idx_1_to_2_init = torch.arange(h * w, device=device)[None, :].repeat(b, 1)
    p_init = lin_to_pixel(idx_1_to_2_init, w)
    p_init = p_init.float()

    return rays_with_grad_img, pts3d_norm, p_init


#  Method 1: Iterative projective matching (paper default)

def match_iterative_proj(X11, X21, D11, D21, idx_1_to_2_init=None):
    cfg = config["matching"]
    b, h, w = X21.shape[:3]
    device = X11.device

    rays_with_grad_img, pts3d_norm, p_init = prep_for_iter_proj(
        X11, X21, idx_1_to_2_init
    )
    p1, valid_proj2 = mast3r_slam_backends.iter_proj(
        rays_with_grad_img,
        pts3d_norm,
        p_init,
        cfg["max_iter"],
        cfg["lambda_init"],
        cfg["convergence_thresh"],
    )
    p1 = p1.long()

    # Check for occlusion based on distances
    batch_inds = torch.arange(b, device=device)[:, None].repeat(1, h * w)
    dists2 = torch.linalg.norm(
        X11[batch_inds, p1[..., 1], p1[..., 0], :].reshape(b, h, w, 3) - X21,
        dim=-1,
    )
    valid_dists2 = (dists2 < cfg["dist_thresh"]).view(b, -1)
    valid_proj2 = valid_proj2 & valid_dists2

    if cfg["radius"] > 0:
        (p1,) = mast3r_slam_backends.refine_matches(
            D11.half(),
            D21.view(b, h * w, -1).half(),
            p1,
            cfg["radius"],
            cfg["dilation_max"],
        )

    # Convert to linear index
    idx_1_to_2 = pixel_to_lin(p1, w)

    return idx_1_to_2, valid_proj2.unsqueeze(-1)


#  Method 2: k-d tree on 3D points

def match_kdtree(X11, X21):
    """For each point in X21, find the nearest neighbour in X11 using a k-d tree built on the 3D coordinates"""

    cfg      = config["matching"]
    b, h, w  = X21.shape[:3]
    device   = X11.device
    dist_thr = cfg["dist_thresh"]

    idx_list   = []
    valid_list = []

    for bi in range(b):
        X1_np = X11[bi].reshape(-1, 3).cpu().numpy()
        X2_np = X21[bi].reshape(-1, 3).cpu().numpy()

        tree = KDTree(X1_np)
        dists, inds = tree.query(X2_np, k=1, workers=-1)

        valid = torch.from_numpy(dists < dist_thr).to(device)
        inds  = torch.from_numpy(inds.astype(np.int64)).to(device)

        idx_list.append(inds)
        valid_list.append(valid)

    idx_1_to_2 = torch.stack(idx_list,   dim=0)
    valid      = torch.stack(valid_list, dim=0).unsqueeze(-1)
    return idx_1_to_2, valid


#  Method 3: Brute-force mutual nearest neighbour on descriptors

def match_bruteforce(D11, D21):
    b, h, w, fdim = D11.shape
    device = D11.device
    chunk      = config["matching"].get("bruteforce_chunk", 256)
    subsample  = config["matching"].get("bruteforce_subsample", 8)

    idx_list   = []
    valid_list = []

    for bi in range(b):
        d1_full = F.normalize(D11[bi].reshape(-1, fdim), dim=-1).cpu()
        d2_full = F.normalize(D21[bi].reshape(-1, fdim), dim=-1).cpu()
        N = d1_full.shape[0]

        # Subsample to make computing feasible
        sub_idx = torch.arange(0, N, subsample)
        d1 = d1_full[sub_idx]
        d2 = d2_full[sub_idx]
        Ns = d1.shape[0]

        best_idx_2to1 = torch.zeros(Ns, dtype=torch.long)
        best_idx_1to2 = torch.zeros(Ns, dtype=torch.long)

        for start in range(0, Ns, chunk):
            end  = min(start + chunk, Ns)
            sims = d2[start:end] @ d1.T
            best_idx_2to1[start:end] = sims.max(dim=1).indices

        for start in range(0, Ns, chunk):
            end  = min(start + chunk, Ns)
            sims = d1[start:end] @ d2.T
            best_idx_1to2[start:end] = sims.max(dim=1).indices

        mutual = best_idx_1to2[best_idx_2to1] == torch.arange(Ns)

        # Remap to original indices
        full_idx = sub_idx[best_idx_2to1]

        # Build index in full reslution
        idx_full = torch.arange(N)
        idx_full[sub_idx] = full_idx
        valid_full = torch.zeros(N, dtype=torch.bool)
        valid_full[sub_idx] = mutual

        idx_list.append(idx_full.to(device))
        valid_list.append(valid_full.to(device))

    idx_1_to_2 = torch.stack(idx_list,   dim=0)
    valid      = torch.stack(valid_list, dim=0).unsqueeze(-1)
    return idx_1_to_2, valid
