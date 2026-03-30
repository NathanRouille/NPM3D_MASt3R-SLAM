import lietorch
import torch
from mast3r_slam.config import config
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.geometry import (
    constrain_points_to_ray,
    skew_sym,  # needed for solve_GN_points Jacobian
)
from mast3r_slam.mast3r_utils import mast3r_match_symmetric
import mast3r_slam_backends


class FactorGraph:
    def __init__(self, model, frames: SharedKeyframes, K=None, device="cuda"):
        self.model = model
        self.frames = frames
        self.device = device
        self.cfg = config["local_opt"]
        self.ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.idx_ii2jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.idx_jj2ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.valid_match_j = torch.as_tensor([], dtype=torch.bool, device=self.device)
        self.valid_match_i = torch.as_tensor([], dtype=torch.bool, device=self.device)
        self.Q_ii2jj = torch.as_tensor([], dtype=torch.float32, device=self.device)
        self.Q_jj2ii = torch.as_tensor([], dtype=torch.float32, device=self.device)
        self.window_size = self.cfg["window_size"]

        self.K = K

    def add_factors(self, ii, jj, min_match_frac, is_reloc=False):
        kf_ii = [self.frames[idx] for idx in ii]
        kf_jj = [self.frames[idx] for idx in jj]
        feat_i = torch.cat([kf_i.feat for kf_i in kf_ii])
        feat_j = torch.cat([kf_j.feat for kf_j in kf_jj])
        pos_i = torch.cat([kf_i.pos for kf_i in kf_ii])
        pos_j = torch.cat([kf_j.pos for kf_j in kf_jj])
        shape_i = [kf_i.img_true_shape for kf_i in kf_ii]
        shape_j = [kf_j.img_true_shape for kf_j in kf_jj]

        (
            idx_i2j,
            idx_j2i,
            valid_match_j,
            valid_match_i,
            Qii,
            Qjj,
            Qji,
            Qij,
        ) = mast3r_match_symmetric(
            self.model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
        )

        batch_inds = torch.arange(idx_i2j.shape[0], device=idx_i2j.device)[
            :, None
        ].repeat(1, idx_i2j.shape[1])
        Qj = torch.sqrt(Qii[batch_inds, idx_i2j] * Qji)
        Qi = torch.sqrt(Qjj[batch_inds, idx_j2i] * Qij)

        valid_Qj = Qj > self.cfg["Q_conf"]
        valid_Qi = Qi > self.cfg["Q_conf"]
        valid_j = valid_match_j & valid_Qj
        valid_i = valid_match_i & valid_Qi
        nj = valid_j.shape[1] * valid_j.shape[2]
        ni = valid_i.shape[1] * valid_i.shape[2]
        match_frac_j = valid_j.sum(dim=(1, 2)) / nj
        match_frac_i = valid_i.sum(dim=(1, 2)) / ni

        ii_tensor = torch.as_tensor(ii, device=self.device)
        jj_tensor = torch.as_tensor(jj, device=self.device)

        # NOTE: Saying we need both edge directions to be above thrhreshold to accept either
        invalid_edges = torch.minimum(match_frac_j, match_frac_i) < min_match_frac
        consecutive_edges = ii_tensor == (jj_tensor - 1)
        invalid_edges = (~consecutive_edges) & invalid_edges

        if invalid_edges.any() and is_reloc:
            return False

        valid_edges = ~invalid_edges
        ii_tensor = ii_tensor[valid_edges]
        jj_tensor = jj_tensor[valid_edges]
        idx_i2j = idx_i2j[valid_edges]
        idx_j2i = idx_j2i[valid_edges]
        valid_match_j = valid_match_j[valid_edges]
        valid_match_i = valid_match_i[valid_edges]
        Qj = Qj[valid_edges]
        Qi = Qi[valid_edges]

        self.ii = torch.cat([self.ii, ii_tensor])
        self.jj = torch.cat([self.jj, jj_tensor])
        self.idx_ii2jj = torch.cat([self.idx_ii2jj, idx_i2j])
        self.idx_jj2ii = torch.cat([self.idx_jj2ii, idx_j2i])
        self.valid_match_j = torch.cat([self.valid_match_j, valid_match_j])
        self.valid_match_i = torch.cat([self.valid_match_i, valid_match_i])
        self.Q_ii2jj = torch.cat([self.Q_ii2jj, Qj])
        self.Q_jj2ii = torch.cat([self.Q_jj2ii, Qi])

        added_new_edges = valid_edges.sum() > 0
        return added_new_edges

    def get_unique_kf_idx(self):
        return torch.unique(torch.cat([self.ii, self.jj]), sorted=True)

    def prep_two_way_edges(self):
        ii = torch.cat((self.ii, self.jj), dim=0)
        jj = torch.cat((self.jj, self.ii), dim=0)
        idx_ii2jj = torch.cat((self.idx_ii2jj, self.idx_jj2ii), dim=0)
        valid_match = torch.cat((self.valid_match_j, self.valid_match_i), dim=0)
        Q_ii2jj = torch.cat((self.Q_ii2jj, self.Q_jj2ii), dim=0)
        return ii, jj, idx_ii2jj, valid_match, Q_ii2jj

    def get_poses_points(self, unique_kf_idx):
        kfs = [self.frames[idx] for idx in unique_kf_idx]
        Xs = torch.stack([kf.X_canon for kf in kfs])
        T_WCs = lietorch.Sim3(torch.stack([kf.T_WC.data for kf in kfs]))
        Cs = torch.stack([kf.get_average_conf() for kf in kfs])
        return Xs, T_WCs, Cs

    # Modification : dispatch function to choose solver based on config
    # Called from solve_GN_rays so existing caller-side code is unchanged
    def solve_GN_rays(self):
        """Dispatches to solve_GN_points when error_formulation == 'point', otherwise uses the original CUDA ray solver"""
        error_formulation = self.cfg.get("error_formulation", "ray")
        if error_formulation == "point":
            return self.solve_GN_points()
        return self._solve_GN_rays_cuda()

    # original solve_GN_rays function
    def _solve_GN_rays_cuda(self):
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)
        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        max_iter = self.cfg["max_iters"]
        sigma_ray = self.cfg["sigma_ray"]
        sigma_dist = self.cfg["sigma_dist"]
        delta_thresh = self.cfg["delta_norm"]

        pose_data = T_WCs.data[:, 0, :]
        mast3r_slam_backends.gauss_newton_rays(
            pose_data,
            Xs,
            Cs,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            sigma_ray,
            sigma_dist,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        # Update the keyframe T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])

    # pure PyTorch Gauss-Newton with 3D point error
    def solve_GN_points(self):
        """Python Gauss-Newton solver using 3D point error instead of ray+distance"""
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)
        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        sigma_point  = self.cfg.get("sigma_point", 0.05)
        C_thresh     = self.cfg["C_conf"]
        Q_thresh     = self.cfg["Q_conf"]
        max_iter     = self.cfg["max_iters"]
        delta_thresh = self.cfg["delta_norm"]
        subsample    = int(self.cfg.get("point_subsample", 4))

        device = self.device
        dtype  = Xs.dtype
        n      = n_unique_kf

        # Map global keyframe index to local index
        kf_to_local = {int(k): v for v, k in enumerate(unique_kf_idx.tolist())}

        for _ in range(max_iter):
            H = torch.zeros(7 * n, 7 * n, device=device, dtype=dtype)
            g = torch.zeros(7 * n, 1, device=device, dtype=dtype)

            n_edges = ii.shape[0]
            for e in range(n_edges):
                i_global = int(ii[e])
                j_global = int(jj[e])
                i_local  = kf_to_local[i_global]
                j_local  = kf_to_local[j_global]

                # Per-edge data
                idx_i2j_e = idx_ii2jj[e].view(-1)
                valid_e   = valid_match[e].view(-1, 1)
                Q_e       = Q_ii2jj[e].view(-1, 1)

                Xi = Xs[i_local]
                Xj = Xs[j_local]
                Ci = Cs[i_local]
                Cj = Cs[j_local]

                # Get Sim3 poses
                T_WCi = lietorch.Sim3(T_WCs.data[i_local : i_local + 1])
                T_WCj = lietorch.Sim3(T_WCs.data[j_local : j_local + 1])

                # Combined validity mask
                valid_C = (Ci > C_thresh) & (Cj[idx_i2j_e] > C_thresh)
                valid_all = valid_e & valid_C & (Q_e > Q_thresh)
                mask = valid_all.squeeze(1)

                if mask.sum() == 0:
                    continue

                # Optional subsampling to speed up backend
                if subsample > 1:
                    idxs = torch.where(mask)[0][::subsample]
                    mask_sub = torch.zeros_like(mask)
                    mask_sub[idxs] = True
                    mask = mask_sub

                Xi_valid  = Xi[mask]
                Xj_valid  = Xj[idx_i2j_e[mask]]
                Q_valid   = Q_e[mask]

                # Relative transform in camera i frame
                T_ij     = T_WCi.inv() * T_WCj
                T_ij     = lietorch.Sim3(T_ij.data.squeeze(0))
                Xj_in_i  = T_ij.act(Xj_valid)

                # Residual
                r = Xi_valid - Xj_in_i

                # Jacobian via world-frame positions
                # y_j: world-frame positions of matched j-points
                T_WCj_sq = lietorch.Sim3(T_WCj.data.squeeze(0))
                yj = T_WCj_sq.act(Xj_valid)

                J_base = self._compute_J_base(yj)

                # Apply linear part of T_WCi.inv() to each column
                J_world = self._apply_T_inv_lin(T_WCi, J_base)

                # Information weight
                sqrt_info = (1.0 / sigma_point) * torch.sqrt(Q_valid).repeat(1, 3)

                # Whitened residual & Jacobians
                wr  = (sqrt_info * r).reshape(-1, 1)
                wJi = (sqrt_info[..., None] * J_world).reshape(-1, 7)
                wJj = (-sqrt_info[..., None] * J_world).reshape(-1, 7)

                # Accumulate normal equations
                si = slice(7 * i_local, 7 * (i_local + 1))
                sj = slice(7 * j_local, 7 * (j_local + 1))

                H[si, si] += wJi.T @ wJi
                H[sj, sj] += wJj.T @ wJj
                H[si, sj] += wJi.T @ wJj
                H[sj, si] += wJj.T @ wJi

                g[si] += -(wJi.T @ wr)
                g[sj] += -(wJj.T @ wr)

            # Solve for free poses
            free_start = 7 * pin
            H_free = H[free_start:, free_start:]
            g_free = g[free_start:]

            # Tikhonov regularisation for stability
            H_free = H_free + 1e-6 * torch.eye(H_free.shape[0], device=device, dtype=dtype)

            try:
                L = torch.linalg.cholesky(H_free, upper=False)
                delta_free = torch.cholesky_solve(g_free, L, upper=False).view(-1)
            except RuntimeError:
                break

            # Apply incremental updates to free poses
            for k, local_idx in enumerate(range(pin, n)):
                delta_k = delta_free[7 * k : 7 * (k + 1)].unsqueeze(0)
                T_WCs.data[local_idx] = T_WCs[local_idx].retr(delta_k).data[0]

            if delta_free.norm().item() < delta_thresh:
                break

        # Write back updated poses
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])


    def solve_GN_calib(self):
        K = self.K
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)

        # Constrain points to ray
        img_size = self.frames[0].img.shape[-2:]
        Xs = constrain_points_to_ray(img_size, Xs, K)

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        pixel_border = self.cfg["pixel_border"]
        z_eps = self.cfg["depth_eps"]
        max_iter = self.cfg["max_iters"]
        sigma_pixel = self.cfg["sigma_pixel"]
        sigma_depth = self.cfg["sigma_depth"]
        delta_thresh = self.cfg["delta_norm"]

        pose_data = T_WCs.data[:, 0, :]

        height, width = img_size

        mast3r_slam_backends.gauss_newton_calib(
            pose_data,
            Xs,
            Cs,
            K,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            height,
            width,
            pixel_border,
            z_eps,
            sigma_pixel,
            sigma_depth,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        # Update the keyframe T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])


    # Helpers for solve_GN_points
    @staticmethod
    def _compute_J_base(y: torch.Tensor):
        """Build the base Jacobian for a batch of world-frame points y"""
        N = y.shape[0]
        device, dtype = y.device, y.dtype

        I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(N, -1, -1)
        neg_skew = -skew_sym(y)
        y_col = y.unsqueeze(-1)

        return torch.cat([I, neg_skew, y_col], dim=-1)

    @staticmethod
    def _apply_T_inv_lin(T_WCi: lietorch.Sim3, J_base: torch.Tensor):
        """Apply the *linear* (rotation+scale, no translation) part of T_WCi.inv() to every column of J_base"""
        N = J_base.shape[0]
        device, dtype = J_base.device, J_base.dtype

        T_WCi_inv = lietorch.Sim3(T_WCi.data.squeeze(0)).inv()

        # Translation of T_WCi^{-1}  (subtracted to isolate the linear part)
        zero = torch.zeros(1, 3, device=device, dtype=dtype)
        t_inv = T_WCi_inv.act(zero)

        cols = J_base.permute(0, 2, 1).reshape(-1, 3)

        # Apply T_WCi^{-1} and subtract translation : linear part only
        transformed = T_WCi_inv.act(cols) - t_inv

        J_world = transformed.reshape(N, 7, 3).permute(0, 2, 1)

        return J_world
