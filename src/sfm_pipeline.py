"""
Structure from Motion (SfM) Pipeline — Stereo variant.

SfM reconstructs 3D structure and camera poses from a sequential collection
of images.  This stereo variant uses left + right images to get metric-scale
3D points from stereo disparity, then runs global Bundle Adjustment over ALL
frames — the key differentiator from SLAM.

  Stereo SfM (this file)                   SLAM (new_slam_pipeline.py)
  ─────────────────────────────────────    ────────────────────────────────────
  Offline / batch processing               Online / real-time incremental
  Stereo depth from known baseline         Stereo depth from known baseline
  Metric scale from stereo baseline        Metric scale from stereo baseline
  Global Bundle Adjustment over ALL        Local pose-only refinement per
    frames and points after all frames       frame (+ optional sliding-window)
  Stereo disparity → metric 3D →           PnP from known 3D-2D → refine
    grow map → global BA                     with stereo BA
  First stereo pair gives 3D directly      First stereo pair gives 3D directly

Pipeline:
  1. Feature extraction (ORB) on left image; stereo match to right for depth
  2. Initialisation: stereo disparity → metric-scale 3D map from frame 0
  3. For each new frame:
       a. Match features to existing 3D map points
       b. PnP to estimate pose
       c. Triangulate new points from stereo disparity
       d. Adaptive intermediate BA when PnP reprojection error signals drift
  4. Final global BA over all poses (two rounds: Huber-robust + L2 refine)
  5. Align to GT via rigid transform (no scale — already metric)
"""

import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import KittiLoader
from trajectory_utils import align_trajectory


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def project(pts_3d, rvec, tvec, K):
    """Project N×3 world points to 2D via (rvec, tvec, K)."""
    pts2d, _ = cv2.projectPoints(
        np.ascontiguousarray(pts_3d, dtype=np.float64),
        rvec.reshape(3, 1), tvec.reshape(3, 1), K, distCoeffs=None)
    return pts2d.reshape(-1, 2)


def triangulate_two_views(K, R1, t1, R2, t2, pts1, pts2):
    """Triangulate matched 2D points from two calibrated views.
    
    Returns Nx3 world points (only those in front of both cameras).
    """
    P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K @ np.hstack([R2, t2.reshape(3, 1)])

    pts4d = cv2.triangulatePoints(P1, P2,
                                  pts1.T.astype(np.float64),
                                  pts2.T.astype(np.float64))
    pts3d = (pts4d[:3] / pts4d[3]).T  # Nx3

    # Chirality: keep points in front of both cameras
    cam1 = (R1 @ pts3d.T + t1.reshape(3, 1))
    cam2 = (R2 @ pts3d.T + t2.reshape(3, 1))
    mask = (cam1[2] > 0) & (cam2[2] > 0)

    return pts3d, mask


# ---------------------------------------------------------------------------
# Bundle Adjustment (global, all poses + all points)
# ---------------------------------------------------------------------------

def pose_only_ba_residuals(cam_params_flat, n_cams, pts_3d, cam_indices, pt_indices, obs, K):
    """Residuals for pose-only BA (3D points fixed)."""
    cam_params = cam_params_flat.reshape(n_cams, 6)
    residuals = np.empty(len(obs) * 2)

    for ci in range(n_cams):
        mask = cam_indices == ci
        if not np.any(mask):
            continue
        idxs = np.where(mask)[0]
        pi = pt_indices[idxs]
        pts = pts_3d[pi]

        rv = cam_params[ci, :3]
        tv = cam_params[ci, 3:6]
        R, _ = cv2.Rodrigues(rv.reshape(3, 1))

        p_cam = (R @ pts.T).T + tv
        p_proj = (K @ p_cam.T).T
        uv = p_proj[:, :2] / p_proj[:, 2:3]

        residuals[idxs * 2]     = obs[idxs, 0] - uv[:, 0]
        residuals[idxs * 2 + 1] = obs[idxs, 1] - uv[:, 1]

    return residuals


def _build_jac_sparsity(n_cams, n_obs, cam_indices):
    """Build sparse Jacobian structure: each observation depends on its camera only.
    
    Residuals layout: [obs0_u, obs0_v, obs1_u, obs1_v, ...] → 2*n_obs rows
    Parameters: [cam0_rv(3), cam0_tv(3), cam1_rv(3), ...] → 6*n_cams cols
    """
    m = 2 * n_obs   # residual rows
    n = 6 * n_cams  # parameter cols
    J = lil_matrix((m, n), dtype=int)
    for i in range(n_obs):
        ci = cam_indices[i]
        row_u = 2 * i
        row_v = 2 * i + 1
        col_start = 6 * ci
        J[row_u, col_start:col_start + 6] = 1
        J[row_v, col_start:col_start + 6] = 1
    return J


def run_pose_only_ba(poses, points_3d, observations, K, max_obs=20000,
                     loss='linear', max_nfev=100):
    """Pose-only BA: optimise camera poses with fixed 3D points.
    
    Uses sparse Jacobian structure to keep memory bounded.
    Subsamples observations if total exceeds max_obs.
    """
    n_cams = len(poses)

    # Subsample observations if too many (memory guard)
    n_orig = len(observations)
    if n_orig > max_obs:
        rng = np.random.RandomState(0)
        indices = rng.choice(n_orig, max_obs, replace=False)
        observations = [observations[i] for i in sorted(indices)]
        print(f"    BA subsampled to {max_obs} observations (from {n_orig})")

    cam_params = np.zeros((n_cams, 6))
    for i, (R, t) in enumerate(poses):
        rv, _ = cv2.Rodrigues(R)
        cam_params[i, :3] = rv.flatten()
        cam_params[i, 3:6] = t.flatten()

    x0 = cam_params.ravel()

    cam_indices = np.array([o[0] for o in observations])
    pt_indices = np.array([o[1] for o in observations])
    obs_2d = np.array([[o[2], o[3]] for o in observations])

    # Sparse Jacobian structure (critical for memory)
    J_sparsity = _build_jac_sparsity(n_cams, len(observations), cam_indices)

    # Fix first camera
    lower = np.full_like(x0, -np.inf)
    upper = np.full_like(x0, np.inf)
    lower[:6] = x0[:6] - 1e-12
    upper[:6] = x0[:6] + 1e-12

    res = least_squares(pose_only_ba_residuals, x0,
                        args=(n_cams, points_3d, cam_indices, pt_indices, obs_2d, K),
                        jac_sparsity=J_sparsity,
                        method='trf', x_scale='jac',
                        loss=loss,
                        ftol=1e-6, max_nfev=max_nfev,
                        bounds=(lower, upper),
                        verbose=1)

    opt_cams = res.x.reshape(n_cams, 6)
    opt_poses = []
    for i in range(n_cams):
        R, _ = cv2.Rodrigues(opt_cams[i, :3].reshape(3, 1))
        t = opt_cams[i, 3:6].reshape(3, 1)
        opt_poses.append((R, t))

    return opt_poses, res.cost


# ---------------------------------------------------------------------------
# Incremental SfM
# ---------------------------------------------------------------------------

class IncrementalSfM:
    """Stereo incremental SfM using ORB features + stereo disparity for depth."""

    MAX_MAP_POINTS = 100000 # generous cap (pose-only BA doesn't scale with #pts)

    # Adaptive BA parameters — no dataset-specific tuning
    BA_REPROJ_THRESH = 2.0   # trigger BA when mean PnP reproj error exceeds (px)
    BA_MIN_INTERVAL  = 10    # never run BA more often than this (frames)
    BA_MAX_INTERVAL  = 50    # always run BA at least this often (safety net)

    def __init__(self, K, baseline=None):
        self.K = K
        self.baseline = baseline           # stereo baseline in metres (None → mono fallback)
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Map state
        self.poses = []           # list of (R, t) — world-to-camera
        self.points_3d = None     # Mx3 global map points
        self.observations = []    # list of (cam_idx, pt_idx, u, v)
        self.pt_last_seen = {}    # pt_idx → last cam_idx that observed it

        # Per-frame features
        self.frame_kp = []
        self.frame_des = []
        self.frame_pt_idx = []    # frame_pt_idx[f][i] = global point index or -1

        # Adaptive BA state
        self._last_ba_frame = 0        # frame index when BA last ran
        self._pnp_reproj_err = 0.0     # latest PnP mean reproj error (px)

        self.initialised = False

    # ------------------------------------------------------------------
    # Stereo disparity → metric 3D
    # ------------------------------------------------------------------
    def _stereo_triangulate(self, kp_l, des_l, img_r):
        """Match left keypoints to right image and triangulate via disparity.

        Returns:
            pts_3d : dict  kp_index → np.array([X, Y, Z]) in camera frame
        """
        kp_r, des_r = self.orb.detectAndCompute(img_r, None)
        if des_r is None or len(kp_r) == 0:
            return {}

        matches = self.bf.match(des_l, des_r)

        f = self.K[0, 0]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        b = self.baseline

        pts_3d = {}
        for m in matches:
            ul, vl = kp_l[m.queryIdx].pt
            ur, vr = kp_r[m.trainIdx].pt
            if abs(vl - vr) > 1.5:          # epipolar check (rectified)
                continue
            disp = ul - ur
            if disp < 1.0:
                continue
            Z = f * b / disp
            if Z > 200 or Z < 0.5:
                continue
            X = (ul - cx) * Z / f
            Y = (vl - cy) * Z / f
            pts_3d[m.queryIdx] = np.array([X, Y, Z])
        return pts_3d

    def add_frame(self, img_l, img_r=None):
        """Add a stereo pair (or mono image) to the reconstruction."""
        if len(img_l.shape) == 3:
            img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        if img_r is not None and len(img_r.shape) == 3:
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        kp, des = self.orb.detectAndCompute(img_l, None)
        frame_id = len(self.frame_kp)
        self.frame_kp.append(kp)
        self.frame_des.append(des)
        self.frame_pt_idx.append([-1] * len(kp))

        # Stereo-triangulate current frame's keypoints (if baseline available)
        stereo_pts = {}
        if self.baseline is not None and img_r is not None:
            stereo_pts = self._stereo_triangulate(kp, des, img_r)

        if frame_id == 0:
            # First frame — set as origin
            R0 = np.eye(3)
            t0 = np.zeros((3, 1))
            self.poses.append((R0, t0))
            if stereo_pts:
                return self._initialise_stereo(frame_id, stereo_pts)
            return np.eye(4)

        if not self.initialised and not stereo_pts:
            return self._initialise_two_view(frame_id)

        if not self.initialised and stereo_pts:
            # Still initialise from stereo on first frame; shouldn't reach here
            # normally because frame 0 already initialises.
            pass

        T = self._add_incremental(frame_id, stereo_pts)

        # Adaptive intermediate BA — triggered by PnP reprojection error
        frames_since_ba = frame_id - self._last_ba_frame
        if len(self.poses) >= 3 and (
            (frames_since_ba >= self.BA_MIN_INTERVAL
             and self._pnp_reproj_err > self.BA_REPROJ_THRESH)
            or frames_since_ba >= self.BA_MAX_INTERVAL
        ):
            self._run_intermediate_ba()
            self._last_ba_frame = frame_id

        return T

    def _initialise_stereo(self, frame_id, stereo_pts):
        """Bootstrap map from stereo disparity on the first frame (metric scale)."""
        kp = self.frame_kp[frame_id]
        map_pts_list = []
        for kp_idx, pt3d in stereo_pts.items():
            gidx = len(map_pts_list)
            self.frame_pt_idx[frame_id][kp_idx] = gidx
            u, v = kp[kp_idx].pt
            self.observations.append((frame_id, gidx, u, v))
            self.pt_last_seen[gidx] = frame_id
            map_pts_list.append(pt3d)

        self.points_3d = np.array(map_pts_list)
        self.initialised = True
        print(f"  SfM stereo init: {len(map_pts_list)} metric 3D points from frame {frame_id}")
        return np.eye(4)

    def _initialise_two_view(self, frame_id):
        """Fallback: use Essential matrix when no stereo baseline is available."""
        des0 = self.frame_des[0]
        des1 = self.frame_des[frame_id]
        kp0 = self.frame_kp[0]
        kp1 = self.frame_kp[frame_id]

        matches = self.bf.match(des0, des1)
        matches = sorted(matches, key=lambda m: m.distance)[:500]
        assert len(matches) >= 8, f"Only {len(matches)} matches — not enough for Essential matrix"

        pts0 = np.array([kp0[m.queryIdx].pt for m in matches])
        pts1 = np.array([kp1[m.trainIdx].pt for m in matches])

        E, mask_e = cv2.findEssentialMat(pts0, pts1, self.K, method=cv2.RANSAC,
                                         prob=0.999, threshold=1.0)
        mask_e = mask_e.ravel().astype(bool)

        _, R, t, mask_rp = cv2.recoverPose(E, pts0[mask_e], pts1[mask_e], self.K)

        R0, t0 = self.poses[0]
        self.poses.append((R, t))

        # Triangulate inlier matches
        pts0_in = pts0[mask_e]
        pts1_in = pts1[mask_e]
        match_in = [m for m, ok in zip(matches, mask_e) if ok]

        pts3d, chirality = triangulate_two_views(self.K, R0, t0, R, t,
                                                  pts0_in, pts1_in)
        
        # Filter by chirality and reasonable depth
        good = chirality & (pts3d[:, 2] > 0) & (pts3d[:, 2] < 200)

        map_pts = pts3d[good]
        self.points_3d = map_pts

        # Associate observations
        gi = 0
        for j, ok in enumerate(good):
            if not ok:
                continue
            m = match_in[j]
            idx0 = m.queryIdx
            idx1 = m.trainIdx
            self.frame_pt_idx[0][idx0] = gi
            self.frame_pt_idx[frame_id][idx1] = gi
            self.observations.append((0, gi, kp0[idx0].pt[0], kp0[idx0].pt[1]))
            self.observations.append((frame_id, gi, kp1[idx1].pt[0], kp1[idx1].pt[1]))
            gi += 1

        self.initialised = True
        print(f"  SfM init (mono fallback): {len(map_pts)} map points from {len(matches)} matches")
        return self._pose_to_T(R, t)

    def _add_incremental(self, frame_id, stereo_pts=None):
        """Register a new frame via PnP, triangulate new points from stereo or two-view."""
        if stereo_pts is None:
            stereo_pts = {}
        prev_id = frame_id - 1
        kp_prev = self.frame_kp[prev_id]
        des_prev = self.frame_des[prev_id]
        kp_curr = self.frame_kp[frame_id]
        des_curr = self.frame_des[frame_id]

        matches = self.bf.match(des_prev, des_curr)
        matches = sorted(matches, key=lambda m: m.distance)

        # -- Step 1: PnP from known 3D points --
        obj_pts = []
        img_pts = []
        match_with_3d = []

        for m in matches:
            gidx = self.frame_pt_idx[prev_id][m.queryIdx]
            if gidx < 0:
                continue
            obj_pts.append(self.points_3d[gidx])
            img_pts.append(kp_curr[m.trainIdx].pt)
            match_with_3d.append(m)

        if len(obj_pts) < 6:
            # Not enough 3D-2D correspondences — copy previous pose
            self.poses.append(self.poses[-1])
            return self._pose_to_T(*self.poses[-1])

        obj_pts = np.array(obj_pts, dtype=np.float64)
        img_pts = np.array(img_pts, dtype=np.float64)

        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, self.K, None,
            iterationsCount=300, reprojectionError=4.0,
            flags=cv2.SOLVEPNP_ITERATIVE)
        R_curr, _ = cv2.Rodrigues(rvec)
        t_curr = tvec.reshape(3, 1)
        self.poses.append((R_curr, t_curr))

        # Compute mean reprojection error of PnP inliers (adaptive BA signal)
        if inliers is not None and len(inliers) > 0:
            inl_idx = inliers.flatten()
            proj = project(obj_pts[inl_idx], rvec.flatten(), tvec.flatten(), self.K)
            self._pnp_reproj_err = float(np.mean(
                np.linalg.norm(proj - img_pts[inl_idx], axis=1)))
        else:
            self._pnp_reproj_err = float('inf')  # force BA on total failure

        # Record observations for PnP inliers
        if inliers is not None:
            for idx in inliers.flatten():
                m = match_with_3d[idx]
                gidx = self.frame_pt_idx[prev_id][m.queryIdx]
                self.frame_pt_idx[frame_id][m.trainIdx] = gidx
                pt2d = kp_curr[m.trainIdx].pt
                self.observations.append((frame_id, gidx, pt2d[0], pt2d[1]))
                self.pt_last_seen[gidx] = frame_id

        # -- Step 2: Triangulate new points --
        # Skip if map is already at capacity
        if self.points_3d is not None and len(self.points_3d) >= self.MAX_MAP_POINTS:
            return self._pose_to_T(R_curr, t_curr)

        room = self.MAX_MAP_POINTS - len(self.points_3d)

        # Prefer stereo disparity for new 3D points (metric scale)
        if stereo_pts:
            new_pts = []
            new_obs = []
            for m in matches:
                curr_idx = m.trainIdx
                if self.frame_pt_idx[frame_id][curr_idx] >= 0:
                    continue  # already associated
                if curr_idx not in stereo_pts:
                    continue
                if len(new_pts) >= room:
                    break
                # Camera-frame 3D → world-frame 3D
                p_cam = stereo_pts[curr_idx]  # in current camera frame
                p_world = R_curr.T @ (p_cam.reshape(3, 1) - t_curr.reshape(3, 1))
                p_world = p_world.flatten()

                gidx = len(self.points_3d) + len(new_pts)
                new_pts.append(p_world)
                self.frame_pt_idx[frame_id][curr_idx] = gidx
                self.pt_last_seen[gidx] = frame_id
                u, v = kp_curr[curr_idx].pt
                new_obs.append((frame_id, gidx, u, v))
                # Also associate in prev frame if matched
                prev_idx = m.queryIdx
                if self.frame_pt_idx[prev_id][prev_idx] < 0:
                    self.frame_pt_idx[prev_id][prev_idx] = gidx
                    u2, v2 = kp_prev[prev_idx].pt
                    new_obs.append((prev_id, gidx, u2, v2))

            if new_pts:
                self.points_3d = np.vstack([self.points_3d, np.array(new_pts)])
                self.observations.extend(new_obs)
        else:
            # Fallback: two-view temporal triangulation (mono)
            R_prev, t_prev = self.poses[prev_id]
            P_prev = self.K @ np.hstack([R_prev, t_prev.reshape(3, 1)])
            P_curr = self.K @ np.hstack([R_curr, t_curr.reshape(3, 1)])

            new_match_idxs = []
            pts_a_list = []
            pts_b_list = []

            for m in matches:
                if self.frame_pt_idx[prev_id][m.queryIdx] >= 0:
                    continue
                pts_a_list.append(kp_prev[m.queryIdx].pt)
                pts_b_list.append(kp_curr[m.trainIdx].pt)
                new_match_idxs.append(m)

            if pts_a_list:
                pts_a = np.array(pts_a_list)
                pts_b = np.array(pts_b_list)

                pts4d = cv2.triangulatePoints(P_prev, P_curr,
                                              pts_a.T.astype(np.float64),
                                              pts_b.T.astype(np.float64))
                pts3d_all = (pts4d[:3] / pts4d[3]).T

                cam_prev = (R_prev @ pts3d_all.T + t_prev.reshape(3, 1))
                cam_curr = (R_curr @ pts3d_all.T + t_curr.reshape(3, 1))
                good = (cam_prev[2] > 0) & (cam_curr[2] > 0) & (cam_prev[2] < 200)

                new_pts = []
                new_obs = []
                for j in range(len(new_match_idxs)):
                    if not good[j]:
                        continue
                    if len(new_pts) >= room:
                        break
                    m = new_match_idxs[j]
                    gidx = len(self.points_3d) + len(new_pts)
                    new_pts.append(pts3d_all[j])
                    self.frame_pt_idx[prev_id][m.queryIdx] = gidx
                    self.frame_pt_idx[frame_id][m.trainIdx] = gidx
                    new_obs.append((prev_id, gidx, pts_a[j, 0], pts_a[j, 1]))
                    new_obs.append((frame_id, gidx, pts_b[j, 0], pts_b[j, 1]))

                if new_pts:
                    self.points_3d = np.vstack([self.points_3d, np.array(new_pts)])
                    self.observations.extend(new_obs)

        return self._pose_to_T(R_curr, t_curr)

    @staticmethod
    def _pose_to_T(R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T

    def get_trajectory(self):
        """Return Nx3 camera positions in world frame."""
        positions = []
        for R, t in self.poses:
            # Camera centre in world = -R^T @ t
            C = -R.T @ t
            positions.append(C.flatten())
        return np.array(positions)

    def _cull_outlier_observations(self, max_reproj=8.0):
        """Remove observations with reprojection error > max_reproj pixels."""
        kept = []
        for cam_idx, pt_idx, u, v in self.observations:
            if pt_idx >= len(self.points_3d):
                continue
            R, t = self.poses[cam_idx]
            pt_c = R @ self.points_3d[pt_idx].reshape(3, 1) + t.reshape(3, 1)
            if pt_c[2, 0] <= 0:
                continue
            proj = self.K @ pt_c
            pu, pv = proj[0, 0] / proj[2, 0], proj[1, 0] / proj[2, 0]
            err2 = (u - pu) ** 2 + (v - pv) ** 2
            if err2 <= max_reproj ** 2:
                kept.append((cam_idx, pt_idx, u, v))
        n_before = len(self.observations)
        self.observations = kept
        print(f"    Outlier cull: {n_before} → {len(kept)} obs "
              f"(removed {n_before - len(kept)}, thresh={max_reproj}px)")

    def _run_intermediate_ba(self):
        """Run pose-only BA on all frames processed so far and cull outliers."""
        n = len(self.poses)
        print(f"  Intermediate BA at {n} cams, {len(self.observations)} obs ...")
        self._cull_outlier_observations(max_reproj=8.0)
        self.poses, cost = run_pose_only_ba(
            self.poses, self.points_3d, self.observations, self.K,
            max_obs=30000, loss='huber', max_nfev=200)
        print(f"    Intermediate BA cost: {cost:.2f}")

    def run_final_ba(self):
        """Pose-only global BA over all cameras (points fixed).

        Two rounds: first pass to fix gross drift, cull remaining outliers,
        then refine.
        """
        assert self.points_3d is not None and len(self.points_3d) > 10
        print(f"  Running final BA: {len(self.poses)} cams, "
              f"{len(self.points_3d)} pts, {len(self.observations)} obs")

        # Round 1: Huber-robust BA + outlier cull
        self._cull_outlier_observations(max_reproj=8.0)
        self.poses, cost1 = run_pose_only_ba(
            self.poses, self.points_3d, self.observations, self.K,
            max_obs=40000, loss='huber', max_nfev=300)
        print(f"    Round 1 cost: {cost1:.2f}")

        # Round 2: tighter cull + L2 refinement
        self._cull_outlier_observations(max_reproj=5.0)
        self.poses, cost2 = run_pose_only_ba(
            self.poses, self.points_3d, self.observations, self.K,
            max_obs=40000, loss='linear', max_nfev=300)
        print(f"    Round 2 cost: {cost2:.2f}")

