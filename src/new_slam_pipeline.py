
import cv2
import numpy as np
import torch
import torchvision
import kornia
from kornia.feature import LoFTR
import matplotlib.pyplot as plt
import os
import sys

# Add src to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import KittiLoader, load_gt
from trajectory_utils import align_trajectory
from irls_pnp import solve_pnp_irls
from bundle_adjustment import project_points, compose_pose, stereo_pose_refine
from scipy.optimize import least_squares as scipy_least_squares


# ---------------------------------------------------------------------------
# Two-tier SLAM back-end: data structures
# ---------------------------------------------------------------------------

class Keyframe:
    """Stores keyframe data for sliding-window BA and loop closure."""
    __slots__ = ('id', 'frame_idx', 'pose', 'des_l',
                 'world_pts', 'obs_l', 'obs_r', 'valid_des_indices')

    def __init__(self, kf_id, frame_idx, pose, des_l,
                 world_pts, obs_l, obs_r, valid_des_indices):
        self.id = kf_id
        self.frame_idx = frame_idx
        self.pose = pose.copy()                             # T_w_c (4x4)
        self.des_l = des_l                                  # (N, 32) full ORB descriptors
        self.world_pts = np.asarray(world_pts, dtype=np.float64)  # (M, 3) world-frame
        self.obs_l = np.asarray(obs_l, dtype=np.float64)          # (M, 2) left 2-D obs
        self.obs_r = np.asarray(obs_r, dtype=np.float64)          # (M, 2) right 2-D obs
        self.valid_des_indices = np.asarray(valid_des_indices, dtype=int)


# ---------------------------------------------------------------------------
# Pose-graph optimisation helpers
# ---------------------------------------------------------------------------

def _pose_graph_cost(params, n_poses, edges, weights=None):
    """
    Pose-graph cost.  params = [pose_0(6), …, pose_{n-1}(6)].
    edges : list of (i, j, T_ij_measured 4×4)  where T_ij = inv(T_i) @ T_j.
    weights : optional per-edge weight (default 1.0).
    Returns : (n_edges * 6,) residual – rotation (3) + translation (3) per edge.
    """
    poses = params.reshape(n_poses, 6)
    residuals = np.empty(len(edges) * 6)
    for e_idx, (i, j, T_ij_meas) in enumerate(edges):
        # Build T_w_ci and T_w_cj from params
        R_i, _ = cv2.Rodrigues(poses[i, :3].reshape(3, 1))
        T_i = np.eye(4); T_i[:3, :3] = R_i; T_i[:3, 3] = poses[i, 3:6]
        R_j, _ = cv2.Rodrigues(poses[j, :3].reshape(3, 1))
        T_j = np.eye(4); T_j[:3, :3] = R_j; T_j[:3, 3] = poses[j, 3:6]

        # Estimated relative transform
        T_ij_est = np.linalg.inv(T_i) @ T_j
        # Error transform
        T_err = np.linalg.inv(T_ij_meas) @ T_ij_est
        rv_err, _ = cv2.Rodrigues(T_err[:3, :3])
        t_err = T_err[:3, 3]
        w = weights[e_idx] if weights is not None else 1.0
        base = e_idx * 6
        residuals[base:base+3] = rv_err.flatten() * w
        residuals[base+3:base+6] = t_err * w
    return residuals

class StereoSLAM:
    def __init__(self, K, baseline, strategy='orb', backend='lm',
                 use_local_ba=False, use_loop_closure=False,
                 window_size=10, kf_interval=3):
        self.K = K
        self.b = baseline
        self.strategy = strategy
        self.backend = backend

        # Build stereo extrinsic: T_stereo transforms left-cam coords to right-cam coords
        # For a horizontal baseline: t_x = -baseline (right cam is shifted left in world)
        self.T_stereo = np.eye(4)
        self.T_stereo[0, 3] = -baseline

        self.device = torch.device('cpu')
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')

        # Initialize Feature Extractors
        if self.strategy == 'orb':
            self.orb = cv2.ORB_create(nfeatures=3000)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif self.strategy == 'deep':
            self.matcher = LoFTR(pretrained='outdoor').to(self.device)
            self.matcher.eval()

        # Frame-to-frame tracking state
        self.prev_img_l = None
        self.prev_points_3d = None
        self.prev_kp_l = None
        self.prev_des_l = None
        self.prev_stereo_r = None              # right-image 2-D coords (parallel to prev_points_3d)

        self.trajectory = [np.eye(4)]
        self.frame_count = 0

        # ---- Two-tier back-end state ----
        self.use_local_ba = use_local_ba
        self.use_loop_closure = use_loop_closure
        self.window_size = window_size         # W – max keyframes in local window
        self.kf_interval = kf_interval         # min frames between keyframes
        self.keyframes = []                    # list[Keyframe]
        self.next_kf_id = 0
        self.last_kf_frame = -999
        self.loop_constraints = []             # (kf_i_id, kf_j_id, T_ij 4×4)
        self.odometry_edges = []               # (kf_i_id, kf_j_id, T_ij 4×4)
        self.MAX_BA_OBS = 5000                 # memory cap on BA observations

    def process_frame(self, img_l, img_r):
        """
        Process a stereo pair. 
        img_l, img_r: Grayscale/Color images (numpy arrays)
        Returns: Current Global Pose T_wc
        """
        # Preprocessing
        if len(img_l.shape) == 3: img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        if len(img_r.shape) == 3: img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        
        if self.prev_img_l is None:
            # Initialization
            if self.strategy == 'orb':
                self.prev_kp_l, self.prev_des_l, self.prev_points_3d, self.prev_stereo_r = \
                    self._compute_stereo_features_orb(img_l, img_r)
            elif self.strategy == 'deep':
                self.prev_points_3d, self.prev_kp_l = self._compute_stereo_features_deep(img_l, img_r)

            self.prev_img_l = img_l

            # First frame is always a keyframe
            if self.use_local_ba and self.strategy == 'orb':
                self._insert_keyframe(self.trajectory[-1], self.frame_count)
                self.last_kf_frame = self.frame_count
            self.frame_count += 1
            return self.trajectory[-1]

        # Tracking
        T_rel = np.eye(4)
        
        if self.strategy == 'orb':
            # 1. Match prev_l -> curr_l
            kp_curr, des_curr = self.orb.detectAndCompute(img_l, None)
            matches = self.bf.match(self.prev_des_l, des_curr)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # 2. Get 3D-2D correspondences
            obj_points = []
            img_points = []
            
            for m in matches:
                idx_prev = m.queryIdx
                idx_curr = m.trainIdx
                
                p3d = self.prev_points_3d[idx_prev]
                if p3d is None: continue
                
                obj_points.append(p3d)
                img_points.append(kp_curr[idx_curr].pt)

            obj_points = np.array(obj_points)
            img_points = np.array(img_points)
            
            # 3. Solve PnP
            if len(obj_points) > 10:
                T_rel = self._solve_pnp(obj_points, img_points)
            
            # 4. Update for next frame
            curr_kp_l, curr_des_l, curr_points_3d, curr_stereo_r = \
                self._compute_stereo_features_orb(img_l, img_r)

            self.prev_kp_l = curr_kp_l
            self.prev_des_l = curr_des_l
            self.prev_points_3d = curr_points_3d
            self.prev_stereo_r = curr_stereo_r
            self.prev_img_l = img_l

        elif self.strategy == 'deep':
             mkpts0, mkpts1 = self._match_loftr(self.prev_img_l, img_l)
             
             obj_points = []
             img_points = []
             
             if len(mkpts0) > 0 and len(self.prev_kp_l) > 0:
                 from scipy.spatial import cKDTree
                 tree = cKDTree(self.prev_kp_l)
                 dists, indices = tree.query(mkpts0, distance_upper_bound=2.0)
                 
                 for i, (dist, idx) in enumerate(zip(dists, indices)):
                     if dist == float('inf'): continue
                     if idx < len(self.prev_points_3d):
                         p3d = self.prev_points_3d[idx]
                         if p3d is not None:
                             obj_points.append(p3d)
                             img_points.append(mkpts1[i])

             obj_points = np.array(obj_points)
             img_points = np.array(img_points)
             
             if len(obj_points) > 10:
                 T_rel = self._solve_pnp(obj_points, img_points)
                 
             # Update
             self.prev_points_3d, self.prev_kp_l = self._compute_stereo_features_deep(img_l, img_r)
             self.prev_img_l = img_l

        # Update Global Pose
        # solvePnP gives T_curr_prev (model=prev cam frame -> camera=curr frame).
        # Absolute: T_w_curr = T_w_prev * inv(T_curr_prev)
        
        # Invert T_curr_prev to get T_prev_curr
        R = T_rel[:3, :3]
        t = T_rel[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t

        T_update = np.eye(4)
        T_update[:3, :3] = R_inv
        T_update[:3, 3] = t_inv

        new_pose = self.trajectory[-1] @ T_update
        self.trajectory.append(new_pose)

        # ---- Two-tier back-end: keyframe selection + local BA + loop closure ----
        if self.use_local_ba and self.strategy == 'orb':
            if self._is_keyframe(new_pose):
                self._insert_keyframe(new_pose, self.frame_count)
                self.last_kf_frame = self.frame_count

                # Sliding-window local BA (Tier 1)
                if len(self.keyframes) >= 2:
                    self._run_local_ba()

                # Loop closure detection + pose-graph (Tier 2)
                if self.use_loop_closure and len(self.keyframes) >= 6:
                    self._detect_and_close_loops()

        self.frame_count += 1
        return self.trajectory[-1]

    def _solve_pnp(self, obj_points, img_points):
        # Ensure contiguous arrays
        obj_points = np.ascontiguousarray(obj_points).astype(np.float32)
        img_points = np.ascontiguousarray(img_points).astype(np.float32)
        
        if self.backend == 'irls':
            retval, rvec, tvec = cv2.solvePnP(obj_points, img_points, self.K, None, flags=cv2.SOLVEPNP_EPNP)
            assert retval, "Initial PnP failed for IRLS refinement"
            rvec, tvec = solve_pnp_irls(obj_points, img_points, self.K, rvec, tvec)
        else:
            # LM / Standard CV2
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(obj_points, img_points, self.K, None)
            
        T = np.eye(4)
        if hasattr(rvec, 'shape'): # Check if valid
            R, _ = cv2.Rodrigues(rvec)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()
        return T

    def _compute_stereo_features_orb(self, img_l, img_r):
        """Returns (kp_l, des_l, points_3d, stereo_right_obs).

        stereo_right_obs is parallel to points_3d: ``stereo_right_obs[i]``
        is the right-image (u, v) for kp_l[i] when a valid stereo match
        exists, else ``None``.
        """
        kp_l, des_l = self.orb.detectAndCompute(img_l, None)
        kp_r, des_r = self.orb.detectAndCompute(img_r, None)

        matches = self.bf.match(des_l, des_r)

        points_3d = [None] * len(kp_l)
        stereo_right_obs = [None] * len(kp_l)

        f = self.K[0, 0]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        b = self.b

        for m in matches:
            idx_l = m.queryIdx
            idx_r = m.trainIdx

            ul = kp_l[idx_l].pt[0]
            vl = kp_l[idx_l].pt[1]
            ur = kp_r[idx_r].pt[0]
            vr = kp_r[idx_r].pt[1]

            # Epipolar check (rectified stereo → same scanline)
            if abs(vl - vr) > 1.5:
                continue

            disparity = ul - ur
            if disparity < 1.0:
                continue

            z = (f * b) / disparity
            x = (ul - cx) * z / f
            y = (vl - cy) * z / f

            if z > 80.0:
                continue

            points_3d[idx_l] = np.array([x, y, z])
            stereo_right_obs[idx_l] = np.array([ur, vr])

        return kp_l, des_l, points_3d, stereo_right_obs

    # ------------------------------------------------------------------
    # Two-tier back-end implementation
    # ------------------------------------------------------------------

    def _is_keyframe(self, pose):
        """Decide whether the current frame should be a keyframe.

        Criteria (both must hold):
          - At least ``kf_interval`` frames since the last keyframe.
          - Translation since last keyframe exceeds 0.3 m  (parallax proxy).
        """
        if self.frame_count - self.last_kf_frame < self.kf_interval:
            return False
        if len(self.keyframes) == 0:
            return True
        dt = np.linalg.norm(pose[:3, 3] - self.keyframes[-1].pose[:3, 3])
        return dt > 0.3

    def _insert_keyframe(self, pose, frame_idx):
        """Create a ``Keyframe`` from the current stereo features and store it."""
        R_wc = pose[:3, :3]
        t_wc = pose[:3, 3]

        world_pts, obs_l, obs_r, valid_idx = [], [], [], []
        for i, p3d in enumerate(self.prev_points_3d):
            if p3d is None:
                continue
            r_obs = self.prev_stereo_r[i] if self.prev_stereo_r is not None else None
            if r_obs is None:
                continue
            # Camera-frame → world-frame
            pw = R_wc @ p3d + t_wc
            world_pts.append(pw)
            obs_l.append(np.array(self.prev_kp_l[i].pt))
            obs_r.append(r_obs)
            valid_idx.append(i)

        if len(world_pts) < 10:
            return  # not enough observations

        des = self.prev_des_l   # (N, 32) full ORB descriptor array
        kf = Keyframe(self.next_kf_id, frame_idx, pose, des,
                      world_pts, obs_l, obs_r, valid_idx)
        self.keyframes.append(kf)

        # Record odometry edge to previous keyframe (before any BA changes it)
        if len(self.keyframes) >= 2:
            kf_prev = self.keyframes[-2]
            T_ij = np.linalg.inv(kf_prev.pose) @ kf.pose
            self.odometry_edges.append((kf_prev.id, kf.id, T_ij.copy()))

        self.next_kf_id += 1

    # ---- Tier 1: per-keyframe stereo pose refinement + window ----------

    def _run_local_ba(self):
        """Refine recent keyframe poses using joint stereo reprojection.

        Uses ``stereo_pose_refine`` (from bundle_adjustment.py) which
        minimises reprojection error in *both* left and right cameras
        simultaneously — adding a stereo constraint that PnP alone lacks.

        After refinement, world-frame 3-D points are corrected and the
        pose delta is propagated to all intermediate (non-keyframe) frames
        between corrected keyframes so that the entire trajectory benefits.

        Note: cross-keyframe constraints (matching prev_kf's 3-D points
        against curr_kf's observations) were tested but do NOT help with
        pose-only BA.  The previous keyframe's 3-D points carry its
        accumulated drift, and pose-only BA imports that bias.  For cross-
        keyframe constraints to work, one needs joint structure+pose BA
        (which is what global SfM BA does).
        """
        W = min(self.window_size, len(self.keyframes))
        window = self.keyframes[-W:]

        for kf in window:
            if len(kf.world_pts) < 10:
                continue

            T_wc_old = kf.pose.copy()
            T_cw = np.linalg.inv(T_wc_old)
            rvec_init, _ = cv2.Rodrigues(T_cw[:3, :3])
            tvec_init = T_cw[:3, 3]

            rvec_opt, tvec_opt, info = stereo_pose_refine(
                kf.world_pts.astype(np.float64),
                kf.obs_l.astype(np.float64),
                kf.obs_r.astype(np.float64),
                self.K, rvec_init, tvec_init, self.T_stereo,
                mode="joint")

            R_cw, _ = cv2.Rodrigues(rvec_opt.reshape(3, 1))
            T_cw_opt = np.eye(4)
            T_cw_opt[:3, :3] = R_cw
            T_cw_opt[:3, 3] = tvec_opt
            T_wc_new = np.linalg.inv(T_cw_opt)

            delta = T_wc_new @ np.linalg.inv(T_wc_old)
            kf.world_pts = (delta[:3, :3] @ kf.world_pts.T
                            + delta[:3, 3:4]).T.copy()

            kf.pose = T_wc_new.copy()
            fi = kf.frame_idx
            if fi < len(self.trajectory):
                self.trajectory[fi] = T_wc_new.copy()

            # Propagate correction to intermediate frames
            next_kf_frame = len(self.trajectory)
            kf_idx_in_list = self.keyframes.index(kf)
            if kf_idx_in_list + 1 < len(self.keyframes):
                next_kf_frame = self.keyframes[kf_idx_in_list + 1].frame_idx
            for f in range(fi + 1, min(next_kf_frame, len(self.trajectory))):
                self.trajectory[f] = delta @ self.trajectory[f]

    # ---- Tier 2: loop closure + pose-graph optimisation -----------------

    def _detect_and_close_loops(self):
        """Bag-of-descriptors loop closure with geometric verification.

        For each new keyframe, compare its ORB descriptors against all
        keyframes outside a temporal window.  If enough matches survive
        RANSAC PnP **and** the loop-derived translation is consistent
        with odometry, add a loop edge and run pose-graph optimisation.

        Anti-aliasing safeguards (cf. Note.md §9.5):
          - Raised PnP inlier threshold (25)
          - Odometry consistency: reject if loop translation < 50% of
            odometry distance (perceptual alias puts camera near the
            reference KF even though it is far away in reality).
        """
        MIN_KF_GAP = 10       # ignore recent keyframes (avoid false positives)
        MIN_MATCHES = 30      # descriptor-match threshold
        MIN_INLIERS = 25      # PnP RANSAC inlier threshold (raised from 15)
        ODOM_CONSIST  = 0.5   # loop_t must be ≥ 50 % of odometry dist

        curr_kf = self.keyframes[-1]
        curr_des = curr_kf.des_l[curr_kf.valid_des_indices]
        if curr_des is None or len(curr_des) < 20:
            return

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        loop_found = False

        for prev_kf in self.keyframes[:-MIN_KF_GAP]:
            prev_des = prev_kf.des_l[prev_kf.valid_des_indices]
            if prev_des is None or len(prev_des) < 20:
                continue

            matches = bf.match(curr_des, prev_des)
            good = [m for m in matches if m.distance < 50]
            if len(good) < MIN_MATCHES:
                continue

            # Geometric verification: PnP using prev_kf's world 3-D points
            obj_pts, img_pts = [], []
            for m in good:
                fi_prev = m.trainIdx
                fi_curr = m.queryIdx
                if fi_prev < len(prev_kf.world_pts) and fi_curr < len(curr_kf.obs_l):
                    obj_pts.append(prev_kf.world_pts[fi_prev])
                    img_pts.append(curr_kf.obs_l[fi_curr])

            if len(obj_pts) < 10:
                continue
            obj_pts = np.ascontiguousarray(obj_pts, dtype=np.float32)
            img_pts = np.ascontiguousarray(img_pts, dtype=np.float32)
            ret, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts, img_pts, self.K, None,
                iterationsCount=200, reprojectionError=3.0)
            if not ret or inliers is None or len(inliers) < MIN_INLIERS:
                continue

            # Build relative transform T_curr_prev (world-to-cam via PnP)
            R_cw, _ = cv2.Rodrigues(rvec)
            T_cw = np.eye(4); T_cw[:3, :3] = R_cw; T_cw[:3, 3] = tvec.flatten()
            T_wc_curr_from_loop = np.linalg.inv(T_cw)
            # Relative: T_ij = inv(T_wi) @ T_wj
            T_ij = np.linalg.inv(prev_kf.pose) @ T_wc_curr_from_loop

            # ---- Odometry consistency check ----
            # Reject if the loop-derived translation is much smaller
            # than odometry distance (perceptual alias symptom).
            odom_dist = np.linalg.norm(
                curr_kf.pose[:3, 3] - prev_kf.pose[:3, 3])
            loop_t = np.linalg.norm(T_ij[:3, 3])
            if odom_dist > 5.0 and loop_t < ODOM_CONSIST * odom_dist:
                print(f"  [Loop REJECTED] KF {prev_kf.id} ↔ KF {curr_kf.id}  "
                      f"({len(inliers)} inliers, loop_t={loop_t:.1f}m "
                      f"vs odom={odom_dist:.1f}m — perceptual alias)")
                continue

            self.loop_constraints.append((prev_kf.id, curr_kf.id, T_ij))
            print(f"  [Loop] KF {prev_kf.id} ↔ KF {curr_kf.id}  "
                  f"({len(inliers)} inliers, loop_t={loop_t:.1f}m, "
                  f"odom={odom_dist:.1f}m)")
            loop_found = True
            break   # one loop per keyframe is enough

        if loop_found:
            self._run_pose_graph_optimization()

    def _run_pose_graph_optimization(self):
        """Optimise all keyframe poses given stored odometry + loop edges.

        Uses per-edge weights (odometry=1.0, loop=0.3) and Huber loss
        so that any remaining false loop edges are down-weighted rather
        than allowed to dominate the solution.
        """
        n = len(self.keyframes)
        if n < 3:
            return

        ODOM_WEIGHT = 1.0
        LOOP_WEIGHT = 0.3     # loop edges are less trusted than odometry

        # Build edges + per-edge weights
        edges = []
        weights = []
        kf_id_to_idx = {kf.id: idx for idx, kf in enumerate(self.keyframes)}

        # Odometry edges (recorded at keyframe creation time — immutable)
        for (id_i, id_j, T_ij) in self.odometry_edges:
            if id_i in kf_id_to_idx and id_j in kf_id_to_idx:
                edges.append((kf_id_to_idx[id_i], kf_id_to_idx[id_j], T_ij))
                weights.append(ODOM_WEIGHT)

        # Loop-closure edges
        for (id_i, id_j, T_ij) in self.loop_constraints:
            if id_i in kf_id_to_idx and id_j in kf_id_to_idx:
                edges.append((kf_id_to_idx[id_i], kf_id_to_idx[id_j], T_ij))
                weights.append(LOOP_WEIGHT)

        if len(edges) == 0:
            return

        # Initial params: T_w_c expressed as (rvec, tvec)
        x0 = np.empty(n * 6)
        for i, kf in enumerate(self.keyframes):
            rv, _ = cv2.Rodrigues(kf.pose[:3, :3])
            x0[i*6:i*6+3] = rv.flatten()
            x0[i*6+3:i*6+6] = kf.pose[:3, 3]

        # Save original poses for world_pts correction later
        old_poses = [kf.pose.copy() for kf in self.keyframes]

        # Fix first keyframe (anchor) by not including it in free params
        def augmented_cost(params):
            full = np.concatenate([x0[:6], params])  # KF-0 is fixed
            return _pose_graph_cost(full, n, edges, weights)

        res = scipy_least_squares(
            augmented_cost, x0[6:].copy(),
            method='trf', x_scale='jac', loss='huber', f_scale=0.1,
            ftol=1e-8, max_nfev=200, verbose=0)

        opt_params = np.concatenate([x0[:6], res.x])

        # Write back and correct world_pts
        for i in range(n):
            rv = opt_params[i*6:i*6+3]
            tv = opt_params[i*6+3:i*6+6]
            R, _ = cv2.Rodrigues(rv.reshape(3, 1))
            T_wc = np.eye(4); T_wc[:3, :3] = R; T_wc[:3, 3] = tv
            # Correct world_pts:  p_new = T_wc_new @ inv(T_wc_old) @ p_old
            delta = T_wc @ np.linalg.inv(old_poses[i])
            if len(self.keyframes[i].world_pts) > 0:
                self.keyframes[i].world_pts = (
                    delta[:3, :3] @ self.keyframes[i].world_pts.T
                    + delta[:3, 3:4]).T.copy()
            self.keyframes[i].pose = T_wc.copy()
            fi = self.keyframes[i].frame_idx
            if fi < len(self.trajectory):
                self.trajectory[fi] = T_wc.copy()

        print(f"  [PGO] Optimised {n} keyframes with {len(edges)} edges "
              f"(cost {res.cost:.4f})")

    def _match_loftr(self, img0, img1):
        # Prep tensors
        t_img0, scale0 = self._np_to_torch(img0)
        t_img1, scale1 = self._np_to_torch(img1)
        
        batch = {'image0': t_img0, 'image1': t_img1}
        with torch.no_grad():
            matches = self.matcher(batch)
            
        kpts0 = matches['keypoints0'].cpu().numpy()
        kpts1 = matches['keypoints1'].cpu().numpy()
        
        # Rescale points back
        kpts0 = kpts0 / scale0
        kpts1 = kpts1 / scale1
        
        return kpts0, kpts1

    def _compute_stereo_features_deep(self, img_l, img_r):
        """Returns (points_3d, kp_l_coords)."""
        mkpts0, mkpts1 = self._match_loftr(img_l, img_r)
        
        points_3d = []
        keep_kpts_l = []
        
        f = self.K[0,0]
        cx = self.K[0,2]
        cy = self.K[1,2]
        b = self.b
        
        for i in range(len(mkpts0)):
            ul, vl = mkpts0[i]
            ur, vr = mkpts1[i]
            
            # Simple rectification check (Epipolar constraint: vertical disparity should be low)
            # If large vertical disparity, the match is likely wrong or the system is not rectified.
            if abs(vl - vr) > 1.5: continue
            
            disparity = ul - ur
            if disparity < 1.0: continue
            
            # Depth from Disparity
            z = (f * b) / disparity
            
            # Filter unrealistic depth
            if z > 80.0: continue
            
            x = (ul - cx) * z / f
            y = (vl - cy) * z / f
            
            points_3d.append(np.array([x, y, z]))
            keep_kpts_l.append(mkpts0[i])
            
        return points_3d, np.array(keep_kpts_l)

    def _np_to_torch(self, img):
        # Resize to max dim 640 for speed on CPU/MPS
        h, w = img.shape
        max_dim = 640.0
        scale = 1.0
        
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        
        # LoFTR expects grayscale, [0,1]
        img_f = img.astype(np.float32) / 255.0
        h, w = img_f.shape
        
        # Pad to be divisible by 8
        new_h = int(np.ceil(h / 8) * 8)
        new_w = int(np.ceil(w / 8) * 8)
        
        # Use padding instead of resizing to avoid scale issues with small mismatches
        pad_h = new_h - h
        pad_w = new_w - w
        
        if pad_h > 0 or pad_w > 0:
             img_f = np.pad(img_f, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
             
        t = torch.from_numpy(img_f)[None, None].to(self.device)
        return t, scale

# --- Runner ---
def run_comparison(drive_override: str | None = None):
    from sfm_pipeline import IncrementalSfM
    import argparse

    # Setup
    base_dir = '/Users/hsunwenfang/Documents/machine-vision/data/kitti'
    date = '2011_09_26'

    if drive_override is not None:
        drive = drive_override
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--drive', default='0001', help='KITTI drive number, e.g. 0001')
        args, _ = parser.parse_known_args()
        drive = args.drive

    print(f"=== Dataset: {date}, Drive: {drive} ===")

    loader = KittiLoader(base_dir, date=date, drive=drive)
    gt_path = load_gt(base_dir, date, drive)

    # ---- Instantiate pipelines ----

    # No-window stereo SLAM (frame-to-frame PnP, no BA back-end)
    print("Running: No-Window Stereo SLAM (ORB + PnP)")
    slam_no_win = StereoSLAM(loader.K, loader.b, strategy='orb', backend='lm')

    # Sliding-window stereo SLAM (per-keyframe stereo BA + loop closure)
    print("Running: Sliding-Window Stereo SLAM (ORB + Local BA + LC)")
    slam_win = StereoSLAM(loader.K, loader.b, strategy='orb', backend='lm',
                           use_local_ba=True, use_loop_closure=True,
                           window_size=10, kf_interval=3)

    # Stereo SfM (global BA at the end, metric scale from baseline)
    print("Running: Stereo SfM (ORB + Global BA)")
    sfm = IncrementalSfM(loader.K, baseline=loader.b)

    traj_nw = []
    traj_sw = []

    N = len(loader)
    for i in range(N):
        print(f"Processing frame {i}/{N}")
        frame = loader.get_frame(i)
        assert frame is not None, f"Failed to load frame {i}"
        img_l = frame['image_left']
        img_r = frame['image_right']

        pose_nw = slam_no_win.process_frame(img_l, img_r)
        pose_sw = slam_win.process_frame(img_l.copy(), img_r.copy())
        sfm.add_frame(img_l.copy(), img_r.copy())

        traj_nw.append(pose_nw[:3, 3].copy())
        traj_sw.append(pose_sw[:3, 3].copy())

    # Finalise SfM with global BA
    sfm.run_final_ba()
    traj_sfm_raw = sfm.get_trajectory()  # Nx3

    # Re-extract sliding-window trajectory (BA may have updated earlier poses)
    traj_sw_final = []
    for i in range(min(N, len(slam_win.trajectory))):
        traj_sw_final.append(slam_win.trajectory[i][:3, 3].copy())
    traj_sw = np.array(traj_sw_final) if traj_sw_final else np.array(traj_sw)

    traj_nw = np.array(traj_nw)
    if not isinstance(traj_sw, np.ndarray):
        traj_sw = np.array(traj_sw)

    if len(traj_nw) == 0:
        print("No trajectory generated.")
        return

    # ---- Alignment ----
    gt_path = gt_path - gt_path[0]
    min_len = min(len(gt_path), len(traj_nw))
    gt_trunc = gt_path[:min_len]
    ALIGN_FRAMES = 50

    # No-window SLAM (stereo → rigid alignment, no scale)
    traj_nw_trunc = traj_nw[:min_len]
    traj_nw_aligned, _, _ = align_trajectory(traj_nw_trunc, gt_trunc, align_frames=ALIGN_FRAMES)

    # Sliding-window SLAM (stereo → rigid alignment, no scale)
    traj_sw_trunc = traj_sw[:min_len]
    traj_sw_aligned, _, _ = align_trajectory(traj_sw_trunc, gt_trunc, align_frames=ALIGN_FRAMES)

    # Stereo SfM (metric scale from baseline → rigid alignment, no scale)
    sfm_min_len = min(len(traj_sfm_raw), min_len)
    traj_sfm_trunc = traj_sfm_raw[:sfm_min_len]
    gt_trunc_sfm = gt_trunc[:sfm_min_len]
    traj_sfm_aligned, _, _ = align_trajectory(
        traj_sfm_trunc, gt_trunc_sfm, align_frames=ALIGN_FRAMES)

    # ---- Print metrics ----
    d_gt = np.linalg.norm(gt_trunc[-1] - gt_trunc[0])
    d_nw = np.linalg.norm(traj_nw_trunc[-1] - traj_nw_trunc[0])
    d_sw = np.linalg.norm(traj_sw_trunc[-1] - traj_sw_trunc[0])
    d_sfm = np.linalg.norm(traj_sfm_trunc[-1] - traj_sfm_trunc[0])

    print(f"\nAlignment using first {ALIGN_FRAMES} frames.")
    print(f"End-to-End Distance (GT):               {d_gt:.2f}m")
    print(f"End-to-End Distance (No-Window SLAM):    {d_nw:.2f}m")
    print(f"End-to-End Distance (Sliding-Window):    {d_sw:.2f}m")
    print(f"End-to-End Distance (Stereo SfM):        {d_sfm:.2f}m")

    # Print sliding-window stats
    n_kf = len(slam_win.keyframes)
    n_loops = len(slam_win.loop_constraints)
    print(f"Sliding-window: {n_kf} keyframes, {n_loops} loop closures")

    # ---- Plot ----
    plt.figure(figsize=(12, 7))

    plt.plot(gt_trunc[:, 0], gt_trunc[:, 2], 'k--',
             label='Ground Truth', linewidth=2, alpha=0.8)
    plt.plot(traj_nw_aligned[:, 0], traj_nw_aligned[:, 2],
             label='No-Window Stereo SLAM', marker='.', alpha=0.6)
    plt.plot(traj_sw_aligned[:, 0], traj_sw_aligned[:, 2],
             label='Sliding-Window Stereo SLAM', marker='.', alpha=0.6, linewidth=2)
    plt.plot(traj_sfm_aligned[:sfm_min_len, 0], traj_sfm_aligned[:sfm_min_len, 2],
             label='Stereo SfM (global BA)', marker='.', alpha=0.6)

    plt.title("No-Window SLAM  vs  Sliding-Window SLAM  vs  Stereo SfM")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.legend()
    plt.grid()
    plt.axis('equal')

    out_path = f'doc/slam_comparison_{date}_drive_{drive}.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved comparison plot to {out_path}")

if __name__ == "__main__":
    run_comparison()
