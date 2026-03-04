"""
Stereo Bundle Adjustment with three optimization modes:

  Mode 1 (Left-Only):  Optimize T_L only; derive T_R = T_stereo * T_L
  Mode 2 (Right-Only): Optimize T_R only; derive T_L = T_stereo^-1 * T_R
  Mode 3 (Joint):      Optimize T_L + delta_T_stereo; use both images

See spec.md E.1 for details.
"""

import numpy as np
import cv2
from scipy.optimize import least_squares


def project_points(points_3d, rvec, tvec, K):
    """Projects 3D points to 2D using a single camera."""
    pts, _ = cv2.projectPoints(
        np.ascontiguousarray(points_3d, dtype=np.float64),
        rvec.reshape(3, 1), tvec.reshape(3, 1), K, distCoeffs=None)
    return pts.reshape(-1, 2)


def compose_pose(T_stereo, rvec_L, tvec_L):
    """Compose T_R = T_stereo @ T_L  (both world-to-cam)."""
    R_L, _ = cv2.Rodrigues(rvec_L.reshape(3, 1))
    R_S = T_stereo[:3, :3]
    t_S = T_stereo[:3, 3]
    R_R = R_S @ R_L
    t_R = R_S @ tvec_L.reshape(3) + t_S
    rvec_R, _ = cv2.Rodrigues(R_R)
    return rvec_R.flatten(), t_R.flatten()


def invert_T(T):
    """Invert a 4x4 rigid transform."""
    T_inv = np.eye(4)
    R, t = T[:3, :3], T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------

def cost_left_only(params, n_pts, obs_L, K, T_stereo):
    """Mode 1: Left-Only. Minimize reprojection in left image only."""
    rv, tv = params[:3], params[3:6]
    pts = params[6:].reshape(n_pts, 3)
    proj = project_points(pts, rv, tv, K)
    return (obs_L - proj).ravel()


def cost_right_only(params, n_pts, obs_R, K, T_stereo):
    """Mode 2: Right-Only. Minimize reprojection in right image only."""
    rv, tv = params[:3], params[3:6]
    pts = params[6:].reshape(n_pts, 3)
    proj = project_points(pts, rv, tv, K)
    return (obs_R - proj).ravel()


def cost_joint(params, n_pts, obs_L, obs_R, K, T_stereo_init):
    """Mode 3: Joint. Minimize reprojection in BOTH images with refinable T_stereo."""
    rv_L, tv_L = params[:3], params[3:6]
    d_rv_S, d_tv_S = params[6:9], params[9:12]
    pts = params[12:].reshape(n_pts, 3)

    # Build refined T_stereo = delta * T_stereo_init
    R_init = T_stereo_init[:3, :3]
    t_init = T_stereo_init[:3, 3]
    dR, _ = cv2.Rodrigues(d_rv_S.reshape(3, 1))
    T_stereo = np.eye(4)
    T_stereo[:3, :3] = dR @ R_init
    T_stereo[:3, 3] = t_init + d_tv_S

    # Left projection
    proj_L = project_points(pts, rv_L, tv_L, K)
    err_L = (obs_L - proj_L).ravel()

    # Right projection (derived via rigid constraint)
    rv_R, tv_R = compose_pose(T_stereo, rv_L, tv_L)
    proj_R = project_points(pts, rv_R, tv_R, K)
    err_R = (obs_R - proj_R).ravel()

    return np.concatenate([err_L, err_R])


# ---------------------------------------------------------------------------
# Pose-only cost functions (fixed 3D points — typical SLAM refinement)
# ---------------------------------------------------------------------------

def pose_cost_left(params, pts_3d, obs_L, K, T_stereo):
    """Pose-only Mode 1: optimise T_L, project into left camera."""
    rv, tv = params[:3], params[3:6]
    proj = project_points(pts_3d, rv, tv, K)
    return (obs_L - proj).ravel()


def pose_cost_right(params, pts_3d, obs_R, K, T_stereo):
    """Pose-only Mode 2: optimise T_R, project into right camera."""
    rv, tv = params[:3], params[3:6]
    proj = project_points(pts_3d, rv, tv, K)
    return (obs_R - proj).ravel()


def pose_cost_joint(params, pts_3d, obs_L, obs_R, K, T_stereo):
    """Pose-only Mode 3: optimise T_L, project into both cameras via rigid baseline."""
    rv_L, tv_L = params[:3], params[3:6]
    proj_L = project_points(pts_3d, rv_L, tv_L, K)
    err_L = (obs_L - proj_L).ravel()
    rv_R, tv_R = compose_pose(T_stereo, rv_L, tv_L)
    proj_R = project_points(pts_3d, rv_R, tv_R, K)
    err_R = (obs_R - proj_R).ravel()
    return np.concatenate([err_L, err_R])


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def stereo_pose_refine(pts_3d, img_points_L, img_points_R, K,
                       rvec_init, tvec_init, T_stereo, mode="joint"):
    """
    Pose-only stereo refinement (3D points held fixed).
    This is the typical SLAM scenario where triangulated points are known.

    Returns: (rvec_opt, tvec_opt, info_dict)
    All poses returned are for the LEFT camera.
    """
    rv0 = rvec_init.flatten()
    tv0 = tvec_init.flatten()
    x0 = np.concatenate([rv0, tv0])

    if mode == "left_only":
        res = least_squares(pose_cost_left, x0,
            args=(pts_3d, img_points_L, K, T_stereo),
            method="trf", x_scale="jac", ftol=1e-8, max_nfev=500)
        return res.x[:3], res.x[3:6], {"cost": res.cost, "mode": mode,
               "nfev": res.nfev, "nresiduals": len(res.fun)}

    elif mode == "right_only":
        rv_R0, tv_R0 = compose_pose(T_stereo, rv0, tv0)
        x0_R = np.concatenate([rv_R0, tv_R0])
        res = least_squares(pose_cost_right, x0_R,
            args=(pts_3d, img_points_R, K, T_stereo),
            method="trf", x_scale="jac", ftol=1e-8, max_nfev=500)
        T_inv = invert_T(T_stereo)
        rv_opt, tv_opt = compose_pose(T_inv, res.x[:3], res.x[3:6])
        return rv_opt, tv_opt, {"cost": res.cost, "mode": mode,
               "nfev": res.nfev, "nresiduals": len(res.fun)}

    elif mode == "joint":
        res = least_squares(pose_cost_joint, x0,
            args=(pts_3d, img_points_L, img_points_R, K, T_stereo),
            method="trf", x_scale="jac", ftol=1e-8, max_nfev=500)
        return res.x[:3], res.x[3:6], {"cost": res.cost, "mode": mode,
               "nfev": res.nfev, "nresiduals": len(res.fun)}
    else:
        raise ValueError(f"Unknown mode: {mode}")
