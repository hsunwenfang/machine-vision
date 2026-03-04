import numpy as np

def align_trajectory(traj_est, traj_gt, align_frames=None):
    """
    Aligns traj_est to traj_gt using Umeyama algorithm (rotation + translation, no scale).
    
    Args:
        traj_est: (N, 3) numpy array, estimated trajectory (camera frame or unaligned)
        traj_gt: (N, 3) numpy array, ground truth trajectory (world frame)
        align_frames: (int or None) If set, only use the first 'align_frames' to compute alignment.
                      The minimal number of frames required is 3.
        
    Returns:
        traj_est_aligned: (N, 3) numpy array, aligned estimated trajectory
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    
    # Check dimensions
    if traj_est.shape != traj_gt.shape:
        raise ValueError(f"Trajectories must have same shape. Got {traj_est.shape} and {traj_gt.shape}")
        
    N, D = traj_est.shape
    if D != 3:
        raise ValueError("Trajectories must be 3D (Nx3)")
    
    # Determine the subset of points used for alignment
    if align_frames is not None:
        if align_frames < 3:
            raise ValueError("Need at least 3 frames to align.")
        n_align = min(align_frames, N)
    else:
        n_align = N
        
    # Standard Umeyama on the subset
    subset_est = traj_est[:n_align]
    subset_gt = traj_gt[:n_align]
    
    # 1. Compute centroids
    mu_est = np.mean(subset_est, axis=0)
    mu_gt = np.mean(subset_gt, axis=0)
    
    # 2. Center the points
    P = subset_est - mu_est
    Q = subset_gt - mu_gt
    
    # 3. Compute covariance matrix
    # H = sum( (pi - mu_p)(qi - mu_q)^T )
    H = P.T @ Q
    
    # 4. SVD
    U, S, Vt = np.linalg.svd(H)
    
    # 5. Compute Rotation
    # R = V U^T
    R = Vt.T @ U.T
    
    # 6. Check for reflection (determinant < 0)
    if np.linalg.det(R) < 0:
        # Fix reflection by negating the last column of V (or last row of Vt)
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 7. Compute Translation (t maps rotated est centroid to gt centroid)
    # t = mu_gt - R @ mu_est
    t = mu_gt - R @ mu_est
    
    # 8. Transform the ENTIRE trajectory using the computed R and t
    traj_est_aligned = (R @ traj_est.T).T + t
    
    # 9. Force the start point to match GT start point exactly (optional but good for visualization)
    # This prevents the "floating start" issue where best-fit minimizes centroid error but offsets the start
    start_offset = traj_gt[0] - traj_est_aligned[0]
    traj_est_aligned += start_offset
    
    return traj_est_aligned, R, t
