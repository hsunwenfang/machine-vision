import numpy as np
import cv2
from scipy.optimize import least_squares

def project_points(points_3d, rvec, tvec, K):
    """Projects 3D points to 2D image plane."""
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, distCoeffs=None)
    return points_2d.reshape(-1, 2)

def reprojection_error(params, n_points, points_2d, K):
    """
    Cost function to minimize.
    params: [rotation_vector (3), translation_vector (3), point_3d_1 (3), point_3d_2 (3), ...]
    """
    rvec = params[0:3]
    tvec = params[3:6]
    points_3d = params[6:].reshape((n_points, 3))
    
    # Project current 3D estimates to 2D
    projected = project_points(points_3d, rvec, tvec, K)
    
    # Error is difference between Observed 2D pixels and Projected 3D estimates
    return (points_2d - projected).ravel()

def run_bundle_adjustment_demo():
    """
    Demonstrates optimizing the pose (R, t) and Structure (3D points) for a single stereo pair.
    """
    print("--- Running Bundle Adjustment Demo ---")
    
    # 1. Setup Mock Data (Ideally this comes from sfm_pipeline)
    # Let's say we have 10 points
    N = 10
    K = np.array([[718, 0, 607], [0, 718, 185], [0, 0, 1]], dtype=float)
    
    # Ground Truth Random Points in 3D
    points_3d_true = np.random.rand(N, 3) * 10 + np.array([-5, -5, 10]) # In front of camera
    
    # Ground Truth Pose (Camera moved slightly right and forward)
    rvec_true = np.array([0.01, 0.01, 0.0])
    tvec_true = np.array([-0.5, 0.0, 1.0])
    
    # Observations (What the camera actually "sees")
    points_2d_obs = project_points(points_3d_true, rvec_true, tvec_true, K)
    
    # 2. Add Noise (Simulate imperfect feature matching/triangulation)
    # Initial Guess for Pose (Slightly wrong)
    rvec_guess = np.array([0.0, 0.0, 0.0])
    tvec_guess = np.array([0.0, 0.0, 0.0])
    
    # Initial Guess for 3D points (Slightly wrong)
    points_3d_guess = points_3d_true + np.random.normal(0, 0.5, (N, 3))
    
    # 3. Construct Parameter Vector
    x0 = np.hstack((rvec_guess, tvec_guess, points_3d_guess.ravel()))
    
    print(f"Initial Error: {np.sum(np.abs(reprojection_error(x0, N, points_2d_obs, K))):.4f}")
    
    # 4. Minimize Error
    res = least_squares(reprojection_error, x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf', 
                        args=(N, points_2d_obs, K))
    
    # 5. Extract Results
    rvec_opt = res.x[0:3]
    tvec_opt = res.x[3:6]
    
    print("\noptimization complete.")
    print(f"Final Error: {np.sum(np.abs(res.fun)):.4f}")
    print(f"True Translation: {tvec_true}")
    print(f"Optimized Translation: {tvec_opt}")

if __name__ == "__main__":
    run_bundle_adjustment_demo()
