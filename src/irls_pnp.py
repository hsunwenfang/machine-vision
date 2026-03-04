import numpy as np
import cv2
from scipy.optimize import least_squares

def project_points(rvec, tvec, obj_points, K, dist_coeffs=None):
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)
    img_points_proj, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist_coeffs)
    return img_points_proj.reshape(-1, 2)

def pnp_residuals(params, obj_points, img_points, K, dist_coeffs=None):
    rvec = params[:3]
    tvec = params[3:]
    img_points_proj = project_points(rvec, tvec, obj_points, K, dist_coeffs)
    predictions = img_points_proj.reshape(-1, 2)
    observations = img_points.reshape(-1, 2)
    residuals = (predictions - observations).ravel()
    return residuals

def solve_pnp_irls(obj_points, img_points, K, rvec_init, tvec_init, dist_coeffs=None):
    """
    Refines PnP solution using Iteratively Reweighted Least Squares (via SciPy robust least squares).
    """
    params_init = np.hstack((rvec_init.ravel(), tvec_init.ravel()))
    
    # robust loss function 'soft_l1' or 'huber' mimics IRLS behavior
    # 'huber' is robust to outliers
    res = least_squares(pnp_residuals, params_init, 
                        args=(obj_points, img_points, K, dist_coeffs),
                        loss='huber', f_scale=1.0) 
                        
    rvec_refined = res.x[:3].reshape(3, 1)
    tvec_refined = res.x[3:].reshape(3, 1)
    return rvec_refined, tvec_refined

if __name__ == "__main__":
    # Create some synthetic data
    obj_points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], dtype=np.float32)

    rvec_true = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    tvec_true = np.array([0.5, 0.5, 2.0], dtype=np.float32)
    K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
    
    # Generate perfect 2D points
    img_points_true, _ = cv2.projectPoints(obj_points, rvec_true, tvec_true, K, None)
    img_points_obs = img_points_true.reshape(-1, 2)
    
    # Add noise
    img_points_obs += np.random.normal(0, 0.5, img_points_obs.shape)
    
    # Add outlier
    img_points_obs[0] += [10, 10]
    
    # Initialization
    rvec_init = rvec_true + np.random.normal(0, 0.1, 3)
    tvec_init = tvec_true + np.random.normal(0, 0.1, 3)
    
    print("Initial Error:")
    err_init = np.linalg.norm(pnp_residuals(np.hstack((rvec_init, tvec_init)), obj_points, img_points_obs, K))
    print(err_init)

    rvec_opt, tvec_opt = solve_pnp_irls(obj_points, img_points_obs, K, rvec_init, tvec_init)
    
    print("\nRefined Error:")
    err_opt = np.linalg.norm(pnp_residuals(np.hstack((rvec_opt.ravel(), tvec_opt.ravel())), obj_points, img_points_obs, K))
    print(err_opt)
