import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# We need to import calibration logic, but for simplicity we wrap the slam logic here
# to avoid circular dependencies or complex class imports from previous scripts.

def load_calibration(calib_path):
    """
    Load calibration for Camera 0 (Left Gray) and Camera 1 (Right Gray).
    Returns:
        K (3x3): Intrinsic matrix for Camera 0
        b (float): Baseline in meters
    """
    file_path = os.path.join(calib_path, 'calib_cam_to_cam.txt')
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Using default KITTI parameters.")
        # Default KITTI calibration (approximate)
        f = 718.856
        cx = 607.1928
        cy = 185.2157
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        b = 0.54 # Approximate baseline
        return K, b

    with open(file_path, 'r') as f:
        calib_data = {}
        for line in f:
            line = line.strip()
            if not line: continue
            if ':' in line:
                key, value = line.split(':', 1)
                # Parse numeric arrays, only if they look numeric
                # Simple heuristic: try to parse first element
                parts = value.split()
                if not parts: 
                    continue
                # Specific check for date/time strings which caused the crash ('09-Jan-2012')
                if '-' in parts[0] and parts[0][0].isdigit():
                     continue
                
                calib_data[key] = np.array([float(x) for x in parts])
    
    P0 = calib_data['P_rect_00'].reshape(3, 4)
    P1 = calib_data['P_rect_01'].reshape(3, 4)
    
    K = P0[:3, :3]
    b = (P0[0,3] - P1[0,3]) / P0[0,0]
    
    return K, b

def compute_3d_points(kp_left, kp_right, matches_lr, K, b):
    f = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]
    
    points_3d = []
    valid_indices = [] 
    
    for i, m in enumerate(matches_lr):
        idx_l = m.queryIdx
        idx_r = m.trainIdx
        
        ul = kp_left[idx_l].pt[0]
        vl = kp_left[idx_l].pt[1]
        ur = kp_right[idx_r].pt[0]
        
        disparity = ul - ur
        
        if disparity < 0.1: continue
            
        z = (f * b) / disparity
        x = (ul - cx) * z / f
        y = (vl - cy) * z / f
        
        points_3d.append([x, y, z])
        valid_indices.append(i)
        
    return np.array(points_3d), valid_indices

class StereoSLAM_Wrapper:
    def __init__(self, base_data_path, base_calib_path, drive):
        self.base_data_path = base_data_path
        self.base_calib_path = base_calib_path
        
        # Paths
        left_folder = os.path.join(base_data_path, f'2011_09_26_drive_{drive}_extract', 'image_00', 'data')
        right_folder = os.path.join(base_data_path, f'2011_09_26_drive_{drive}_extract', 'image_01', 'data')
        
        self.left_images = sorted([os.path.join(left_folder, f) for f in os.listdir(left_folder) if f.endswith('.png')])
        self.right_images = sorted([os.path.join(right_folder, f) for f in os.listdir(right_folder) if f.endswith('.png')])
        
        # Calibration
        self.K, self.b = load_calibration(base_calib_path)
        
        # Estimators
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Internal State (Current Frame Data)
        self.prev_kps = None
        self.prev_des = None
        self.prev_p3d = None
        self.prev_des_tracked = None
        
        # Init first frame
        imgL = cv2.imread(self.left_images[0], cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(self.right_images[0], cv2.IMREAD_GRAYSCALE)
        self._init_frame(imgL, imgR)
        
        self.current_R = np.eye(3)
        self.current_t = np.zeros((3,1))

    def _init_frame(self, imgL, imgR):
        kpL, desL = self.orb.detectAndCompute(imgL, None)
        kpR, desR = self.orb.detectAndCompute(imgR, None)
        
        matches = self.bf.match(desL, desR)
        good = [m for m in matches if abs(kpL[m.queryIdx].pt[1] - kpR[m.trainIdx].pt[1]) < 3.0]
        
        p3d, valid_idx = compute_3d_points(kpL, kpR, good, self.K, self.b)
        des_tracked = np.array([desL[good[i].queryIdx] for i in valid_idx])
        
        self.prev_kps = kpL # Not strictly needed for tracking but good for ref
        self.prev_des = desL 
        self.prev_p3d = p3d
        self.prev_des_tracked = des_tracked

    def process_frame(self, imgL_prev, imgR_prev, imgL_curr):
        """
        Calculates relative motion from Prev -> Curr.
        Note: We reload prev images just to keep this function stateless-ish for the loop, 
        but optimally we rely on self.prev_* state.
        """
        # Feature Det Current
        kpL_curr, desL_curr = self.orb.detectAndCompute(imgL_curr, None)
        
        # Track Temporal (Prev 3D -> Curr 2D)
        matches_temporal = self.bf.match(self.prev_des_tracked, desL_curr)
        
        obj_pts = []
        img_pts = []
        
        for m in matches_temporal:
            obj_pts.append( self.prev_p3d[m.queryIdx] )
            img_pts.append( kpL_curr[m.trainIdx].pt )
            
        obj_pts = np.array(obj_pts)
        img_pts = np.array(img_pts)

        R_rel = np.eye(3)
        t_rel = np.zeros((3,1))
        
        if len(obj_pts) > 10:
            success, rvec, tvec, _ = cv2.solvePnPRansac(obj_pts, img_pts, self.K, None)
            if success:
                # PnP gives World->Cam. We want Cam->World motion? 
                # Actually PnP gives T that transforms Frame(t-1) points to Frame(t).
                # X_curr = R * X_prev + t
                # So (R, t) is the motion of points relative to camera? No.
                # It is the motion of the CAMERA relative to the POINTS.
                # Points stay still. Camera moves. 
                # If Camera moves Forward (+Z), points appear to move Backward (-Z).
                # PnP tvec will generally be negative Z.
                
                # To get World Trajectory accumulator:
                # T_global_curr = T_global_prev * T_rel_inv
                
                R_mat, _ = cv2.Rodrigues(rvec)
                
                # Create 4x4
                T_pnp = np.eye(4)
                T_pnp[:3,:3] = R_mat
                T_pnp[:3, 3] = tvec.flatten()
                
                # This matrix moves World Points (Prev) to Camera (Curr).
                # The Camera's position in World is the inverse of this (if World was at Camera Prev).
                T_rel_inv = np.linalg.inv(T_pnp)
                
                R_rel = T_rel_inv[:3, :3]
                t_rel = T_rel_inv[:3, 3].reshape(3,1)
        
        # Prepare Next State (Stereo for Current)
        # We assume the caller provides imgR matching imgL_curr. 
        # But wait, our API above only passed imgL_curr. 
        # For this script we need to peek at the image list inside the class.
        
        # Find index of curr
        # This is a bit hacker-y for a quick script, but robust enough for a demo
        return R_rel, t_rel, kpL_curr # Return essentials

    def update_internal_state(self, i):
        # Perform stereo for frame i (which was "curr") to make it "prev" for next loop
        imgL = cv2.imread(self.left_images[i], cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(self.right_images[i], cv2.IMREAD_GRAYSCALE)
        self._init_frame(imgL, imgR)


def load_oxts_ground_truth(base_path, date, drive):
    oxts_path = os.path.join(base_path, f'{date}_drive_{drive}_extract', 'oxts/data')
    files = sorted([f for f in os.listdir(oxts_path) if f.endswith('.txt')])
    
    gt_traj = []
    
    er = 6378137. 
    scale = None
    origin = None
    
    for f in files:
        with open(os.path.join(oxts_path, f), 'r') as file:
            line = file.readline().split()
            # lat, lon, alt ...
            lat = float(line[0])
            lon = float(line[1])
            alt = float(line[2])
            
            if scale is None:
                scale = np.cos(lat * np.pi / 180.0)
                
            tx = scale * lon * np.pi * er / 180.0
            ty = scale * er * np.log( np.tan((90.0 + lat) * np.pi / 360.0) )
            
            if origin is None:
                origin = np.array([tx, ty, alt])
                
            # Local Z-up, X-East, Y-North
            x = tx - origin[0]
            y = ty - origin[1]
            z = alt - origin[2]
            
            # Map to Camera Coords roughly
            # Cam: Z forward, X right.
            # GPS X (East) -> Cam X (Right)
            # GPS Y (North) -> Cam Z (Forward)
            gt_traj.append([x, z, y]) # Swapped Y/Z for plotting convenience (Flat ground plane)
            
    return np.array(gt_traj)

def main():
    BASE_DATA_PATH = '/Users/hsunwenfang/Documents/machine-vision/data/2011_09_26_data'
    BASE_CALIB_PATH = '/Users/hsunwenfang/Documents/machine-vision/data/2011_09_26_calib'
    DATE = '2011_09_26'
    DRIVE = '0001'
    
    print("Loading Ground Truth...")
    gt_path = load_oxts_ground_truth(BASE_DATA_PATH, DATE, DRIVE)
    
    print("Running Stereo SLAM...")
    slam = StereoSLAM_Wrapper(BASE_DATA_PATH, BASE_CALIB_PATH, DRIVE)
    
    est_path = []
    errors = []
    
    # Init
    cur_pos = np.array(gt_path[0]) # Start at known correct spot
    slam.current_t = cur_pos.reshape(3,1)
    
    est_path.append(cur_pos)
    
    N = min(len(slam.left_images), len(gt_path)) - 1
    
    print(f"Processing {N} frames...")
    
    for i in range(N):
        imgL_prev = cv2.imread(slam.left_images[i], cv2.IMREAD_GRAYSCALE)
        
        # Actually our wrapper does temporal matching from internal state.
        # Check `process_frame`. We pass imgL_curr.
        imgL_curr = cv2.imread(slam.left_images[i+1], cv2.IMREAD_GRAYSCALE)
        
        R, t, _ = slam.process_frame(None, None, imgL_curr)
        
        # Accumulate
        # T_global = T_global * T_rel
        # t_global = t_global + R_global * t_rel
        
        # Just update pos for now
        slam.current_t = slam.current_t + slam.current_R.dot(t)
        slam.current_R = slam.current_R.dot(R)
        
        est_pos = slam.current_t.flatten()
        est_path.append(est_pos)
        
        # Compare to GT
        # GT is at index i+1
        gt_pos = gt_path[i+1]
        
        # Simple Distance Error (Ignoring rotation mismatch for now)
        # We only compare X/Z (Planar) distance for robustness against pitch/roll noise
        dist_err = np.linalg.norm(est_pos[[0,1]] - gt_pos[[0,1]])
        errors.append(dist_err)
        
        # Important: Update SLAM internal state (perform stereo on new frame)
        slam.update_internal_state(i+1)
        
        if i % 20 == 0:
            print(f"Frame {i}: Error {dist_err:.2f}m")
            
    # Plot
    est_path = np.array(est_path)
    gt_path_trunc = gt_path[:len(est_path)] 
    
    # --- RIGID ALIGNMENT (Umeyama/Procrustes) ---
    # We want to find R, t, s that minimizes || s * R * est + t - gt ||
    # For strict VO verification we usually keep Scale (s=1) fixed if we claim to solve scale.
    # Here we just solve for Rotation (orientation) alignment to fix the "V-shape" error.
    
    # Simple alignment: Subtract centroid, then find rotation
    mu_est = np.mean(est_path[:, [0,1]], axis=0) # Only X, Z columns
    mu_gt = np.mean(gt_path_trunc[:, [0,1]], axis=0)
    
    est_centered = est_path[:, [0,1]] - mu_est
    gt_centered = gt_path_trunc[:, [0,1]] - mu_gt
    
    # Covariance H
    H = est_centered.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T
    
    # Check reflection case
    if np.linalg.det(R_align) < 0:
        Vt[1,:] *= -1
        R_align = Vt.T @ U.T
        
    est_aligned = (est_centered @ R_align.T) + mu_gt
    
    # Recalculate Errors after alignment
    errors_aligned = np.linalg.norm(est_aligned - gt_path_trunc[:, [0,1]], axis=1)
    
    print(f"Mean Error (Raw): {np.mean(errors):.2f}m")
    print(f"Mean Error (Aligned): {np.mean(errors_aligned):.2f}m")

    plt.figure(figsize=(10,8))
    
    plt.subplot(2,1,1)
    plt.title("Trajectory Verification (Aligned)")
    plt.plot(gt_path[:,0], gt_path[:,1], 'g-', label="GPS Ground Truth")
    plt.plot(est_aligned[:,0], est_aligned[:,1], 'b--', label="Stereo SLAM (Aligned)")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.axis('equal')
    plt.grid()
    
    plt.subplot(2,1,2)
    plt.title("Position Error (Aligned)")
    plt.plot(errors_aligned, 'r')
    plt.ylabel("Error [m]")
    plt.xlabel("Frame")
    plt.grid()
    
    plt.tight_layout()
    # plt.show()
    print("Trajectory verification plot ready (plt.show() disabled for automation)")

if __name__ == "__main__":
    main()
