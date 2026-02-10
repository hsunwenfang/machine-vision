import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def find_stereo_matches_optical_flow(img_left, img_right, kp_left):
    """
    Alternative to detecting features in Right image.
    Uses Lucas-Kanade Optical Flow to 'look' for the Left point in the Right image.
    """
    # Convert Keypoints to numpy array (N, 1, 2)
    pts_left = np.float32([p.pt for p in kp_left]).reshape(-1, 1, 2)
    
    # Use Optical Flow to find corresponding points in Right image
    # We search in a small window. Since images are rectified, y should not change much.
    pts_right, status, err = cv2.calcOpticalFlowPyrLK(
        img_left, img_right, pts_left, None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    
    good_matches = []
    
    # Filter bad tracking points
    for i in range(len(pts_left)):
        if status[i] == 1:
            pt_l = pts_left[i].ravel()
            pt_r = pts_right[i].ravel()
            
            # Constraint 1: Epipolar Constraint (Rectified images)
            # The feature should appear on the same Y-row (roughly)
            # Relaxed tolerance because Flow isn't perfect
            if abs(pt_l[1] - pt_r[1]) > 2.0: 
                continue
                
            # Constraint 2: Disparity Constraint
            # Objects are in front of camera, so x_left must be >= x_right
            if pt_l[0] < pt_r[0]:
                continue
                
            # Store (left_index, right_coord)
            good_matches.append((i, pt_r))
            
    return good_matches # List of (index_in_left, coord_in_right)

def compute_3d_points_from_flow(kp_left, flow_matches, K, b):
    """
    Compute 3D points from Optical Flow matches.
    """
    f = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]
    
    points_3d = []
    valid_indices = [] 
    
    for idx_l, pt_r in flow_matches:
        ul = kp_left[idx_l].pt[0]
        vl = kp_left[idx_l].pt[1]
        ur = pt_r[0]
        
        disparity = ul - ur
        
        if disparity < 0.1: continue
            
        z = (f * b) / disparity
        x = (ul - cx) * z / f
        y = (vl - cy) * z / f
        
        points_3d.append([x, y, z])
        valid_indices.append(idx_l)
        
    return np.array(points_3d), valid_indices

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

    calib_data = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if ':' not in line: continue
            key, value = line.split(':', 1)
            
            # Skip metadata lines
            if key == 'calib_time':
                continue
                
            calib_data[key] = np.array([float(x) for x in value.split()])

    # P_rect_00 = [fu, 0, cu, -fu*Tx]
    # P_rect_01 = [fu, 0, cu, -fu*(Tx + B)]
    # For KITTI, P_rect_xx projects point in identifying rectified camera 0 coords to image xx.
    # The reference frame is Camera 0.
    
    P0 = calib_data['P_rect_00'].reshape(3, 4)
    P1 = calib_data['P_rect_01'].reshape(3, 4)
    
    # Intrinsic K is the first 3x3 of P0 (since P0 projects cam0 to img0)
    K = P0[:3, :3]
    
    # Baseline calc
    # P1[0,3] = P0[0,3] - f * b
    # b = (P0[0,3] - P1[0,3]) / P0[0,0]
    
    b = (P0[0,3] - P1[0,3]) / P0[0,0]
    
    return K, b

def compute_3d_points(kp_left, kp_right, matches_lr, K, b):
    """
    Triangulate 2D points from stereo match to get 3D points.
    Returns:
        points_3d: (N, 3) coordinates in camera frame
        valid_indices: Indices of left keypoints that were successfully triangulated
    """
    f = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]
    
    points_3d = []
    valid_indices = [] # Indices into the original list of keypoints/descriptors
    
    for i, m in enumerate(matches_lr):
        idx_l = m.queryIdx
        idx_r = m.trainIdx
        
        ul = kp_left[idx_l].pt[0]
        vl = kp_left[idx_l].pt[1]
        ur = kp_right[idx_r].pt[0]
        
        disparity = ul - ur
        
        if disparity < 0.1: # Skip points at infinity or invalid matches
            continue
            
        # Z = (f * b) / d
        z = (f * b) / disparity
        x = (ul - cx) * z / f
        y = (vl - cy) * z / f
        
        points_3d.append([x, y, z])
        valid_indices.append(i)
        
    return np.array(points_3d), valid_indices

def run_stereo_slam(method='descriptor_matching'):
    # --- Config ---
    # Updated paths to match workspace structure
    BASE_DIR_CALIB = '/Users/hsunwenfang/Documents/machine-vision/data/2011_09_26_calib'
    BASE_DIR_DATA = '/Users/hsunwenfang/Documents/machine-vision/data/2011_09_26_data'
    DRIVE = '0001'
    
    # Use image_00 (Left) and image_01 (Right) - Grayscale
    # Note: Structure is data/2011_09_26_data/2011_09_26_drive_0001_extract/...
    LEFT_FOLDER = os.path.join(BASE_DIR_DATA, f'2011_09_26_drive_{DRIVE}_extract', 'image_00', 'data')
    RIGHT_FOLDER = os.path.join(BASE_DIR_DATA, f'2011_09_26_drive_{DRIVE}_extract', 'image_01', 'data')
    
    # --- Load Calibration ---
    K, b = load_calibration(BASE_DIR_CALIB)
    print(f"Calibration Loaded: f={K[0,0]:.2f}, Baseline={b:.4f}m")
    
    images_l = sorted([f for f in os.listdir(LEFT_FOLDER) if f.endswith('.png')])
    images_r = sorted([f for f in os.listdir(RIGHT_FOLDER) if f.endswith('.png')])
    
    # Global Pose
    T_global = np.eye(4)
    trajectory = []
    
    # Feature Detector
    orb = cv2.ORB_create(nfeatures=3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # CrossCheck is strict, good for stereo

    print(f"Processing {len(images_l)} frames... Method: {method}")
    
    # Initialization
    img_l_prev = cv2.imread(os.path.join(LEFT_FOLDER, images_l[0]), cv2.IMREAD_GRAYSCALE)
    img_r_prev = cv2.imread(os.path.join(RIGHT_FOLDER, images_r[0]), cv2.IMREAD_GRAYSCALE)
    
    kp_l_prev, des_l_prev = orb.detectAndCompute(img_l_prev, None)
    
    points_3d_prev = []
    des_l_prev_tracked = []
    
    if method == 'optical_flow':
        # Q2: Check images corresponding coord on the other one
        flow_matches = find_stereo_matches_optical_flow(img_l_prev, img_r_prev, kp_l_prev)
        points_3d_prev, valid_indices = compute_3d_points_from_flow(kp_l_prev, flow_matches, K, b)
        des_l_prev_tracked = np.array([des_l_prev[i] for i in valid_indices])
    else:
        # ORIGINAL: Descriptor Matching (Detect Features in Both + Match)
        kp_r_prev, des_r_prev = orb.detectAndCompute(img_r_prev, None)
        matches_stereo = bf.match(des_l_prev, des_r_prev)
        good_stereo = []
        for m in matches_stereo:
            if abs(kp_l_prev[m.queryIdx].pt[1] - kp_r_prev[m.trainIdx].pt[1]) < 3.0:
                good_stereo.append(m)
        points_3d_prev, valid_indices = compute_3d_points(kp_l_prev, kp_r_prev, good_stereo, K, b)
        des_l_prev_tracked = np.array([des_l_prev[good_stereo[idx].queryIdx] for idx in valid_indices])
    
    trajectory.append(T_global[:3, 3].copy())

    # --- Main Loop ---
    for i in range(1, len(images_l) - 1): # Process fewer frames for demo speed
        # Load Current
        img_l_curr = cv2.imread(os.path.join(LEFT_FOLDER, images_l[i]), cv2.IMREAD_GRAYSCALE)
        img_r_curr = cv2.imread(os.path.join(RIGHT_FOLDER, images_r[i]), cv2.IMREAD_GRAYSCALE)
        
        kp_l_curr, des_l_curr = orb.detectAndCompute(img_l_curr, None)
        
        # 1. Temporal Matching (Left_prev -> Left_curr)
        matches_temporal = bf.match(des_l_prev_tracked, des_l_curr)
        
        # 2. PnP (2D-3D Correspondences)
        obj_points = [] # 3D points from Prev Frame
        img_points = [] # 2D points in Curr Frame
        
        for m in matches_temporal:
            obj_points.append(points_3d_prev[m.queryIdx])
            img_points.append(kp_l_curr[m.trainIdx].pt)
            
        obj_points = np.array(obj_points)
        img_points = np.array(img_points)
        
        if len(obj_points) < 10:
            print(f"Frame {i}: Lost tracking (too few matches)")
            continue
            
        # Solve PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(obj_points, img_points, K, None)
        
        if success:
            R, _ = cv2.Rodrigues(rvec)
            T_rel = np.eye(4)
            T_rel[:3, :3] = R
            T_rel[:3, 3] = tvec.flatten()
            
            T_rel_inv = np.linalg.inv(T_rel)
            
            T_global = T_global @ T_rel_inv
            
            trajectory.append(T_global[:3, 3].copy())
            if i % 20 == 0:
                print(f"Frame {i}: Pos {T_global[0,3]:.2f}, {T_global[2,3]:.2f}")
            
        # 3. Prepare for Next Frame
        if method == 'optical_flow':
             # Use Flow for Stereo
            flow_matches_curr = find_stereo_matches_optical_flow(img_l_curr, img_r_curr, kp_l_curr)
            points_3d_curr, valid_indices_curr = compute_3d_points_from_flow(kp_l_curr, flow_matches_curr, K, b)
            if len(valid_indices_curr) > 0:
                des_l_curr_tracked = np.array([des_l_curr[i] for i in valid_indices_curr])
                points_3d_prev = points_3d_curr
                des_l_prev_tracked = des_l_curr_tracked
        else:
            # ORIGINAL: Descriptor Matching
            kp_r_curr, des_r_curr = orb.detectAndCompute(img_r_curr, None)
            matches_stereo_curr = bf.match(des_l_curr, des_r_curr)
            good_stereo_curr = []
            for m in matches_stereo_curr:
                if abs(kp_l_curr[m.queryIdx].pt[1] - kp_r_curr[m.trainIdx].pt[1]) < 3.0:
                    good_stereo_curr.append(m)
                    
            points_3d_curr, valid_indices_curr = compute_3d_points(kp_l_curr, kp_r_curr, good_stereo_curr, K, b)
            if len(valid_indices_curr) > 0:
                 des_l_curr_tracked = np.array([des_l_curr[good_stereo_curr[idx].queryIdx] for idx in valid_indices_curr])
                 points_3d_prev = points_3d_curr
                 des_l_prev_tracked = des_l_curr_tracked
    
    return np.array(trajectory)

if __name__ == "__main__":
    print("\n=== Running Descriptor Matching Method ===")
    traj_desc = run_stereo_slam(method='descriptor_matching')
    
    print("\n=== Running Optical Flow Method ===")
    traj_flow = run_stereo_slam(method='optical_flow')
    
    # --- Comparison Plot ---
    plt.figure(figsize=(10, 6))
    
    # Plot Descriptor Matching
    plt.plot(traj_desc[:,0], traj_desc[:,2], label='Descriptor Matching', color='blue', linewidth=2)
    plt.scatter(traj_desc[-1,0], traj_desc[-1,2], color='blue', marker='x') # End point
    
    # Plot Optical Flow
    plt.plot(traj_flow[:,0], traj_flow[:,2], label='Optical Flow', color='red', linestyle='--', linewidth=2)
    plt.scatter(traj_flow[-1,0], traj_flow[-1,2], color='red', marker='x') # End point
    
    plt.title('Comparison: Descriptor Matching vs Optical Flow')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    print("\n--- Analysis ---")
    if len(traj_desc) > 0 and len(traj_flow) > 0:
        print(f"Descriptor Matching Final Pos: X={traj_desc[-1,0]:.2f}, Z={traj_desc[-1,2]:.2f}")
        print(f"Optical Flow Final Pos:      X={traj_flow[-1,0]:.2f}, Z={traj_flow[-1,2]:.2f}")
        print("Difference: {:.2f}m".format(np.linalg.norm(traj_desc[-1, [0,2]] - traj_flow[-1, [0,2]])))
    
    # plt.show()
    print("Stereo SLAM plot ready (plt.show() disabled for automation)")
