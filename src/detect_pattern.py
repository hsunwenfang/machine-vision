import cv2
import numpy as np

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

def get_3d_points_from_optical_flow(img_left, img_right, kp_left, des_left, K, b):
    # Q2: Check images corresponding coord on the other one
    flow_matches = find_stereo_matches_optical_flow(img_left, img_right, kp_left)
    points_3d, valid_indices = compute_3d_points_from_flow(kp_left, flow_matches, K, b)
    if len(valid_indices) > 0:
        des_tracked = np.array([des_left[i] for i in valid_indices])
    else:
        des_tracked = np.empty((0, 32), dtype=np.uint8) # Default ORB descriptor size
    return points_3d, des_tracked

def get_3d_points_from_descriptor_matching(img_right, kp_left, des_left, orb, bf, K, b):
    # ORIGINAL: Descriptor Matching (Detect Features in Both + Match)
    kp_right, des_right = orb.detectAndCompute(img_right, None)
    matches_stereo = bf.match(des_left, des_right)
    good_stereo = []
    for m in matches_stereo:
        if abs(kp_left[m.queryIdx].pt[1] - kp_right[m.trainIdx].pt[1]) < 3.0:
            good_stereo.append(m)
    points_3d, valid_indices = compute_3d_points(kp_left, kp_right, good_stereo, K, b)
    if len(valid_indices) > 0:
        des_tracked = np.array([des_left[good_stereo[idx].queryIdx] for idx in valid_indices])
    else:
        des_tracked = np.empty((0, 32), dtype=np.uint8)
    return points_3d, des_tracked
