import numpy as np
import cv2
import os

def parse_calib_file(filepath):
    """
    Parses KITTI calib_cam_to_cam.txt file
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line: continue
            key, value = line.split(':', 1)
            if key == 'calib_time': continue
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def step1_undistort(raw_img, calib_data, cam_id):
    """
    Step 1: Undistortion.
    Removes lens distortion (radial/tangential) using coefficients D.
    The output is a 'perfect' pinhole camera image, but still in the original orientation.
    """
    K = calib_data[f'K_{cam_id}'].reshape(3, 3)
    D = calib_data[f'D_{cam_id}'] 
    
    # cv2.undistort returns the undistorted image.
    # We pass K as the newCameraMatrix to preserve the original intrinsics.
    undistorted = cv2.undistort(raw_img, K, D, None, K)
    return undistorted

def step2_rectify(undistorted_img, calib_data, cam_id):
    """
    Step 2: Rectification.
    Rotates the image plane so that it is coplanar with the other stereo camera.
    This involves a homography warp: H = K_new * R_rect * inv(K_old).
    
    Note: 'undistorted_img' must have intrinsics K_{cam_id} (no distortion).
    """
    h, w = undistorted_img.shape[:2]
    
    K_old = calib_data[f'K_{cam_id}'].reshape(3, 3)
    R_rect = calib_data[f'R_rect_{cam_id}'].reshape(3, 3)
    P_rect = calib_data[f'P_rect_{cam_id}'].reshape(3, 4)
    K_new = P_rect[:3, :3] # The new, common intrinsics
    
    # Compute Homography: p_new = H * p_old
    # p_new = K_new * R_rect * K_old^-1 * p_old
    H = K_new @ R_rect @ np.linalg.inv(K_old)
    
    rectified = cv2.warpPerspective(undistorted_img, H, (w, h))
    return rectified

def unified_rectification(raw_img, calib_data, cam_id):
    """
    Standard Efficient Implementation.
    Combines undistortion and rectification into a single remap operation.
    This avoids resampling artifacts from doing two separate warps.
    
    Rectifies a raw image using KITTI calibration parameters.
    Args:
        raw_img: The raw (distorted) image
        calib_data: Dictionary containing parsed calibration data
        cam_id: '00', '01', '02', '03'
    """
    # 1. Load Raw Intrinsics (K) and Distortion (D)
    K_raw = calib_data[f'K_{cam_id}'].reshape(3, 3)
    D_raw = calib_data[f'D_{cam_id}'] 
    
    # 2. Load Rectification Rotation (R_rect)
    R_rect = calib_data[f'R_rect_{cam_id}'].reshape(3, 3)
    
    # 3. Load Projection Matrix (P_rect)
    P_rect = calib_data[f'P_rect_{cam_id}'].reshape(3, 4)
    K_new = P_rect[:3, :3]
    
    h, w = raw_img.shape[:2]
    
    # 4. Generate InitUndistortRectifyMap
    map1, map2 = cv2.initUndistortRectifyMap(K_raw, D_raw, R_rect, K_new, (w, h), cv2.CV_32FC1)
    
    # 5. Remap
    rectified_img = cv2.remap(raw_img, map1, map2, cv2.INTER_LINEAR)
    
    return rectified_img

def find_checkerboard_corners(images, pattern_size=(9, 12)): # KITTI is often 9x12 or similar large board
    """
    Finds corners in a list of images.
    Returns object points and image points.
    """
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    
    # Square size usually doesn't affect K/D, only T translation scale.
    # We'll assume unit specific or adjust if we knew the square size (e.g. 0.1 meters)
    # KITTI square size is approx 10cm? But for matching calibration params (which are unitless/pixels for K),
    # let's just keep it simple. Only T depends on this.
    
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    found_count = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        # Try a few common sizes if the default fails? 
        # For this specific implementation, we will stick to the passed pattern_size
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret == True:
            objpoints.append(objp)
            
            # Refine corner locations
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            found_count += 1
            # print(f"Found corners in {os.path.basename(fname)}")
        else:
            pass
            # print(f"Could not find corners in {os.path.basename(fname)}")

    return objpoints, imgpoints, (gray.shape[1], gray.shape[0]), found_count

def run_recalibration_experiment(calib_img_dir, target_calib_data):
    """
    Runs full calibration pipeline on raw checkerboard images and compares with KITTI txt.
    """
    print("\nXXX RUNNING RE-CALIBRATION EXPERIMENT XXX")
    
    # 1. Gather Images
    left_dir = os.path.join(calib_img_dir, 'image_02', 'data') # Gray Left
    right_dir = os.path.join(calib_img_dir, 'image_03', 'data') # Gray Right
    
    if not os.path.exists(left_dir):
        print(f"Calibration images not found at {left_dir}")
        return

    left_images = sorted([os.path.join(left_dir, f) for f in os.listdir(left_dir) if f.endswith('.png')])
    right_images = sorted([os.path.join(right_dir, f) for f in os.listdir(right_dir) if f.endswith('.png')])

    print(f"Found {len(left_images)} stereo pairs for calibration.")
    
    # 2. Find Corners
    # Try different pattern sizes since we don't know the exact one used
    # Note: OpenCV pattern size is (inner_cols, inner_rows)
    # KITTI usually uses a large board. The user data might only show a PART of the board
    # or the board is smaller.
    # Let's try to visualize one image to see what's going on if you were debugging manually,
    # but here we'll just try more patterns.
    pattern_size = None
    found_common_pattern = False
    
    # Pre-load a few images
    test_params_l = [cv2.imread(f, 0) for f in left_images] # Load ALL to check
    test_params_r = [cv2.imread(f, 0) for f in right_images]

    # Try to find ANY pattern that works for BOTH Left and Right
    best_pattern = None
    max_valid_pairs = 0
    
    # Expanded list based on typical boards
    candidate_patterns = [(5,7), (7,5), (6, 11), (11, 6)]

    for pat in candidate_patterns:
        valid_pairs_count = 0
        # Check pairs
        for im_l, im_r in zip(test_params_l, test_params_r):
             if im_l is None or im_r is None: continue
             ret_l, _ = cv2.findChessboardCorners(im_l, pat, None)
             if ret_l:
                 ret_r, _ = cv2.findChessboardCorners(im_r, pat, None)
                 if ret_r:
                     valid_pairs_count += 1
        print(f"Pattern {pat}: {valid_pairs_count} valid stereo pairs found.")
        if valid_pairs_count > max_valid_pairs:
            max_valid_pairs = valid_pairs_count
            best_pattern = pat
    
    if max_valid_pairs < 3:
         print("Could not find a pattern size with enough valid stereo pairs (>3).")
         return
         
    pattern_size = best_pattern
    print(f"Selected best pattern: {pattern_size} with {max_valid_pairs} pairs.")

    print(f"Detecting all corners with pattern: {pattern_size}...")
    objpoints, imgpoints_l, shape, n_l = find_checkerboard_corners(left_images, pattern_size)
    _, imgpoints_r, _, n_r = find_checkerboard_corners(right_images, pattern_size)
    
    if n_l < 3 or n_r < 3:
        print(f"Not enough corners found overall (Left: {n_l}, Right: {n_r}).")
        return

    # We need matching pairs where BOTH found corners
    # (Since find_checkerboard_corners returns lists only for successes, indices might misalign 
    # if one image failed and the other succeeded. A robust implementation tracks indices.)
    # Quick fix: Iterate loop manually paired.
    
    objpoints = []
    imgpoints_l = []
    imgpoints_r = []
    
    # Reliable paired collection
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    
    # Scale factor (approximate square size in meters, e.g. 0.1m)
    # This affects translation (T) magnitude.
    square_size = 0.1 
    objp = objp * square_size
    
    valid_pairs = 0
    for f_l, f_r in zip(left_images, right_images):
        img_l = cv2.imread(f_l, 0)
        img_r = cv2.imread(f_r, 0)
        
        ret_l, corners_l = cv2.findChessboardCorners(img_l, pattern_size, None)
        ret_r, corners_r = cv2.findChessboardCorners(img_r, pattern_size, None)
        
        if ret_l and ret_r:
            objpoints.append(objp)
            
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_l = cv2.cornerSubPix(img_l, corners_l, (11, 11), (-1, -1), term)
            corners_r = cv2.cornerSubPix(img_r, corners_r, (11, 11), (-1, -1), term)
            
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)
            valid_pairs += 1
            
    print(f"Used {valid_pairs} valid pairs for stereo calibration.")
    if valid_pairs < 3:
        print("Error: Too few valid pairs.")
        return

    # 3. Individual Calibration (Optional, helps Stereo)
    # Using FIX_ASPECT_RATIO and other flags to make it more robust for small datasets
    print("Performing individual camera calibration...")
    ret_l, K_l, D_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, shape, None, None)
    ret_r, K_r, D_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, shape, None, None)
    
    # 4. Stereo Calibration
    # Computes R, T between cameras
    # Using CALIB_USE_INTRINSIC_GUESS because we just calculated K_l, K_r
    flags = cv2.CALIB_FIX_INTRINSIC
    
    print("Performing stereo calibration...")
    ret_s, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        K_l, D_l,
        K_r, D_r,
        shape,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=flags
    )
    
    print("\n--- Calculated Calibration (From Images) ---")
    print("K_left (Intrinsics):\n", K_l)
    print("D_left (Distortion):\n", D_l)
    print("T (Translation):\n", T.flatten())
    
    # 5. Stereo Rectification
    # Computes R1, R2, P1, P2 (rectification transforms)
    # alpha=0 means 'crop image to valid area'. alpha=1 means 'resize to fit all pixels'
    R1, R2, P1, P2, Q, roi_l, roi_r = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, shape, R, T, alpha=0
    )
    
    print("\n--- Calculated Rectification ---")
    print("P_rect_left (Calculated):\n", P1)
    
    # 6. Comparison with Provided Text File
    if target_calib_data:
        print("\n--- COMPARISON with calib_cam_to_cam.txt ---")
        
        # Compare K (Intrinsics)
        # Note: KITTI calib file uses 'K_02' for left gray, 'K_03' for right gray usually?
        # Or 00/01. The user path used image_02/03 in prior steps?
        # Let's assume we are comparing against 02 (Left)
        
        K_kitti = target_calib_data.get('K_02')
        if K_kitti is not None:
            K_kitti = K_kitti.reshape(3,3)
            err_fx = abs(K_l[0,0] - K_kitti[0,0])
            err_cx = abs(K_l[0,2] - K_kitti[0,2])
            print(f"Focal Length (fx) Diff: {err_fx:.4f} (Calc: {K_l[0,0]:.2f} vs Txt: {K_kitti[0,0]:.2f})")
            print(f"Principal Point (cx) Diff: {err_cx:.4f} (Calc: {K_l[0,2]:.2f} vs Txt: {K_kitti[0,2]:.2f})")
        
        # Compare P_rect (New Projection)
        P_kitti = target_calib_data.get('P_rect_02')
        if P_kitti is not None:
            P_kitti = P_kitti.reshape(3,4)
            # P1 is calculated P_rect for left
            # Scale P1 by the ratio of fx for fair comparison if checker size was wrong?
            # No, P matrix elements are in pixels, so they should match if K matches.
            err_p_fx = abs(P1[0,0] - P_kitti[0,0])
            print(f"Rectified fx Diff: {err_p_fx:.4f} (Calc: {P1[0,0]:.2f} vs Txt: {P_kitti[0,0]:.2f})")
    
    print("\nNote on Discrepancies:")
    print("1. Checkerboard square size is unknown (assumed 0.1m).")
    print("   -> This affects Translation (T) magnitude linearly, but NOT rotation (R) or Principal Point.")
    print("2. The subset of images used here is likely smaller than the full set KITTI researchers used.")
    print("3. Our corner detection parameter (pattern size) was determined automatically.")
    print("4. IMPORTANT: If 'f' differs significantly, the checkerboard corner detection might be noisy.")

def main():
    # Paths (Adjusted to workspace structure)
    BASE_DIR_CALIB = '/Users/hsunwenfang/Documents/machine-vision/data/2011_09_26_calib'
    calib_file = os.path.join(BASE_DIR_CALIB, 'calib_cam_to_cam.txt')
    
    # Try alternate location if strict structure not found
    if not os.path.exists(calib_file):
        BASE_DIR_CALIB = '/Users/hsunwenfang/Documents/machine-vision/data/2011_09_26'
        calib_file = os.path.join(BASE_DIR_CALIB, 'calib_cam_to_cam.txt')
    
    if not os.path.exists(calib_file):
        print(f"Error: Calibration file not found at {calib_file}")
        return

    print(f"Loading calibration from {calib_file}")
    calib_data = parse_calib_file(calib_file)
    
    # Test on an image if available
    BASE_DIR_DATA = '/Users/hsunwenfang/Documents/machine-vision/data/2011_09_26_data'
    drive_path = os.path.join(BASE_DIR_DATA, '2011_09_26_drive_0001_extract', 'image_00', 'data')
    
    if os.path.exists(drive_path):
        img_names = sorted([f for f in os.listdir(drive_path) if f.endswith('.png')])
        if not img_names:
             print("No PNG images found in drive path.")
             return
             
        img_name = img_names[0]
        img_path = os.path.join(drive_path, img_name)
        
        print(f"Testing rectification on: {img_path}")
        raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if raw_img is None:
             print("Failed to load image.")
             return

        # Demonstrate 2-step process
        print("Performing Step 1: Undistortion...")
        undistorted = step1_undistort(raw_img, calib_data, '00')
        
        print("Performing Step 2: Rectification (Homography Warp)...")
        rectified_2step = step2_rectify(undistorted, calib_data, '00')
        
        print("Performing Unified Rectification (Reference)...")
        rectified_unified = unified_rectification(raw_img, calib_data, '00')
        
        diff = np.abs(rectified_2step.astype(float) - rectified_unified.astype(float)).mean()
        print(f"Mean pixel difference between 2-step and unified: {diff:.4f}")
        print("(Small differences are expected due to interpolation/resampling artifacts)")
        
    else:
        print(f"No data found at {drive_path}")

    # Add Experiment Call
    CALIB_IMG_DIR = '/Users/hsunwenfang/Documents/machine-vision/data/2011_09_26_calib_img/2011_09_26_drive_0119_extract'
    if os.path.exists(CALIB_IMG_DIR):
        run_recalibration_experiment(CALIB_IMG_DIR, calib_data)
    else:
        print(f"Directory for calibration images not found: {CALIB_IMG_DIR}")

if __name__ == "__main__":
    main()
