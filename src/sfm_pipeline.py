import cv2
import numpy as np
import os
# import sys # Unused

# Ensure we can import from the current directory
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import get_virtual_stereo_pair # Import our augmentation tool

def load_kitti_images(data_path, drive, cam_id, frame1_idx, frame2_idx):
    """Loads two consecutive images from the KITTI dataset."""
    drive_folder = f'2011_09_26_drive_{drive}_extract'
    image_folder = f'image_{cam_id:02d}/data'
    
    img1_name = f'{frame1_idx:010d}.png'
    img2_name = f'{frame2_idx:010d}.png'

    img1_path = os.path.join(data_path, drive_folder, image_folder, img1_name)
    img2_path = os.path.join(data_path, drive_folder, image_folder, img2_name)

    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Could not load images from {img1_path} or {img2_path}")
        
    return img1, img2

def load_kitti_intrinsics(calib_path, cam_id):
    """Loads the intrinsic camera matrix from the KITTI calibration file."""
    if not os.path.exists(calib_path):
        print(f"Warning: Calibration file {calib_path} not found. Using default KITTI parameters.")
        # Approximate values for KITTI
        return np.array([[718.856, 0, 607.1928],
                         [0, 718.856, 185.2157],
                         [0, 0, 1]])

    with open(calib_path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            if key == f'K_{cam_id:02d}':
                K = np.array([float(x) for x in value.split()]).reshape(3, 3)
                return K
    raise ValueError(f"Could not find intrinsics for camera {cam_id} in {calib_path}")



def solve_pose_with_scale(img1, img2, K, known_scale=1.0):
    """
    Computes pose and applies scale correction.
    """
    # 1. Feature Extraction
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # 2. Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 5:
        # Not enough matches
        return np.eye(3), np.zeros((3,1)), [], [], None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # 3. Essential Matrix
    E, mask_E = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    if E is None:
        return np.eye(3), np.zeros((3,1)), [], [], None

    # 4. Recover Pose (returns unit translation)
    _, R, t_unit, mask_Rp = cv2.recoverPose(E, pts1, pts2, K)
    
    # 5. Apply Scale
    # Monocular vision creates a scale ambiguity. 
    # t_unit has magnitude 1. We multiply by known_scale to get metric coordinates.
    t_scaled = t_unit * known_scale
    
    return R, t_scaled, pts1, pts2, mask_E


def main():
    # --- Configuration ---
    BASE_DATA_PATH = '/Users/hsunwenfang/Documents/machine-vision/data/2011_09_26_data'
    BASE_CALIB_PATH = '/Users/hsunwenfang/Documents/machine-vision/data/2011_09_26_calib'
    DRIVE = '0001'
    CAMERA_ID = 2 
    FRAME1_INDEX = 0

    # --- Mode Selection ---
    # Set to True to use synthetic data with KNOWN scale
    USE_SYNTHETIC_TEST = True 
    
    img1_path_ref = os.path.join(BASE_DATA_PATH, f'2011_09_26_drive_{DRIVE}_extract', f'image_{CAMERA_ID:02d}/data', f'{FRAME1_INDEX:010d}.png')
    calib_file = os.path.join(BASE_CALIB_PATH, 'calib_cam_to_cam.txt')
    K = load_kitti_intrinsics(calib_file, CAMERA_ID)

    if USE_SYNTHETIC_TEST:
        print("--- Testing Scale Ambiguity with Synthetic Data ---")
        # Generate two datasets with different 'physical' motion scales
        print("Dataset A: Simulated movement of scale 5.0")
        img1_a, img2_a, true_scale_a = get_virtual_stereo_pair(img1_path_ref, baseline_scale=5.0)
        
        print("Dataset B: Simulated movement of scale 20.0")
        img1_b, img2_b, true_scale_b = get_virtual_stereo_pair(img1_path_ref, baseline_scale=20.0)

        # Solve poses (blindly)
        R_a, t_a, _, _, _ = solve_pose_with_scale(img1_a, img2_a, K, known_scale=1.0) # Assume unit
        R_b, t_b, _, _, _ = solve_pose_with_scale(img1_b, img2_b, K, known_scale=1.0) # Assume unit

        print(f"\nResult without scale correction:")
        print(f"Scenario A (Small Move) | Magnitude of calculated t: {np.linalg.norm(t_a):.4f}")
        print(f"Scenario B (Big Move)   | Magnitude of calculated t: {np.linalg.norm(t_b):.4f}")
        print("Notice both are ~1.0. The algorithm cannot 'see' the size difference.")

        # Solve poses (with Known Scale Injection)
        # In a real car, this 'true_scale' comes from the speedometer (CAN bus) * time_delta
        R_a_fixed, t_a_fixed, _, _, _ = solve_pose_with_scale(img1_a, img2_a, K, known_scale=5.0)
        R_b_fixed, t_b_fixed, _, _, _ = solve_pose_with_scale(img1_b, img2_b, K, known_scale=20.0)
        
        print(f"\nResult WITH scale correction:")
        print(f"Scenario A (Corrected) | Magnitude: {np.linalg.norm(t_a_fixed):.4f}")
        print(f"Scenario B (Corrected) | Magnitude: {np.linalg.norm(t_b_fixed):.4f}")

    else:
        # Standard Real Data Run
        img1, img2 = load_kitti_images(BASE_DATA_PATH, DRIVE, CAMERA_ID, FRAME1_INDEX, FRAME1_INDEX+1)
        R, t, _, _, _ = solve_pose_with_scale(img1, img2, K, known_scale=1.0) # Unit scale
        print("Real data recovered pose:\n", t)


if __name__ == '__main__':
    main()
