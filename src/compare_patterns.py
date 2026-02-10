import cv2
import numpy as np
import os

def run_compare_patterns(data_dir, cam_name='image_01'):
    """
    Focused experiment to directly compare (5,7) vs (7,11) patterns.
    """
    print(f"\nXXX RUNNING SINGLE CAMERA EXPERIMENT ({cam_name}) XXX")
    img_dir = os.path.join(data_dir, cam_name, 'data') 
    images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])
    
    patterns_to_test = [(5, 7), (7, 11)]
    
    for pattern_size in patterns_to_test:
        print(f"\n=== TESTING PATTERN {pattern_size} ===")
        # 2. Find Corners
        shape = None
        objpoints = []
        imgpoints = []
        
        # print(f"Scanning {len(images)} images...")
        valid_frames = 0
        for f in images:
            img = cv2.imread(f, 0)
            if shape is None: shape = img.shape[::-1]
            
            # Try both orientations (rows, cols) and (cols, rows)
            pats = [pattern_size, (pattern_size[1], pattern_size[0])]
            
            found = False
            for p in pats:
                ret, corners = cv2.findChessboardCorners(img, p, None)
                if ret:
                    objp = np.zeros((p[0] * p[1], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:p[0], 0:p[1]].T.reshape(-1, 2)
                    objp = objp * 0.1 # 10cm squares
                    
                    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), term)
                    
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    valid_frames += 1
                    found = True
                    break
            
        print(f"found {valid_frames} valid frames")
        if valid_frames < 3: 
            print("Not enough frames.")
            continue
    
        # STRATEGY 1: Standard
        print("--- Standard Calibration (Robust Flags) ---")
        robust_flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None, flags=robust_flags)
        print(f"  -> RMS: {ret:.4f}")
        print(f"  -> fx: {K[0,0]:.1f}")
        print(f"  -> cx, cy: {K[0,2]:.1f}, {K[1,2]:.1f}")

if __name__ == "__main__":
    CALIB_IMG_DIR = '/Users/hsunwenfang/Documents/machine-vision/data/2011_09_26_calib_img/2011_09_26_drive_0119_extract'
    run_compare_patterns(CALIB_IMG_DIR, 'image_01')
