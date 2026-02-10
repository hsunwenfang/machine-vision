import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sfm_pipeline import load_kitti_intrinsics, solve_pose_with_scale

def run_slam_sequence():
    BASE_DATA_PATH = '/Users/hsunwenfang/Documents/machine-vision/data/2011_09_26_data'
    BASE_CALIB_PATH = '/Users/hsunwenfang/Documents/machine-vision/data/2011_09_26_calib'
    DRIVE = '0001'
    CAMERA_ID = 0  # Grayscale usually better for pure processing
    
    # Path setup
    drive_folder = f'2011_09_26_drive_{DRIVE}_extract'
    image_folder = os.path.join(BASE_DATA_PATH, drive_folder, f'image_{CAMERA_ID:02d}/data')
    calib_file = os.path.join(BASE_CALIB_PATH, 'calib_cam_to_cam.txt')
    
    K = load_kitti_intrinsics(calib_file, CAMERA_ID)
    
    # Get all images
    images = sorted([os.path.join(image_folder, x) for x in os.listdir(image_folder) if x.endswith('.png')])
    
    # State Initialization
    # Current Camera Position (Global)
    cur_R = np.eye(3)
    cur_t = np.zeros((3, 1))
    
    trajectory_x = []
    trajectory_z = []
    
    prev_img = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
    
    print(f"Starting SLAM on {len(images)} frames...")
    
    # Loop through sequence
    for i in range(1, len(images)-1): # Process first 50 frames for demo
        curr_img = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
        
        # 1. Estimate Relative Pose
        # Note: We are using scale=1.0 here (Monocular drift will happen). 
        # In a real car, we would pass speed * dt as scale.
        R_rel, t_rel, _, _, _ = solve_pose_with_scale(prev_img, curr_img, K, known_scale=1.0)
        
        # 2. Update Global Pose
        # T_global = T_global_prev * T_rel
        # t_global = t_global_prev + R_global_prev * t_rel
        cur_t = cur_t + cur_R.dot(t_rel)
        cur_R = cur_R.dot(R_rel) # Concatenate rotation
        
        # Store for plotting
        trajectory_x.append(cur_t[0][0])
        trajectory_z.append(cur_t[2][0])
        
        print(f"Frame {i}: Pos: [{cur_t[0][0]:.2f}, {cur_t[2][0]:.2f}]")
            
        prev_img = curr_img

    # 3. Visualization
    plt.figure()
    plt.plot(trajectory_x, trajectory_z, marker='o', markersize=2)
    plt.title("Estimated Trajectory (Monocular, Unscaled)")
    plt.xlabel("X (side)")
    plt.ylabel("Z (forward)")
    plt.gca().invert_yaxis() # Camera Z is forward, so invert to look like a map
    plt.grid()
    # plt.show()
    print("Simple SLAM plot ready (plt.show() disabled for automation)")

if __name__ == "__main__":
    run_slam_sequence()
