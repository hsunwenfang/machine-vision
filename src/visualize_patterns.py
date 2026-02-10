import cv2
import numpy as np
import os

def visualize_comparison():
    base_dir = '/Users/hsunwenfang/Documents/machine-vision/data/2011_09_26_calib_img/2011_09_26_drive_0119_extract/image_01/data'
    img_name = '0000000000.png'
    img_path = os.path.join(base_dir, img_name)
    
    if not os.path.exists(img_path):
        print("Image not found")
        return

    # Load image
    img_orig = cv2.imread(img_path)
    if img_orig is None:
        print("Failed to load")
        return

    # 1. Detect Small Pattern (5, 7)
    img_small = img_orig.copy()
    ret_s, corners_s = cv2.findChessboardCorners(cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY), (5, 7), None)
    if ret_s:
        cv2.drawChessboardCorners(img_small, (5, 7), corners_s, ret_s)
        cv2.putText(img_small, "Small Pattern (5x7) - Covered Region", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 2. Detect Large Pattern (7, 11)
    img_large = img_orig.copy()
    ret_l, corners_l = cv2.findChessboardCorners(cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY), (7, 11), None)
    if ret_l:
        cv2.drawChessboardCorners(img_large, (7, 11), corners_l, ret_l)
        cv2.putText(img_large, "Large Pattern (7x11) - Covered Region", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 3. Stack Side-by-Side
    # Resize for better viewing if needed, but side-by-side preserves scale
    combined = np.hstack((img_small, img_large))
    
    # Draw Distortion Circles (Conceptual) to show why edges matter
    # Center of image
    h, w = img_orig.shape[:2]
    cx, cy = w//2, h//2
    
    # Draw a circle representing the "Linear Region" vs "Distortion Region"
    cv2.circle(combined, (cx, cy), 200, (255, 255, 0), 2) # Inner "Safe" zone
    cv2.circle(combined, (cx+w, cy), 200, (255, 255, 0), 2)
    
    output_path = '/Users/hsunwenfang/Documents/machine-vision/doc/pattern_comparison_vis.png'
    cv2.imwrite(output_path, combined)
    print(f"Comparison image saved to: {output_path}")

if __name__ == "__main__":
    visualize_comparison()
