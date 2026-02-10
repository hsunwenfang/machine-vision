import cv2
import numpy as np
import os
import itertools

def find_best_pattern(image_path):
    print(f"Checking {image_path}")
    if not os.path.exists(image_path):
        print("File not found")
        return

    img = cv2.imread(image_path, 0)
    if img is None:
        print("Failed to load image")
        return

    rows = range(3, 14)
    cols = range(3, 14)
    
    # Generate all combinations (r, c)
    patterns = list(itertools.product(rows, cols))
    # Sort by size (largest first)
    patterns.sort(key=lambda p: p[0]*p[1], reverse=True)
    
    print(f"Scanning {len(patterns)} pattern combinations...")
    
    best_pattern = None
    
    for pat in patterns:
        ret, corners = cv2.findChessboardCorners(img, pat, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            print(f"SUCCESS: Found pattern {pat} (Points: {pat[0]*pat[1]})")
            best_pattern = pat
            break
            
    if best_pattern:
        print(f"-> Max detected pattern size: {best_pattern}")
    else:
        print("-> No pattern found.")

if __name__ == "__main__":
    # Test on one left and one right image
    base = '/Users/hsunwenfang/Documents/machine-vision/data/2011_09_26_calib_img/2011_09_26_drive_0119_extract'
    
    img0 = os.path.join(base, 'image_00', 'data', '0000000000.png')
    img1 = os.path.join(base, 'image_01', 'data', '0000000000.png')
    
    print("--- Left Camera 00 ---")
    find_best_pattern(img0)
    
    print("\n--- Right Camera 01 ---")
    find_best_pattern(img1)
