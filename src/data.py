import cv2
import numpy as np

def generate_scaled_transform(image, scale_factor, tx_base=10, K=None):
    """
    Simulates a camera translation by warping the image.
    This creates a 'synthetic' second frame where we KNOW the translation magnitude.
    
    Args:
        image: Source image
        scale_factor: Multiplier for the translation (simulates speed or physical scale)
        tx_base: Base translation in pixels
        K: Camera intrinsic matrix
    """
    h, w = image.shape[:2]
    
    # Simulate a pure translation along the X-axis (for simplicity in testing)
    # In a real 3D world, Z-translation changes size, X/Y translation shifts pixels.
    # Here we simulate a shift which corresponds to camera motion.
    
    # Translation vector in pixels
    tx = tx_base * scale_factor
    
    # Create transformation matrix (2x3 for affine warp)
    M = np.float32([[1, 0, -tx], [0, 1, 0]])
    
    # Warped image
    shifted_image = cv2.warpAffine(image, M, (w, h))
    
    return shifted_image, tx

def get_virtual_stereo_pair(img_path, baseline_scale=1.0):
    """
    Loads an image and creates a synthetic right view to simulate a stereo pair
    with a specific baseline (scale).
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load {img_path}")
        
    # Create a synthetic second view
    # A larger baseline_scale means the camera moved 'further'
    img_right, true_translation_px = generate_scaled_transform(img, baseline_scale)
    
    return img, img_right, true_translation_px

if __name__ == "__main__":
    # Test
    path = "/Users/hsunwenfang/Documents/machine-vision/data/2011_09_26_data/2011_09_26_drive_0001_extract/image_02/data/0000000000.png"
    i1, i2, tx = get_virtual_stereo_pair(path, baseline_scale=2.0)
    print(f"Generated pair with simulated pixel shift: {tx}")
    # cv2.imshow("Original", i1)
    # cv2.imshow("Synthetic Shift", i2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
