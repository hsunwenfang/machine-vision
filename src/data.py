import cv2
import numpy as np
import os
import glob

class KittiLoader:
    def __init__(self, base_dir, date='2011_09_26', drive='0001', mode='gray'):
        """
        Loader for KITTI Raw Data.
        structure: date_drive_extract/image_xx/data/*.png
        """
        self.mode = mode
        self.drive_path = os.path.join(base_dir, f'{date}_data', f'{date}_drive_{drive}_extract')
        self.calib_path = os.path.join(base_dir, f'{date}_calib')
        
        if mode == 'gray':
            self.left_folder = os.path.join(self.drive_path, 'image_00', 'data')
            self.right_folder = os.path.join(self.drive_path, 'image_01', 'data')
            self.cam_id = '00' # Reference for calibration
        else: # color
            self.left_folder = os.path.join(self.drive_path, 'image_02', 'data')
            self.right_folder = os.path.join(self.drive_path, 'image_03', 'data')
            self.cam_id = '02'

        self.left_images = sorted(glob.glob(os.path.join(self.left_folder, '*.png')))
        self.right_images = sorted(glob.glob(os.path.join(self.right_folder, '*.png')))
        
        self.load_calibration()

    def load_calibration(self):
        calib_file = os.path.join(self.calib_path, 'calib_cam_to_cam.txt')
        if not os.path.exists(calib_file):
            print("Warning: KITTI calib file not found.")
            self.K, self.b = np.eye(3), 0.54
            return

        data = {}
        with open(calib_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, val = line.strip().split(':', 1)
                    vals = val.split()
                    if all(v.replace('.','').replace('-','').replace('+','').replace('e','').replace('E','').isdigit() for v in vals if v):
                        data[key] = np.array([float(x) for x in vals])

        # Determine Right Cam ID (01 for Gray, 03 for Color)
        right_id = '01' if self.cam_id == '00' else '03'

        self.K_00 = data[f'K_{self.cam_id}'].reshape(3,3)
        self.D_00 = data[f'D_{self.cam_id}']
        self.R_rect_00 = data[f'R_rect_{self.cam_id}'].reshape(3,3)
        self.P_rect_00 = data[f'P_rect_{self.cam_id}'].reshape(3,4)

        self.K_01 = data[f'K_{right_id}'].reshape(3,3)
        self.D_01 = data[f'D_{right_id}']
        self.R_rect_01 = data[f'R_rect_{right_id}'].reshape(3,3)
        self.P_rect_01 = data[f'P_rect_{right_id}'].reshape(3,4)

        # Calibration for SLAM (Used after rectification - ideal pinhole)
        self.K = self.P_rect_00[:3, :3]

        # Baseline B = Tx / fx
        fx = self.P_rect_00[0,0]
        self.b = -self.P_rect_01[0,3] / fx

        # Precompute Undistort/Rectify Maps
        s_raw = data[f'S_{self.cam_id}']
        size = (int(s_raw[0]), int(s_raw[1]))

        self.map_l_1, self.map_l_2 = cv2.initUndistortRectifyMap(
            self.K_00, self.D_00, self.R_rect_00, self.P_rect_00, size, cv2.CV_16SC2
        )

        self.map_r_1, self.map_r_2 = cv2.initUndistortRectifyMap(
            self.K_01, self.D_01, self.R_rect_01, self.P_rect_01, size, cv2.CV_16SC2
        )


    def __len__(self):
        return len(self.left_images)

    def get_frame(self, idx):
        if idx >= len(self.left_images): return None
        
        img_l = cv2.imread(self.left_images[idx], cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(self.right_images[idx], cv2.IMREAD_GRAYSCALE)
        
        # Apply rectification maps (Raw -> Rectified)
        img_l = cv2.remap(img_l, self.map_l_1, self.map_l_2, cv2.INTER_LINEAR)
        img_r = cv2.remap(img_r, self.map_r_1, self.map_r_2, cv2.INTER_LINEAR)

        return {
            'image_left': img_l,
            'image_right': img_r,
            'K': self.K,
            'baseline': self.b,
            'timestamp': idx * 0.1 # 10Hz approx
        }


# ---------------------------------------------------------------------------
# Ground Truth loader (KITTI OXTS → GPS trajectory synced to image timestamps)
# ---------------------------------------------------------------------------

def load_gt(base_dir, date, drive):
    """Load GPS ground truth, synchronised to image timestamps.

    Returns Nx3 array of (X, Z, Y) positions in a local ENU frame,
    one per image frame.
    """
    from datetime import datetime

    oxts_path = os.path.join(base_dir, f'{date}_data',
                             f'{date}_drive_{drive}_extract', 'oxts', 'data')
    img_ts_path = os.path.join(base_dir, f'{date}_data',
                               f'{date}_drive_{drive}_extract', 'image_00', 'timestamps.txt')
    oxts_ts_path = os.path.join(base_dir, f'{date}_data',
                                f'{date}_drive_{drive}_extract', 'oxts', 'timestamps.txt')

    assert os.path.exists(img_ts_path), f"Image timestamps not found: {img_ts_path}"
    assert os.path.exists(oxts_ts_path), f"OXTS timestamps not found: {oxts_ts_path}"

    def parse_ts(path):
        with open(path, 'r') as f:
            lines = f.readlines()
        ts = []
        for l in lines:
            dt = datetime.strptime(l.strip()[:-4], "%Y-%m-%d %H:%M:%S.%f")
            ts.append(dt.timestamp())
        return np.array(ts)

    img_ts = parse_ts(img_ts_path)
    oxts_ts = parse_ts(oxts_ts_path)

    oxts_indices = [int(np.abs(oxts_ts - t).argmin()) for t in img_ts]

    gt_traj = []
    origin = None
    er = 6378137.0  # WGS-84 semi-major axis

    for idx in oxts_indices:
        fname = f"{idx:010d}.txt"
        fpath = os.path.join(oxts_path, fname)
        assert os.path.exists(fpath), f"OXTS file missing: {fpath}"

        with open(fpath, 'r') as file:
            line = file.readline().split()
            lat, lon, alt = float(line[0]), float(line[1]), float(line[2])

            x = lon * np.pi * er / 180.0 * np.cos(lat * np.pi / 180.0)
            y = lat * np.pi * er / 180.0
            z = alt

            if origin is None:
                origin = np.array([x, y, z])

            dx = x - origin[0]
            dy = y - origin[1]
            dz = z - origin[2]
            gt_traj.append([dx, dz, dy])

    return np.array(gt_traj)