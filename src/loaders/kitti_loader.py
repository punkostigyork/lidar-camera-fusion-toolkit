import numpy as np
import cv2
import os

class KittiCalib:
    def __init__(self, calib_dir):
        self.velo_to_cam = self._parse_velo_to_cam(f"{calib_dir}/calib_velo_to_cam.txt")
        cam_to_cam = self._parse_cam_to_cam(f"{calib_dir}/calib_cam_to_cam.txt")
        
        self.P2 = cam_to_cam['P2']
        self.R0_rect = cam_to_cam['R0_rect']

    def _parse_velo_to_cam(self, file_path):
        data = {}
        with open(file_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, val = line.split(':', 1)
                    try:
                        float_vals = np.array([float(x) for x in val.split()])
                        if len(float_vals) > 0:
                            data[key.strip()] = float_vals
                    except ValueError:
                        continue
        
        if 'R' in data and 'T' in data:
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = data['R'].reshape(3, 3)
            extrinsic[:3, 3] = data['T']
            return extrinsic
        
        for key in data.keys():
            if 'Tr' in key:
                extrinsic = np.eye(4)
                extrinsic[:3, :4] = data[key].reshape(3, 4)
                return extrinsic
        raise ValueError(f"Could not find valid R/T or Tr numeric data in {file_path}")

    def _parse_cam_to_cam(self, file_path):
        data = {}
        with open(file_path, 'r') as f:
            for line in f:
                if ':' not in line: continue
                key, val = line.split(':', 1)
                try:
                    float_vals = np.array([float(x) for x in val.split()])
                    if len(float_vals) > 0:
                        data[key.strip()] = float_vals
                except ValueError:
                    continue

        p_key = 'P2' if 'P2' in data else 'P_rect_02'
        r_key = 'R0_rect' if 'R0_rect' in data else 'R_rect_00'

        if p_key not in data or r_key not in data:
            raise KeyError(f"Missing matrices in {file_path}. Found: {list(data.keys())}")

        p2 = data[p_key].reshape(3, 4)
        r0_rect = np.eye(4)
        r0_rect[:3, :3] = data[r_key].reshape(3, 3)
        return {'P2': p2, 'R0_rect': r0_rect}

class KittiLoader:
    def __init__(self, frame_dir):
        self.frame_dir = frame_dir

    def load_lidar(self, frame_idx):
        bin_path = os.path.join(self.frame_dir, "velodyne_points/data", f"{frame_idx:010d}.bin")
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"LiDAR file not found: {bin_path}")
        return np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))

    def load_image(self, frame_idx):
        img_path = os.path.join(self.frame_dir, "image_02/data", f"{frame_idx:010d}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        return cv2.imread(img_path)