import numpy as np

def to_homogeneous(points):
    """Converts (N, 3) points to (N, 4) by adding a column of ones."""
    return np.hstack((points, np.ones((points.shape[0], 1))))

def from_homogeneous(points):
    """Converts (N, 4) or (N, 3) back to Euclidean by dividing by the last column."""
    return points[:, :-1] / points[:, -1:]

class Projector:
    def __init__(self, calib: KittiCalib):
        self.calib = calib

    def lidar_to_camera(self, pts_lidar):
        """Transforms points from LiDAR frame to Rectified Camera frame."""
        pts_h = to_homogeneous(pts_lidar)
        # Apply Extrinsics then Rectification
        pts_cam = pts_h @ self.calib.Tr_velo_to_cam.T
        pts_rect = pts_cam @ self.calib.R0_rect.T
        return pts_rect