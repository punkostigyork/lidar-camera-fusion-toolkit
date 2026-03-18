import cv2
import numpy as np

class Projector:
    def __init__(self, calib):
        self.calib = calib

    def project_to_image(self, points_lidar):
        # 1. Extract XYZ and add 1s for Homogeneous Coordinates
        pts_3d = points_lidar[:, :3]
        pts_3d_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))

        # 2. The Full Projection Chain
        # Pixel = P2 * R0_rect * Tr_velo_to_cam * P_lidar
        matrix = self.calib.P2 @ self.calib.R0_rect @ self.calib.velo_to_cam
        pts_2d_hom = pts_3d_hom @ matrix.T

        # 3. Depth check (Points must be in front of camera)
        # We need the depth in the camera frame
        pts_cam = pts_3d_hom @ (self.calib.R0_rect @ self.calib.velo_to_cam).T
        depth = pts_cam[:, 2]
        
        # 4. Perspective Division
        u = pts_2d_hom[:, 0] / pts_2d_hom[:, 2]
        v = pts_2d_hom[:, 1] / pts_2d_hom[:, 2]

        return np.column_stack((u, v)), depth
    
    def get_3d_box_corners(self, box):
        h, w, l = box['dims']
        tx, ty, tz = box['pos']
        yaw = box['yaw']

        # 1. Define corners in object coordinates (centered at base)
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        z_corners = [0, 0, 0, 0, h, h, h, h] # Box starts at ground (z=0)

        # 2. Rotate by Yaw (around Z-axis in LiDAR frame)
        corners_3d = np.vstack([x_corners, y_corners, z_corners])
        rot_mat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,           1]
        ])
        corners_3d = rot_mat @ corners_3d

        # 3. Translate to object position
        corners_3d[0, :] += tx
        corners_3d[1, :] += ty
        corners_3d[2, :] += tz

        return corners_3d.T # Returns (8, 3)

    def draw_3d_box(self, img, corners_2d, color=(0, 255, 0), thickness=2):
        """
        Connects the 8 projected corners to draw a 3D wireframe box.
        Corners are assumed to be in the order: 
        Base: [0, 1, 2, 3], Top: [4, 5, 6, 7]
        """
        corners_2d = corners_2d.astype(int)

        # Draw the base (bottom rectangle)
        for i in range(4):
            cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[(i+1)%4]), color, thickness)

        # Draw the top (top rectangle)
        for i in range(4):
            cv2.line(img, tuple(corners_2d[i+4]), tuple(corners_2d[((i+1)%4)+4]), color, thickness)

        # Draw the vertical pillars connecting base to top
        for i in range(4):
            cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[i+4]), color, thickness)

        return img

    def draw_label(self, img, box, corners_2d):
        """Draws object type and distance at the top of the 3D box."""
        # Find the highest point of the box in the image (min Y coordinate)
        top_y = np.min(corners_2d[:, 1])
        center_x = np.mean(corners_2d[:, 0])
        
        # Calculate Euclidean distance from sensor to object
        dist = np.linalg.norm(box['pos'])
        
        label = f"{box['type']} | {dist:.1f}m"
        
        # Draw a small background rectangle for readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, scale, thickness)
        
        cv2.rectangle(img, 
                      (int(center_x - label_w/2), int(top_y - label_h - 10)),
                      (int(center_x + label_w/2), int(top_y - 5)), 
                      (0, 0, 0), -1)
        
        cv2.putText(img, label, (int(center_x - label_w/2), int(top_y - 10)), 
                    font, scale, (255, 255, 255), thickness)
        return img

    def generate_bev(self, boxes, pc=None, width=400, height=600, scale=10):
        """
        Creates a Top-Down Bird's Eye View map.
        scale: pixels per meter (10 means 1 meter = 10 pixels)
        """
        # 1. Create a black canvas (Height represents Depth/X, Width represents Lateral/Y)
        bev_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 2. Set the "Origin" (Our car's position)
        # Center horizontally, and 50 pixels from the bottom
        origin_y = width // 2
        origin_x = height - 50 

        # Draw horizontal lines every 10 meters to act as a scale
        for dist in range(10, 70, 10): # 10m, 20m, 30m, etc.
            py = int(origin_x - (dist * scale))
            if 0 <= py < height:
                # Draw a dark grey line (50, 50, 50)
                cv2.line(bev_img, (0, py), (width, py), (50, 50, 50), 1)
                # Add text for the distance
                cv2.putText(bev_img, f"{dist}m", (5, py - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        
        # 3. Draw Lidar points (Optional background)
        if pc is not None:
            # Subsample points for performance (every 5th point)
            points = pc[::5]
            # Map Lidar X (forward) to BEV vertical, Lidar Y (left) to BEV horizontal
            for pt in points:
                lx, ly, lz = pt[:3]
                
                # Convert meters to pixels
                # ly is positive to the left, so we subtract from origin_y
                px = int(origin_y - (ly * scale))
                py = int(origin_x - (lx * scale))
                
                if 0 <= px < width and 0 <= py < height:
                    # Draw a dim grey dot for the environment
                    bev_img[py, px] = [80, 80, 80]

        # 4. Draw the Boxes
        for box in boxes:
            tx, ty, tz = box['pos'] # Position in LiDAR frame
            l, w, h = box['dims']
            yaw = box['yaw']

            # Calculate center in pixels
            cx = int(origin_y - (ty * scale))
            cy = int(origin_x - (tx * scale))

            # Define the 2D footprint corners (in meters, centered at 0,0)
            # This accounts for the Yaw/Heading of the car
            rect_corners = np.array([
                [l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]
            ])
            
            # Rotate footprint by yaw
            c, s = np.cos(yaw), np.sin(yaw)
            rot_mat = np.array([[c, -s], [s, c]])
            rect_corners = (rot_mat @ rect_corners.T).T

            # Convert footprint to pixels and shift to center
            pixel_corners = []
            for corner in rect_corners:
                # Map back to our BEV coordinate system
                # l (forward/x) translates to vertical, w (lateral/y) translates to horizontal
                pc_x = int(cx - (corner[1] * scale))
                pc_y = int(cy - (corner[0] * scale))
                pixel_corners.append([pc_x, pc_y])
            
            # Draw the rotated rectangle
            color = (0, 255, 0) if box['type'] == 'Car' else (255, 255, 0)
            cv2.fillPoly(bev_img, [np.array(pixel_corners)], color)

        # 5. Draw "Our Car" (The ego vehicle)
        # Represented as a small white triangle at the origin
        ego_pts = np.array([
            [origin_y - 5, origin_x], [origin_y + 5, origin_x], [origin_y, origin_x - 10]
        ])
        cv2.drawContours(bev_img, [ego_pts], 0, (255, 255, 255), -1)

        return bev_img

    def overlay_bev(self, main_img, bev_img, margin=20):
        """Standardizes the 'Picture-in-Picture' look with auto-resizing."""
        h_main, w_main, _ = main_img.shape
        
        # 1. Add a white border first
        bev_with_border = cv2.copyMakeBorder(bev_img, 2, 2, 2, 2, 
                                            cv2.BORDER_CONSTANT, value=[255, 255, 255])
        h_b, w_b, _ = bev_with_border.shape

        # 2. Check if the BEV is too tall or too wide for the main image
        # We want it to take up at most 70% of the main image height
        max_h = int(h_main * 0.7)
        if h_b > max_h:
            scale_factor = max_h / h_b
            new_w = int(w_b * scale_factor)
            new_h = int(h_b * scale_factor)
            bev_with_border = cv2.resize(bev_with_border, (new_w, new_h))
            h_b, w_b = new_h, new_w

        # 3. Calculate position (Top-Right corner)
        y_offset = margin
        x_offset = w_main - w_b - margin
        
        # Double check that we aren't out of bounds after calculations
        if x_offset < 0 or y_offset + h_b > h_main:
            # If it still doesn't fit, just return the original image
            return main_img

        # 4. Paste it!
        main_img[y_offset:y_offset+h_b, x_offset:x_offset+w_b] = bev_with_border
        return main_img