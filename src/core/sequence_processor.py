import cv2
import os
from tqdm import tqdm
import numpy as np

class SequenceProcessor:
    def __init__(self, loader, projector, labels):
        self.loader = loader
        self.projector = projector
        self.labels = labels

    def process_sequence(self, output_path, num_frames=154):
        # 1. Get metadata from first frame to set video size
        sample_img = self.loader.load_image(0)
        h, w, _ = sample_img.shape
        
        # 2. Initialize Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 10.0, (w, h))
        
        print(f"🎬 Processing {num_frames} frames with Enhanced Fusion...")
        
        for i in tqdm(range(num_frames)):
            img = self.loader.load_image(i)
            pc = self.loader.load_lidar(i)
            boxes = self.labels.get_boxes_for_frame(i)
            
            # --- PHASE 1: LiDAR Projection with Ground Filter ---
            pixels, depths = self.projector.project_to_image(pc)
            
            # THE GROUND FILTER:
            # In KITTI LiDAR coordinates, Z is height. 
            # Sensor is ~1.7m up, so Z > -1.55 removes the asphalt.
            ground_mask = pc[:, 2] > -1.55 
            
            mask = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & \
                   (pixels[:, 1] >= 0) & (pixels[:, 1] < h) & \
                   (depths > 2) & ground_mask
            
            valid_px = pixels[mask].astype(int)
            valid_d = depths[mask]
            
            for j in range(len(valid_px)):
                x, y = valid_px[j]
                color_val = int(min(valid_d[j], 40) / 40 * 255)
                cv2.circle(img, (x, y), 1, (255 - color_val, color_val, 255), -1)
            
            # --- PHASE 2: 3D Boxes + Distance Labels ---
            for box in boxes:
                corners_3d = self.projector.get_3d_box_corners(box)
                corners_2d, depths_box = self.projector.project_to_image(corners_3d)
                
                if np.all(depths_box > 0):
                    # Set Color
                    if box['type'] == 'Car':
                        color = (0, 255, 0) # Green
                    elif box['type'] in ['Pedestrian', 'Cyclist']:
                        color = (255, 255, 0) # Yellow
                    else:
                        color = (255, 0, 0) # Blue
                    
                    # 1. Draw the Wireframe
                    img = self.projector.draw_3d_box(img, corners_2d, color=color, thickness=2)
                    
                    # 2. Draw the Label (using the new method we'll add to projector)
                    img = self.projector.draw_label(img, box, corners_2d)
            
            out.write(img)
            
        out.release()
        print(f"✅ Video saved to {output_path}")