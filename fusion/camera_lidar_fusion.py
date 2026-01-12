import numpy as np
import cv2
import open3d as o3d
import sys
import os

# Add root to python path to import transformations
sys.path.append(os.getcwd())

from transformations.coordinate_frames import SensorFrameManager
from transformations.projection import project_lidar_to_image

class LidarCameraFuser:
    def __init__(self, intrinsic_yaml):
        import yaml
        with open(intrinsic_yaml, 'r') as f:
            data = yaml.safe_load(f)
            
        self.K = np.array([
            [data['fx'], 0, data['cx']],
            [0, data['fy'], data['cy']],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist = np.array([data['k1'], data['k2'], data['p1'], data['p2'], data['k3']], dtype=np.float32)
        self.frames = SensorFrameManager()

    def fuse(self, img, pcd_path):
        """
        Projects LiDAR points onto image and returns:
        1. Image with overlaid points
        2. Colored PointCloud
        """
        # Read Point Cloud
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        
        # Get Transform
        T_cam_lidar = self.frames.get_transform("lidar", "camera") # Lidar -> Camera
        
        # Project
        h, w = img.shape[:2]
        pts_2d, indices, pts_cam_3d = project_lidar_to_image(points, T_cam_lidar, self.K, self.dist, (h, w))
        
        # 1. Overlay on Image
        img_vis = img.copy()
        for i, pt in enumerate(pts_2d):
            depth = pts_cam_3d[i, 2]
            color = self._get_depth_color(depth)
            cv2.circle(img_vis, (int(pt[0]), int(pt[1])), 2, color, -1)
            
        # 2. Color Point Cloud
        # We need to color the ORIGINAL points based on the image color at the projected pixel
        colors = np.zeros_like(points)
        
        # For valid indices
        for i, idx in enumerate(indices):
            u, v = int(pts_2d[i, 0]), int(pts_2d[i, 1])
            # BGR to RGB (Open3D uses RGB 0-1)
            bgr = img[v, u]
            rgb = bgr[::-1] / 255.0
            colors[idx] = rgb
            
        # For points not in FoV, keep them grey or black
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return img_vis, pcd

    def _get_depth_color(self, depth, min_d=0, max_d=20):
        # Rainbow map
        val = max(0, min(1, (depth - min_d) / (max_d - min_d)))
        # Simple R-G-B transition
        import matplotlib.cm
        rgba = matplotlib.cm.jet(val)
        return (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))

if __name__ == "__main__":
    # Test
    import glob
    img_files = glob.glob("data/camera/images/img_*.png")
    pcd_files = glob.glob("data/lidar/pointclouds/cloud_*.pcd")
    
    if img_files and pcd_files:
        fuser = LidarCameraFuser("data/camera/intrinsics.yaml")
        img = cv2.imread(img_files[0])
        fused_img, colored_pcd = fuser.fuse(img, pcd_files[0])
        
        cv2.imwrite("outputs/fusion_results/fused_frame.png", fused_img)
        o3d.io.write_point_cloud("outputs/fusion_results/colored_pointcloud.pcd", colored_pcd)
        print("Fusion test complete. Results saved.")
