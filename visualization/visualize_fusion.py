import numpy as np
import open3d as o3d
import cv2
import sys
import os

# Interactive Visualization
def visualize_results(pcd_path, img_path):
    print("Loading visualization...")
    
    # 1. Point Cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        print(f"Error: Point cloud {pcd_path} is empty or not found.")
        return

    # 2. Image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Image {img_path} not found.")
        return
        
    cv2.imshow("Fused Image (Press Q to exit)", img)
    
    # Open3D Visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Colored Point Cloud", width=800, height=600)
    vis.add_geometry(pcd)
    
    # Add coordinates frame
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
    
    # Render loop
    while True:
        vis.poll_events()
        vis.update_renderer()
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
            
    vis.destroy_window()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # If run directly, define defaults
    pcd = "outputs/fusion_results/colored_pointcloud.pcd"
    img = "outputs/fusion_results/fused_frame.png"
    
    if os.path.exists(pcd) and os.path.exists(img):
        visualize_results(pcd, img)
    else:
        print("Outputs not found. Run fusion/camera_lidar_fusion.py first.")
