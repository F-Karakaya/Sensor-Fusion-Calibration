import numpy as np
import cv2
import os
import open3d as o3d
import yaml
import csv

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_checkerboard_images(output_dir, num_images=10, rows=7, cols=7, square_size=50):
    create_directory(output_dir)
    pattern_size = (rows, cols)
    
    # Define 3D object points in world coordinate (z=0)
    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:,:2] = np.mgrid[0:rows, 0:cols].T.reshape(-1,2) * square_size

    # Camera matrix (simulated)
    width, height = 1280, 720
    fx, fy = 1000, 1000
    cx, cy = width/2, height/2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5,1)) # Zero distortion for simplicity

    print(f"Generating {num_images} synthetic checkerboard images...")

    for i in range(num_images):
        # Create a blank image
        img_size = (height, width) # H, W
        img = np.zeros(img_size, dtype=np.uint8) + 50 # Gray background

        # Rotation and translation for the object
        # Random rotation
        rvec = np.random.rand(3) * 0.5 
        tvec = np.array([0, 0, 1000 + i*100], dtype=np.float32) # Move away in Z

        # Project points
        imgpts, _ = cv2.projectPoints(objp, rvec, tvec, K, dist_coeffs)
        imgpts = np.int32(imgpts).reshape(-1,2)

        # Draw the chessboard on the image manually or use a simpler approach
        # A simpler approach: Just draw a flat chessboard and warp it? 
        # Or better: Draw a perfect 2D chessboard and let that be "image 0", then warp for others.
        # For simplicity in this script, let's just generate PURE 2D chessboards that are valid.
        # We will generate a base board.
        
        base_board = np.zeros((height, width), dtype=np.uint8) + 255
        margin = 100
        start_x = (width - (cols * square_size)) // 2
        start_y = (height - (rows * square_size)) // 2
        
        # Draw squares
        for r in range(rows + 1):
            for c in range(cols + 1):
                if (r+c) % 2 == 1:
                    pt1 = (start_x + c*square_size, start_y + r*square_size)
                    pt2 = (start_x + (c+1)*square_size, start_y + (r+1)*square_size)
                    cv2.rectangle(base_board, pt1, pt2, (0,0,0), -1)
        
        # Warp it to simulate different views
        # Random homography
        src_pts = np.float32([[0,0], [width,0], [width,height], [0,height]])
        dst_pts = src_pts + np.random.uniform(-50, 50, size=(4,2)).astype(np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(base_board, M, (width, height))
        
        filename = os.path.join(output_dir, f"img_{i:02d}.png")
        cv2.imwrite(filename, warped)

    print("Images generated.")

def generate_point_clouds(output_dir, num_clouds=5):
    create_directory(output_dir)
    print(f"Generating {num_clouds} synthetic point clouds...")
    
    for i in range(num_clouds):
        # Generate random points in a box
        points = np.random.rand(1000, 3) * 10 - 5 # -5 to 5 range
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        filename = os.path.join(output_dir, f"cloud_{i:02d}.pcd")
        o3d.io.write_point_cloud(filename, pcd)
    print("Point clouds generated.")

def generate_imu_log(output_file):
    print("Generating IMU log...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "ax", "ay", "az", "gx", "gy", "gz"])
        
        t = 0.0
        for _ in range(1000):
            row = [
                f"{t:.3f}",
                np.random.normal(0, 0.1), np.random.normal(0, 0.1), np.random.normal(9.8, 0.1),
                np.random.normal(0, 0.01), np.random.normal(0, 0.01), np.random.normal(0, 0.01)
            ]
            writer.writerow(row)
            t += 0.01
    print("IMU log generated.")

def generate_configs():
    # Intrinsics
    intrinsics = {
        'fx': 1000.0, 'fy': 1000.0,
        'cx': 640.0, 'cy': 360.0,
        'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0, 'k3': 0.0
    }
    with open('data/camera/intrinsics.yaml', 'w') as f:
        yaml.dump(intrinsics, f)
    
    # Lidar Specs
    specs = {
        'range_min': 0.5, 'range_max': 100.0,
        'angular_resolution_h': 0.2,
        'angular_resolution_v': 2.0
    }
    with open('data/lidar/lidar_specs.yaml', 'w') as f:
        yaml.dump(specs, f)
    print("Config files generated.")

if __name__ == "__main__":
    generate_checkerboard_images("data/camera/images")
    generate_point_clouds("data/lidar/pointclouds")
    generate_imu_log("data/imu/imu_log.csv")
    generate_configs()
