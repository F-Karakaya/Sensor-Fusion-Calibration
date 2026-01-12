import numpy as np
import cv2

def project_lidar_to_image(lidar_points, T_cam_lidar, camera_matrix, dist_coeffs=None, image_shape=None):
    """
    Projects 3D LiDAR points onto the 2D image plane.
    
    Args:
        lidar_points (np.ndarray): Nx3 array of points in LiDAR frame.
        T_cam_lidar (np.ndarray): 4x4 transformation matrix from LiDAR to Camera.
        camera_matrix (np.ndarray): 3x3 camera intrinsic matrix.
        dist_coeffs (np.ndarray): Distortion coefficients (k1, k2, p1, p2, k3).
        image_shape (tuple): (height, width) to filter points outside image.
        
    Returns:
        points_2d (np.ndarray): Mx2 array of projected 2D points.
        indices (np.ndarray): Indices of valid points (indices into original lidar_points).
    """
    if len(lidar_points) == 0:
        return np.array([]), np.array([])
        
    # 1. Transform points to Camera Frame
    # Homogeneous coordinates
    N = lidar_points.shape[0]
    pts_hom = np.hstack((lidar_points, np.ones((N, 1))))
    
    # T_cam_lidar transforms points FROM lidar TO camera
    pts_cam = (T_cam_lidar @ pts_hom.T).T # Nx4
    pts_cam_3d = pts_cam[:, :3]
    
    # 2. Filter points behind the camera (z <= 0)
    valid_z_mask = pts_cam_3d[:, 2] > 0
    
    # 3. Project to Image Plane (using OpenCV for distortion handling)
    # cv2.projectPoints expects Nx3 (Nx1x3 or Nx3)
    # rvec=0, tvec=0 because we already transformed to camera frame
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)
        
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    
    # We only project points that have positive Z to avoid weird back-projection artifacts 
    # handled by OpenCV? No, OpenCV might project behind points incorrectly or give garbage.
    # It's safer to filter first or pass all and filter later.
    # But for optimization, let's pass all and filter by mask.
    
    img_points, _ = cv2.projectPoints(pts_cam_3d, rvec, tvec, camera_matrix, dist_coeffs)
    img_points = img_points.squeeze() # Nx2
    
    # 4. Filter points outside image bounds
    if image_shape is not None:
        h, w = image_shape
        valid_x = (img_points[:, 0] >= 0) & (img_points[:, 0] < w)
        valid_y = (img_points[:, 1] >= 0) & (img_points[:, 1] < h)
        mask = valid_z_mask & valid_x & valid_y
    else:
        mask = valid_z_mask

    points_2d = img_points[mask]
    indices = np.where(mask)[0]
    
    return points_2d, indices, pts_cam_3d[mask] # Return 3D points in camera frame too for depth info
