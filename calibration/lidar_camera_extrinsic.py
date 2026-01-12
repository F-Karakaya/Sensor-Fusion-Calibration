import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class LidarCameraCalibrator:
    """
    Solves for the extrinsic transformation between LiDAR and Camera using paired 3D-2D or 3D-3D correspondences.
    """
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def solve_pnp(self, object_points_lidar, image_points):
        """
        Estimates the pose of the LiDAR frame relative to the Camera frame.
        
        Args:
            object_points_lidar (np.ndarray): Nx3 points in LiDAR frame.
            image_points (np.ndarray): Nx2 corresponding points in image plane.
            
        Returns:
            T_cam_lidar (np.ndarray): 4x4 transformation matrix.
        """
        # solvePnP finds the object pose (LiDAR frame) in Camera frame NO, wait.
        # solvePnP finds the object pose relative to camera.
        # object_points are in "Object Coordinate Space" (here, Lidar Frame).
        # It returns rvec, tvec that transforms Object -> Camera.
        
        if len(object_points_lidar) < 4:
            raise ValueError("Need at least 4 points for PnP.")
            
        success, rvec, tvec = cv2.solvePnP(object_points_lidar, image_points, 
                                           self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        
        if not success:
            raise RuntimeError("PnP solver failed.")
            
        R_mat, _ = cv2.Rodrigues(rvec)
        
        T_cam_lidar = np.eye(4)
        T_cam_lidar[:3, :3] = R_mat
        T_cam_lidar[:3, 3] = tvec.squeeze()
        
        return T_cam_lidar

    def solve_icp(self, lidar_points, camera_points_3d, initial_guess=np.eye(4)):
        """
        Refines transform using ICP if we have 3D points from camera (e.g. from stereo or known target geometry).
        
        Args:
            lidar_points (open3d.geometry.PointCloud): Source.
            camera_points_3d (open3d.geometry.PointCloud): Target.
        """
        reg_p2p = o3d.pipelines.registration.registration_icp(
            lidar_points, camera_points_3d, 0.5, initial_guess,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        return reg_p2p.transformation

if __name__ == "__main__":
    # Example usage (Dummy Data)
    # Assume we calibrated intrinsics already
    K = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros(5)
    
    calib = LidarCameraCalibrator(K, dist)
    
    # Ground Truth Transform (LiDAR -> Camera)
    # Let's say Camera is at [0.1, 0, -0.1] relative to Lidar
    # And rotated.
    # We will simulate points in Lidar frame, transform them to pixel, and try to recover the transform.
    
    # GT Transform
    # Cam Z is forward. Lidar X is forward.
    # Rotation: Cam Z aligned with Lidar X.
    # R_cam_lidar (Lidar -> Cam)
    # If point is [1, 0, 0] in Lidar (1m fwd), it should be approx [0, 0, 1] in Cam.
    R_gt = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]) # X->Z, Y->-X, Z->-Y (Standard robotics frame conversion)
    t_gt = np.array([0.0, 0.0, 0.0]) # Co-located
    
    # Generate random 3D points in Lidar frame
    N = 10
    pts_lidar = np.random.rand(N, 3) * 5 + np.array([2, -1, -1]) # Points 2m to 7m in front
    
    # Project to Image
    pts_cam = (R_gt @ pts_lidar.T).T + t_gt
    img_pts, _ = cv2.projectPoints(pts_cam, np.zeros(3), np.zeros(3), K, dist)
    img_pts = img_pts.squeeze()
    
    # Recover Pose
    print("Running PnP to recover extrinsic...")
    try:
        T_est = calib.solve_pnp(pts_lidar, img_pts)
        print("Estimated Transform:\n", T_est)
        print("Ground Truth Rotation:\n", R_gt)
        print("Estimated Rotation:\n", T_est[:3,:3])
        
        # Save result
        np.save("outputs/calibration_results/extrinsic_matrix.npy", T_est)
    except Exception as e:
        print(e)
