import numpy as np
from .homogeneous_transform import create_homogeneous_matrix, euler_to_matrix

class SensorFrameManager:
    """
    Manages coordinate frame transformations between World, IMU, LiDAR, and Camera.
    
    Standard frames:
    - World: Fixed arbitrary origin (e.g. at start position).
    - IMU: Body frame (x-forward, y-left, z-up).
    - LiDAR: Sensor frame (typically x-forward, y-left, z-up or similar).
    - Camera: Optical frame (x-right, y-down, z-forward).
    """
    def __init__(self):
        # Initialize identity transforms
        self.T_world_imu = np.eye(4)
        self.T_imu_lidar = np.eye(4)
        self.T_lidar_camera = np.eye(4)
        
        # Ground Truth / Initial Guesses (Simulating a setup)
        # Lidar is 0.2m above IMU, Camera is 0.1m below Lidar and slightly forward
        self._init_default_extrinsics()

    def _init_default_extrinsics(self):
        # IMU to LiDAR (Lidar is mounted on top of IMU)
        R_imu_lidar = np.eye(3) # Aligned
        t_imu_lidar = np.array([0.0, 0.0, 0.2]) 
        self.T_imu_lidar = create_homogeneous_matrix(R_imu_lidar, t_imu_lidar)
        
        # LiDAR to Camera (Camera is front facing, but optical frame is rotated)
        # LiDAR: x-fwd, y-left, z-up
        # Camera: z-fwd, x-right, y-down
        # Rotation needed:
        # Camera Z (fwd) -> LiDAR X (fwd)
        # Camera X (right) -> LiDAR -Y (right)
        # Camera Y (down) -> LiDAR -Z (down)
        
        # R_lidar_cam (transforms points from Cam to Lidar)
        # col 0 (Cam X in Lidar) = [0, -1, 0]
        # col 1 (Cam Y in Lidar) = [0, 0, -1]
        # col 2 (Cam Z in Lidar) = [1, 0, 0]
        R_lidar_cam = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])
        t_lidar_cam = np.array([0.1, 0.0, -0.1]) # 10cm forward, 10cm down relative to lidar
        self.T_lidar_camera = create_homogeneous_matrix(R_lidar_cam, t_lidar_cam)

    def get_transform(self, source, target):
        """
        Returns the 4x4 homogenous matrix T_target_source (points in source -> points in target).
        """
        if source == "lidar" and target == "camera":
            # T_cam_lidar = inv(T_lidar_cam)
            from .homogeneous_transform import inverse_homogeneous_matrix
            return inverse_homogeneous_matrix(self.T_lidar_camera)
        elif source == "camera" and target == "lidar":
            return self.T_lidar_camera
        elif source == "imu" and target == "lidar":
            return self.T_imu_lidar
        # Add more as needed or implement a graph search
        raise NotImplementedError(f"Transform from {source} to {target} not implemented.")

    def set_extrinsic(self, source, target, matrix):
        if source == "camera" and target == "lidar":
            self.T_lidar_camera = matrix
        # etc
