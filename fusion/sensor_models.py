import numpy as np

class MotionModel:
    """
    Constant Velocity (CV) Motion Model for 3D tracking.
    State: [x, y, z, vx, vy, vz]
    """
    def get_transition_matrix(self, dt):
        """
        Returns F matrix for state x_k = F * x_{k-1}
        """
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def get_process_noise(self, dt, sigma_a=0.1):
        """
        Discrete process noise matrix Q.
        Assumes random acceleration.
        """
        # Approximated Q for CV model
        G = np.array([
            [0.5*dt**2, 0, 0],
            [0, 0.5*dt**2, 0],
            [0, 0, 0.5*dt**2],
            [dt, 0, 0],
            [0, dt, 0],
            [0, 0, dt]
        ])
        Q = G @ G.T * sigma_a**2
        return Q

class MeasurementModel:
    """
    Measurement models for Lidar and Camera.
    """
    def lidar_measurement_function(self, state):
        """
        LiDAR measures [x, y, z] directly.
        H matrix would be 3x6: [I | 0]
        """
        return state[:3]

    def get_lidar_H(self):
        H = np.zeros((3, 6))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        return H

    def get_lidar_noise(self, std_x=0.1, std_y=0.1, std_z=0.1):
        return np.diag([std_x**2, std_y**2, std_z**2])
