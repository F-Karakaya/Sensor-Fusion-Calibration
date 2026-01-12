import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from .sensor_models import MotionModel, MeasurementModel

class StateEKF:
    """
    Extended Kalman Filter for 3D state estimation.
    """
    def __init__(self, dt=0.1):
        self.dim_x = 6 # x, y, z, vx, vy, vz
        self.dim_z = 3 # x, y, z (LiDAR measurement)
        self.ekf = ExtendedKalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)
        self.dt = dt
        self.models = MotionModel()
        self.meas_models = MeasurementModel()
        
        # Initialize
        self.ekf.x = np.array([0., 0., 0., 0., 0., 0.])
        self.ekf.F = self.models.get_transition_matrix(dt)
        self.ekf.Q = self.models.get_process_noise(dt)
        self.ekf.P *= 10.0 # High initial uncertainty
        self.ekf.R = self.meas_models.get_lidar_noise()

    def predict(self):
        self.ekf.predict()

    def update(self, measurement):
        """
        Update with LiDAR measurement [x, y, z].
        For Linear measurement model, H is constant.
        """
        H = self.meas_models.get_lidar_H()
        self.ekf.update(measurement, HJacobian=lambda x: H, Hx=self.meas_models.lidar_measurement_function)

    def get_state(self):
        return self.ekf.x
