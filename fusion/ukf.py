import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from .sensor_models import MotionModel, MeasurementModel

class StateUKF:
    """
    Unscented Kalman Filter for 3D state estimation.
    """
    def __init__(self, dt=0.1):
        self.dim_x = 6
        self.dim_z = 3
        self.dt = dt
        self.models = MotionModel()
        self.meas_models = MeasurementModel()
        
        # Sigma points
        points = MerweScaledSigmaPoints(self.dim_x, alpha=0.1, beta=2., kappa=-1)
        
        def fx(x, dt):
            F = self.models.get_transition_matrix(dt)
            return F @ x

        def hx(x):
            return self.meas_models.lidar_measurement_function(x)

        self.ukf = UnscentedKalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z, 
                                         dt=dt, fx=fx, hx=hx, points=points)
        
        self.ukf.x = np.array([0., 0., 0., 0., 0., 0.])
        self.ukf.P *= 10.0
        self.ukf.Q = self.models.get_process_noise(dt)
        self.ukf.R = self.meas_models.get_lidar_noise()

    def predict(self):
        self.ukf.predict()

    def update(self, measurement):
        self.ukf.update(measurement)

    def get_state(self):
        return self.ukf.x
