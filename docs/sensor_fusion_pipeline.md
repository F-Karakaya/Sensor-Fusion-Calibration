# Sensor Fusion Pipeline Architecture

The system is designed as a modular pipeline for processing heterogeneous sensor data.

## 1. Data Ingestion
- **Camera**: Raw images (RGB) -> `data/camera/`
- **LiDAR**: Point clouds (PCD/BIN) -> `data/lidar/`
- **IMU**: CSV Logs (Accel/Gyro) -> `data/imu/`

## 2. Transformation Layer
- **Coordinate Frames**: Manages static transforms ($T_{imu}^{lidar}$, $T_{lidar}^{cam}$).
- **Projection**: Maps 3D LiDAR points to 2D image space using $K$ (intrinsics) and $T_{ext}$ (extrinsics).

## 3. Calibration Module
- **Intrinsic**: Computes $K, D$ using chessboard images.
- **Extrinsic**: Computes $T_{lidar}^{cam}$ using PnP/ICP.
- **Compensation**: Adjusts matrices for environmental drift.

## 4. Fusion Engine
- **Early Fusion (Data Level)**:
    - **Projection Fusion**: Overlaying LiDAR depth on Camera images.
    - **Colorization**: Attaching RGB attributes to LiDAR points.
- **Late Fusion (State Level)**:
    - **Filtering**: Fusing measurements from IMU/LiDAR/Camera (if visual odometry is added) into a single state vector $[x, y, z, v_x, v_y, v_z]$.
    - **Algorithms**: EKF and UKF implementations in `fusion/`.

## 5. Visualization
- **2D**: OpenCV windows for image overlays.
- **3D**: Open3D visualizer for colored point clouds.
- **Analysis**: Matplotlib plots for state error and covariance convergence.
