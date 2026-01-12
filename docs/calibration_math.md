# Calibration Mathematics

This document details the mathematical principles used in the calibration modules.

## 1. Perspective-n-Point (PnP)
PnP is the problem of estimating the pose of a calibrated camera given a set of $n$ 3D points in the world and their corresponding 2D projections in the image.

We seek the rotation matrix satisfying $R \in SO(3)$ and translation vector $t \in \mathbb{R}^3$ that minimize the reprojection error:

$$
\min_{R, t} \sum_{i=1}^{n} \| p_i - \pi(K(R P_i + t)) \|^2
$$

Where:
- $P_i$ is the $i$-th 3D point in the world frame (or LiDAR frame).
- $p_i$ is the observed 2D point in the image plane.
- $K$ is the camera intrinsic matrix.
- $\pi$ is the projection function (perspective division).

In this repository, we use `cv2.solvePnP` which typically uses the Levenberg-Marquardt algorithm to optimize this non-linear least squares problem.

## 2. Iterative Closest Point (ICP)
ICP is used to align two point clouds (e.g., LiDAR scan vs. Depth Camera point cloud).
The algorithm iteratively minimizes the distance between corresponding points in two clouds, source $P$ and target $Q$:

$$
E(R, t) = \sum_{i=1}^{N} \| (R p_i + t) - q_i \|^2
$$

1. **Match**: For each point in source, find the closest point in target.
2. **Minimize**: Compute $R, t$ that minimize mean squared error.
3. **Transform**: Apply $R, t$ to source.
4. **Repeat** until convergence.

## 3. Reprojection Error
Reprojection error is the geometric error corresponding to the image distance between a projected 3D point and a measured 2D point. It is used to quantify how closely an estimate of a 3D point recreates the point's true projection.

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} d(p_i, \hat{p}_i)^2}
$$

A low RMS reprojection error (typically < 1.0 pixel) indicates a high-quality calibration.
