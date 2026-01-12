# Kalman Filter Comparison: EKF vs UKF

## Extended Kalman Filter (EKF)
The EKF adapts the standard Linear Kalman Filter to non-linear systems by transforming the state covariance through linear approximations of the non-linear process and measurement models.

### Mechanism
- **Linearization**: Uses First-order Taylor Series expansion (Jacobians) of $f(x)$ and $h(x)$ around the current mean.
- **Prediction**:
  $x_{k|k-1} = f(x_{k-1})$
  $P_{k|k-1} = F_k P_{k-1} F_k^T + Q_k$
- **Update**:
  $K_k = P_{k|k-1} H_k^T (H_k P_{k|k-1} H_k^T + R_k)^{-1}$
  $x_k = x_{k|k-1} + K_k (z_k - h(x_{k|k-1}))$

### Pros & Cons
- **Pros**: Computationally efficient (if Jacobians are simple), standard for well-behaved non-linearities.
- **Cons**: Can diverge if non-linearity is high; tedious to derive Jacobians; first-order approximation only.

## Unscented Kalman Filter (UKF)
The UKF uses a deterministic sampling approach (Sigma Points) to capture the mean and covariance of the state distribution after passing through non-linear functions, avoiding linearization (Jacobians) entirely.

### Mechanism
- **Sigma Points**: Select $2L+1$ points around the mean that capture the covariance structure.
- **Transformation**: Propagate each sigma point through the non-linear function $y = g(x)$.
- **Reconstruction**: Compute the weighted mean and covariance of the transformed sigma points.

### Pros & Cons
- **Pros**: Captures higher-order statistics (up to 3rd order for Gaussian); No Jacobians needed; Robust to strong non-linearities.
- **Cons**: Slightly higher computational cost ($2L+1$ function evaluations); Tuning parameters $(\alpha, \beta, \kappa)$ required.

## Conclusion
For systems with complex dynamics or sensor models where Jacobians are difficult to derive or highly unstable, **UKF** is preferred. For standard tracking tasks where efficiency is key and models are moderately non-linear, **EKF** remains a solid choice. In this repository, we compare both for 3D state estimation.
