import numpy as np
from scipy.spatial.transform import Rotation as R

def create_homogeneous_matrix(rotation_matrix, translation_vector):
    """
    Creates a 4x4 homogeneous transformation matrix.
    
    Args:
        rotation_matrix (np.ndarray): 3x3 rotation matrix.
        translation_vector (np.ndarray): 3x1 or 1x3 translation vector.
        
    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation_vector.flatten()
    return T

def inverse_homogeneous_matrix(T):
    """
    Computes the inverse of a 4x4 homogeneous transformation matrix.
    
    Args:
        T (np.ndarray): 4x4 transformation matrix.
        
    Returns:
        np.ndarray: Inverse of T.
    """
    R_mat = T[:3, :3]
    t_vec = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_mat.T
    T_inv[:3, 3] = -R_mat.T @ t_vec
    return T_inv

def euler_to_matrix(roll, pitch, yaw, degrees=False):
    """
    Converts Euler angles to a rotation matrix (XYZ order).
    
    Args:
        roll (float): Rotation around X-axis.
        pitch (float): Rotation around Y-axis.
        yaw (float): Rotation around Z-axis.
        degrees (bool): If True, input angles are in degrees.
        
    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    return r.as_matrix()

def matrix_to_euler(matrix, degrees=False):
    """
    Converts a rotation matrix to Euler angles (XYZ order).
    
    Args:
        matrix (np.ndarray): 3x3 rotation matrix.
        degrees (bool): If True, output angles are in degrees.
        
    Returns:
        np.ndarray: [roll, pitch, yaw]
    """
    r = R.from_matrix(matrix)
    return r.as_euler('xyz', degrees=degrees)
