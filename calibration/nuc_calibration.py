import numpy as np
import cv2

def generate_nuc_table(sensor_width, sensor_height):
    """
    Generates a synthetic Non-Uniformity Correction (NUC) table.
    Simulates fixed pattern noise (FPN) and gain variations.
    """
    # Gain map: Multiplicative noise (around 1.0)
    gain_map = np.random.normal(1.0, 0.05, (sensor_height, sensor_width)).astype(np.float32)
    
    # Offset map: Additive noise (e.g. thermal dark current)
    offset_map = np.random.normal(0, 5.0, (sensor_height, sensor_width)).astype(np.float32)
    
    return gain_map, offset_map

def apply_nuc(image, gain_map, offset_map):
    """
    Applies NUC to a raw image.
    Corrected = (Raw - Offset) * Gain_Correction
    Wait, usually: Corrected = (Raw - Offset) / Gain OR (Raw - Offset) * Gain_Coeff
    Let's assume gain_map is the correction coefficient (1/response).
    """
    img_float = image.astype(np.float32)
    corrected = (img_float - offset_map) * gain_map
    return np.clip(corrected, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    h, w = 720, 1280
    gain, offset = generate_nuc_table(w, h)
    
    # Save simulated NUC tables
    np.save("calibration/nuc_gain.npy", gain)
    np.save("calibration/nuc_offset.npy", offset)
    print("NUC tables generated.")
