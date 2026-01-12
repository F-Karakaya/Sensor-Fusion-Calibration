import numpy as np

class DriftCompensator:
    """
    Compensates for thermal or temporal drift in sensor extrinsics.
    """
    def __init__(self, temp_coefficient=0.001):
        self.temp_coefficient = temp_coefficient # mm per degree Celsius
        self.base_temp = 25.0
        
    def compensate_extrinsics(self, T_nominal, current_temp):
        """
        Adjusts transformation matrix based on temperature delta.
        """
        dt = current_temp - self.base_temp
        expansion = dt * self.temp_coefficient
        
        T_new = T_nominal.copy()
        # Assume linear expansion in translation (simple model)
        # e.g. the baseline increases
        T_new[0, 3] *= (1 + expansion) 
        T_new[1, 3] *= (1 + expansion)
        T_new[2, 3] *= (1 + expansion)
        
        return T_new

if __name__ == "__main__":
    comp = DriftCompensator()
    T = np.eye(4)
    T[0, 3] = 100.0
    
    T_hot = comp.compensate_extrinsics(T, 50.0)
    print(f"Original Tx: {T[0,3]}")
    print(f"Compensated Tx (50C): {T_hot[0,3]}")
