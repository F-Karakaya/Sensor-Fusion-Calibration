import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.getcwd())
from fusion.ekf import StateEKF
from fusion.ukf import StateUKF

def simulate_and_plot():
    # Simulate ground truth trajectory: Circle
    t = np.linspace(0, 20, 200)
    dt = t[1] - t[0]
    
    # Ground Truth
    gt_x = 10 * np.cos(t)
    gt_y = 10 * np.sin(t)
    gt_z = np.linspace(0, 5, 200)
    
    # Measurements (noisy)
    meas_x = gt_x + np.random.normal(0, 0.5, len(t))
    meas_y = gt_y + np.random.normal(0, 0.5, len(t))
    meas_z = gt_z + np.random.normal(0, 0.5, len(t))
    
    ekf = StateEKF(dt)
    ukf = StateUKF(dt)
    
    ekf_states = []
    ukf_states = []
    
    for i in range(len(t)):
        z = np.array([meas_x[i], meas_y[i], meas_z[i]])
        
        # EKF
        ekf.predict()
        ekf.update(z)
        ekf_states.append(ekf.get_state())
        
        # UKF
        ukf.predict()
        ukf.update(z)
        ukf_states.append(ukf.get_state())
        
    ekf_states = np.array(ekf_states)
    ukf_states = np.array(ukf_states)
    
    # Plot XY Plane
    plt.figure(figsize=(10, 6))
    plt.plot(gt_x, gt_y, 'k--', label='Ground Truth')
    plt.plot(meas_x, meas_y, 'r.', alpha=0.3, label='Measurements')
    plt.plot(ekf_states[:, 0], ekf_states[:, 1], 'b-', label='EKF')
    plt.plot(ukf_states[:, 0], ukf_states[:, 1], 'g-', label='UKF')
    plt.title("Sensor Fusion: EKF vs UKF Tracking (XY Plane)")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/filter_results/ekf_vs_gt.png")
    
    # Plot Error
    ekf_error = np.sqrt((gt_x - ekf_states[:,0])**2 + (gt_y - ekf_states[:,1])**2)
    ukf_error = np.sqrt((gt_x - ukf_states[:,0])**2 + (gt_y - ukf_states[:,1])**2)
    
    plt.figure(figsize=(10, 4))
    plt.plot(t, ekf_error, 'b-', label='EKF Error')
    plt.plot(t, ukf_error, 'g-', label='UKF Error')
    plt.title("Position Error Over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Error [m]")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/filter_results/ukf_vs_gt.png") # Saving error plot too.
    
    print("Plots saved to outputs/filter_results/")

if __name__ == "__main__":
    simulate_and_plot()
