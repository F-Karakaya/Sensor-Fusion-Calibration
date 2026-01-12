import numpy as np
import cv2
import glob
import yaml
import os

class CameraCalibrator:
    def __init__(self, checkerboard_size=(7, 7), square_size_mm=50.0):
        self.points_3d = [] # 3d points in real world space
        self.points_2d = [] # 2d points in image plane
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size_mm
        
        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...
        self.objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1,2) * self.square_size

    def detect_corners(self, image_dir, show=False):
        images = glob.glob(os.path.join(image_dir, '*.png'))
        if not images:
            print("No images found for calibration.")
            return

        print(f"Processing {len(images)} images for calibration...")
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)

            if ret:
                self.points_3d.append(self.objp)
                
                # Refine corners
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), 
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                self.points_2d.append(corners2)
                
                if show:
                    cv2.drawChessboardCorners(img, self.checkerboard_size, corners2, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(100)
        if show:
            cv2.destroyAllWindows()
            
    def calibrate(self, image_size):
        if not self.points_3d:
            print("No calibration data collected.")
            return None, None
            
        print("Running calibration optimization...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.points_3d, self.points_2d, image_size, None, None)
        
        print(f"RMS Re-projection Error: {ret:.4f}")
        return mtx, dist

    def save_calibration(self, mtx, dist, output_file):
        data = {
            'fx': float(mtx[0,0]),
            'fy': float(mtx[1,1]),
            'cx': float(mtx[0,2]),
            'cy': float(mtx[1,2]),
            'k1': float(dist[0,0]),
            'k2': float(dist[0,1]),
            'p1': float(dist[0,2]),
            'p2': float(dist[0,3]),
            'k3': float(dist[0,4])
        }
        with open(output_file, 'w') as f:
            yaml.dump(data, f)
        print(f"Calibration saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    calib = CameraCalibrator(checkerboard_size=(7,7), square_size_mm=50)
    calib.detect_corners("data/camera/images", show=False)
    
    # Assuming all images are same size
    test_img = cv2.imread(glob.glob("data/camera/images/*.png")[0])
    h, w = test_img.shape[:2]
    
    mtx, dist = calib.calibrate((w, h))
    if mtx is not None:
        calib.save_calibration(mtx, dist, "data/camera/intrinsics_calibrated.yaml")
