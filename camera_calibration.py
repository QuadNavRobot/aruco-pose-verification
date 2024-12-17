import numpy as np
import cv2
import os

square_size = 0.023 # (2.3 cm) size of the squares on the chessboard
dirpath = "images_calibration/" # path to the directory containing the calibration images
width = 9 # number of squares on the horizontal axis of the chessboard
height = 6 # number of squares on the vertical axis of the chessboard
visualize = False # set True to visualize the detected corners

def calibrate(dirpath, square_size, width, height, visualize=False):

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    images = os.listdir(dirpath)

    for fname in images:
        img = cv2.imread(os.path.join(dirpath, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

        if visualize:
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('img', 1920, 1080)
            cv2.imshow('img', img)
            cv2.waitKey(0)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]


if __name__ == '__main__':

    ret, mtx, dist, rvecs, tvecs = calibrate(dirpath, square_size, visualize=visualize, width=width, height=height)

    print(mtx)
    print(dist)

    np.save("calibration_matrix", mtx)
    np.save("distortion_coefficients", dist)