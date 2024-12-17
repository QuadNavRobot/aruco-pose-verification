import numpy as np
import cv2

calibration_matrix_path = 'calibration_matrix.npy'
distortion_coefficients_path = 'distortion_coefficients.npy'
url_camera = 'http://172.16.0.109:8080/video'
aruco_dict_type = cv2.aruco.DICT_4X4_50

def pose_estimation_two_markers(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
        gray, cv2.aruco_dict, parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients
    )

    tvecs_dict = {}
    rvecs_dict = {}

    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Get the pose of each marker
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], 0.036, matrix_coefficients, distortion_coefficients
            )

            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.018)

            # Store translation and rotation vectors
            rvecs_dict[ids[i][0]] = rvec[0][0]
            tvecs_dict[ids[i][0]] = tvec[0][0]

            # Show marker ID on image
            position = tuple(corners[i][0][0].astype(int))
            cv2.putText(frame, f"ID: {ids[i][0]}", position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Calculate the distance between two specific markers
        marker_1_id = 0
        marker_2_id = 1

        if marker_1_id in tvecs_dict and marker_2_id in tvecs_dict:
            tvec1 = tvecs_dict[marker_1_id]
            tvec2 = tvecs_dict[marker_2_id]

            dist_x = tvec2[0] - tvec1[0]
            dist_y = tvec2[1] - tvec1[1]
            dist_z = tvec2[2] - tvec1[2]

            # Convert rvec to rotation matrix
            R1, _ = cv2.Rodrigues(rvecs_dict[marker_1_id])
            R2, _ = cv2.Rodrigues(rvecs_dict[marker_2_id])

            # Calculate the rotation angles between the two markers
            rotation_matrix_diff = np.dot(R1.T, R2)
            sy = np.sqrt(rotation_matrix_diff[0, 0]**2 + rotation_matrix_diff[1, 0]**2)

            if sy < 1e-6:
                yaw = np.arctan2(-rotation_matrix_diff[1, 2], rotation_matrix_diff[1, 1])
                pitch = np.arctan2(-rotation_matrix_diff[2, 0], sy)
                roll = 0
            else:
                yaw = np.arctan2(rotation_matrix_diff[2, 1], rotation_matrix_diff[2, 2])
                pitch = np.arctan2(-rotation_matrix_diff[2, 0], sy)
                roll = np.arctan2(rotation_matrix_diff[1, 0], rotation_matrix_diff[0, 0])

            # Convert angles from radians to degrees.
            yaw_deg = np.degrees(yaw)
            pitch_deg = np.degrees(pitch)
            roll_deg = np.degrees(roll)

            # Show data on the image
            cv2.putText(frame, f"Dist_x: {dist_x:.3f}m", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Dist_y: {dist_y:.3f}m", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Dist_z: {dist_z:.3f}m", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Yaw: {yaw_deg:.2f}", (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Pitch: {pitch_deg:.2f}", (10, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Roll: {roll_deg:.2f}", (10, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame

if __name__ == '__main__':
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    video = cv2.VideoCapture(url_camera)

    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        output = pose_estimation_two_markers(frame, aruco_dict_type, k, d)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
