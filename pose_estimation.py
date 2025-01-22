import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from tf_transformations import quaternion_from_euler
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

video_path = 'videos/video.mp4'
ip_camera = 'http://172.16.0.73:8080/video'
calibration_matrix_path = 'calibration_matrix.npy'
distortion_coefficients_path = 'distortion_coefficients.npy'

aruco_dict_type = cv2.aruco.DICT_4X4_50

first_rvec = None
first_tvec = None
first_angles = None

class ArucoPublisher(Node):
    def __init__(self):
        super().__init__('aruco_publisher')
        self.odom_publisher = self.create_publisher(Odometry, 'aruco', 10)
        self.image_publisher = self.create_publisher(Image, 'aruco/image', 10)
        self.bridge = CvBridge()  # For converting between OpenCV images and ROS messages

    def publish_pose(self, x, y, angl_deg):
        angl_rad = np.radians(angl_deg)
        quaternion = quaternion_from_euler(0.0, 0.0, angl_rad)

        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.position.z = 0.0

        odom_msg.pose.pose.orientation = Quaternion(
            x=quaternion[0],
            y=quaternion[1],
            z=quaternion[2],
            w=quaternion[3]
        )

        self.odom_publisher.publish(odom_msg)
        self.get_logger().info(f"Published Odom: x={x:.3f}, y={y:.3f}, angl={angl_deg:.2f}°")

    def publish_image(self, frame):
        # Convert OpenCV image to ROS Image message
        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = "camera_frame"  # Adjust according to your frame

        # Publish the image
        self.image_publisher.publish(image_msg)
        self.get_logger().info("Published image")

def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    global first_rvec, first_tvec, first_angles

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
        gray, cv2.aruco_dict, parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients
    )

    # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], 0.115, matrix_coefficients, distortion_coefficients
            )

            # Save the first detected rvec and tvec
            if first_rvec is None and first_tvec is None:
                first_rvec = rvec
                first_tvec = tvec
                first_angles = calculate_euler_angles(first_rvec)
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, first_rvec, first_tvec, 0.0575)

            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.0575)

            # Call the function and print the results
            yaw_deg, pitch_deg, roll_deg = calculate_euler_angles(rvec)

            # Calculate the displacement from the initial position
            if first_rvec is not None and first_tvec is not None:
                dist_x = tvec[0, 0, 0] - first_tvec[0, 0, 0]
                dist_y = tvec[0, 0, 1] - first_tvec[0, 0, 1]
                dist_z = tvec[0, 0, 2] - first_tvec[0, 0, 2]
                a_yaw = yaw_deg - first_angles[0]
                a_pitch = pitch_deg - first_angles[1]
                a_roll = roll_deg - first_angles[2]
                
                # Wrap angles to -180, 180
                a_yaw = angle_wrap(a_yaw)
                a_pitch = angle_wrap(a_pitch)
                a_roll = angle_wrap(a_roll)

            aruco_publisher.publish_pose(dist_x, dist_y, a_roll)
            
            cv2.putText(frame, f"x: {dist_x:.3f} m", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"y: {dist_y:.3f} m", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # cv2.putText(frame, f"Dist_z: {dist_z:.3f}m", (10, 160),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # cv2.putText(frame, f"Yaw: {a_yaw:.2f}", (10, 190),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # cv2.putText(frame, f"Pitch: {a_pitch:.2f}", (10, 220),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Yaw: {a_roll:.2f} grados", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    aruco_publisher.publish_image(frame)        
    return frame

def calculate_euler_angles(rvec):
    # Convert rvec to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Calculate yaw, pitch, and roll from rotation matrix
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    # Check for singularity (gimbal lock)
    singular = sy < 1e-6
    if not singular:
        yaw = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        yaw = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = 0
    
    # Convert angles from radians to degrees
    yaw_deg = np.degrees(yaw)
    pitch_deg = np.degrees(pitch)
    roll_deg = -np.degrees(roll)
    
    return yaw_deg, pitch_deg, roll_deg

def angle_wrap(degrees):
    if degrees > 180:
        return degrees - 360
    elif degrees < -180:
        return degrees + 360
    return degrees

frame_skip = 3  # Procesa 1 de cada 3 fotogramas
frame_count = 0

if __name__ == '__main__':
    rclpy.init()
    aruco_publisher = ArucoPublisher()    
    
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)
    
    video = cv2.VideoCapture(ip_camera)  # For real-time video
    #video = cv2.VideoCapture(video_path)  # For video file

    try:
        while rclpy.ok():
            ret, frame = video.read()
            if not ret:
                break

            frame_count += 1   
            if frame_count % frame_skip != 0:
                continue
            
            output = pose_esitmation(frame, aruco_dict_type, k, d)

            cv2.imshow('Estimated Pose', output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()
    finally:
        rclpy.shutdown()