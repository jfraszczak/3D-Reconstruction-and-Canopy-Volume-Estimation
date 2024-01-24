from rosbags.rosbag2 import Reader
from sensor_msgs_py.point_cloud2 import read_points, create_cloud_xyz32
import sensor_msgs.msg as sensor_msgs
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2, Image, NavSatFix
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
from .reconstructor import Reconstructor
from .point_cloud_preprocessor import preprocess_point_cloud
from .utils_reconstruction import sensor_transformation, get_timestamp
from .volume_estimator import VolumeEstimator
import yaml
from pyproj import Proj
from matplotlib import pyplot as plt


with Reader('/home/jakub/rosbag2_2023_11_23-23_37_30') as reader:
    gps_data = []
    for connection, _, rawdata in reader.messages():
        if connection.topic == '/odometry/gps':
            msg = deserialize_message(rawdata, Odometry)
            pose = msg.pose.pose
            x, y, z = pose.position.x, pose.position.y, pose.position.z

            gps_data.append((x, y, z))


xs = np.array([val[0] for val in gps_data])
ys = np.array([val[1] for val in gps_data])

plt.scatter(xs, ys)
plt.xlabel('x')
plt.ylabel('y')
plt.title('ODOMETRY/GPS')
plt.show()

with Reader('/home/jakub/rosbag2_2023_11_23-23_37_30') as reader:
    gps_data = []
    for connection, _, rawdata in reader.messages():
        if connection.topic == '/odom':
            msg = deserialize_message(rawdata, Odometry)
            pose = msg.pose.pose
            x, y, z = pose.position.x, pose.position.y, pose.position.z

            gps_data.append((x, y, z))


xs = np.array([val[0] for val in gps_data])
ys = np.array([val[1] for val in gps_data])

plt.scatter(xs, ys)
plt.xlabel('x')
plt.ylabel('y')
plt.title('RGBD ODOMETRY')
plt.show()