from math import cos, sin
import numpy as np
import json
import yaml
import os

with open('src/3d_reconstruction/gps_reconstruction/conf/config.yaml', 'r') as file:
        config = yaml.safe_load(file)


def get_timestamp(msg) -> float:
    """
    Get time stamp of a ROS message.

    Args:
        msg: The deserialized ROS message.

    Returns:
        float: Time stamp expressed in seconds.
    """
    return msg.header.stamp.sec + msg.header.stamp.nanosec / 10 ** 9

def transformation_matrix(roll: float=0, pitch: float=0, yaw: float=0, t_x: float=0, t_y: float=0, t_z: float=0) -> np.ndarray:
    """
    Compute transformation matrix.

    Args:
        roll (float): Roll in radians.
        pitch (float): Pitch in radians.
        yaw (float): Yaw in radians.
        t_x (float): Translation along x axis.
        t_y (float): Translation along y axis.
        t_z (float): Translation along z axis.

    Returns:
        np.ndarray: 4x4 numpy transformation matrix.
    """
    r_z = np.array([
            [cos(yaw), -sin(yaw), 0],
            [sin(yaw), cos(yaw), 0],
            [0, 0, 1]
        ], dtype=np.float64)

    r_y = np.array([
        [cos(pitch), 0, sin(pitch)],
        [0, 1, 0],
        [-sin(pitch), 0, cos(pitch)]
    ], dtype=np.float64)

    r_x = np.array([
        [1, 0, 0],
        [0, cos(roll), -sin(roll)],
        [0, sin(roll), cos(roll)]
    ], dtype=np.float64)

    t = np.zeros((4, 4))
    t[:3, :3] = r_z @ r_y @ r_x
    t[0, 3] = t_x
    t[1, 3] = t_y
    t[2, 3] = t_z
    t[3, 3] = 1.0

    return t

def sensor_transformation(sensor_name: str) -> np.ndarray:
    """
    Compute transformation matrix from a sensor frame to the world frame.

    Args:
        sensor_name (str): Name of sensor's link.

    Returns:
        np.ndarray: 4x4 numpy transformation matrix.
    """
    file = open(os.path.join(os.getcwd(), config['tf_static']['path']))
    sensors = json.load(file)

    for sensor in sensors:
        if sensor['child_frame_id'] == sensor_name:
            roll = sensor['transform']['rotation']['roll']
            pitch = sensor['transform']['rotation']['pitch']
            yaw = sensor['transform']['rotation']['yaw']

            t_x = sensor['transform']['translation']['x']
            t_y = sensor['transform']['translation']['y']
            t_z = sensor['transform']['translation']['z']

            return transformation_matrix(roll, pitch, yaw, t_x, t_y, t_z)
