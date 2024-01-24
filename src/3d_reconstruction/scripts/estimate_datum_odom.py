from rosbags.rosbag2 import Reader
from pyproj import Proj
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
import math
import yaml
from ..gps_reconstruction import get_timestamp


with open('src/3d_reconstruction/conf/config.yaml', 'r') as file:
        config = yaml.safe_load(file)


def get_odom_data(bag_path: str, time_of_interest: float) -> list[tuple[float, tuple[float, float]]]:
    '''
    Returns list of odometry measurements.

    Args:
        bag_path (str): Path to the Bag file from which data is to be read.
        time_of_interest (float): Specifies to collect only measurements from first time_of_interest seconds.
    
    Returns:
        list[tuple[float, tuple[float, float]]]: List of odometry measurements in format (timestamp, (x, y)).
    '''
    with Reader(bag_path) as reader:
        odom_data = []
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/t265/odom/sample':
                msg = deserialize_message(rawdata, Odometry)
                pose = msg.pose.pose
                x, y = pose.position.x, pose.position.y

                if len(odom_data) > 0:
                    x -= odom_data[0][1][0]
                    y -= odom_data[0][1][1]
                else:
                    first_timestamp = get_timestamp(msg)

                if get_timestamp(msg) - first_timestamp > time_of_interest:
                    break
                    
                odom_data.append((get_timestamp(msg), (x, y)))
        odom_data[0] = (odom_data[0][0], (0, 0))

    return odom_data

def get_gps_data(bag_path: str, time_of_interest: float) -> list[tuple[float, tuple[float, float, float, float]]]:
    '''
    Returns list of gps measurements.

    Args:
        bag_path (str): Path to the Bag file from which data is to be read.
        time_of_interest (float): Specifies to collect only measurements from first time_of_interest seconds.
    
    Returns:
        list[tuple[float, tuple[float, float, float, float]]]: List of gps measurements in format (timestamp, (x, y, latitude, longitude)).
    '''
    with Reader(bag_path) as reader:
        gps_data = []
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/fix':
                msg = deserialize_message(rawdata, NavSatFix)
                p = Proj(proj='utm', zone=32, ellps='WGS84', preserve_units=False)
                x, y = p(msg.longitude, msg.latitude)

                if len(gps_data) > 0:
                    x -= gps_data[0][1][0]
                    y -= gps_data[0][1][1]
                else:
                    first_timestamp = get_timestamp(msg)

                if get_timestamp(msg) - first_timestamp > time_of_interest:
                    break
                    
                gps_data.append((get_timestamp(msg), (x, y, msg.latitude, msg.longitude)))
        gps_data[0] = (gps_data[0][0], (0, 0, gps_data[0][1][2], gps_data[0][1][3]))

    return gps_data

def get_rotation_between_gps_odom_frame(bag_path: str) -> float:
    '''
    Returns estimated rotation angle between gps and odometry frames.

    Args:
        bag_path (str): Path to the Bag file from which data is to be read.
    '''
    odom_data = get_odom_data(bag_path, 30.0)
    gps_data = get_gps_data(bag_path, 30.0)
    corresponding_odom_data = []
    for timestamp, _ in gps_data:
        index = odom_data.index(min(odom_data, key=lambda x: abs(x[0] - timestamp)))
        corresponding_odom_data.append(odom_data[index]) 

    x_gps = np.array([val[1][0] for val in gps_data])
    y_gps = np.array([val[1][1] for val in gps_data])

    mse = []
    angles = list(np.linspace(-np.pi, np.pi, 300))
    for angle in angles:
        x_odom = np.array([val[1][0] * np.cos(angle) - val[1][1] * np.sin(angle) for val in corresponding_odom_data])
        y_odom = np.array([val[1][0] * np.sin(angle) + val[1][1] * np.cos(angle) for val in corresponding_odom_data])
        loss = np.mean((x_odom - x_gps) ** 2 + (y_odom - y_gps) ** 2)
        mse.append(loss)
    best_angle = angles[mse.index(min(mse))]

    mse = np.array(mse)
    angles = np.array(angles)
    plt.scatter(angles, mse)
    plt.show()
    
    x_odom_rotated = np.array([val[1][0] * np.cos(best_angle) - val[1][1] * np.sin(best_angle) for val in odom_data])
    y_odom_rotated = np.array([val[1][0] * np.sin(best_angle) + val[1][1] * np.cos(best_angle) for val in odom_data])
    
    x_odom = np.array([val[1][0] for val in odom_data])
    y_odom = np.array([val[1][1] for val in odom_data])

    plt.scatter(x_odom_rotated, y_odom_rotated, label="Odometry after rotation")
    plt.scatter(x_gps, y_gps, label="GPS")
    plt.scatter(x_odom, y_odom, label="Odometry")
    plt.legend()
    plt.show()

    return best_angle

def estimate_datum(bag_path: str) -> list:
    '''
    Function used to estimate initial orientation of the robot and returning datum [latitude, longitude, heading]
    required by navsat_transform node from robot_localization package. It firstly rotates odometry coordinates
    to be in gps frame and then estimates robot's orientation using rotated, continous odometry.

    Args:
        bag_path (str): Path to the Bag file from which data is to be read.

    Returns:
        list: Datum [latitude, longitude, heading] where heading is expressed in radians.
    '''

    def distance(point1: tuple[float, float], point2: tuple[float, float]) -> float:
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    odom_data = get_odom_data(bag_path, 30.0)
    for i in range(len(odom_data)):
        if distance(odom_data[i][1], odom_data[0][1]) > 0.2:
            odom_data = odom_data[:i]
            break

    angle = get_rotation_between_gps_odom_frame(bag_path)
    x = np.expand_dims(np.array([val[1][0] * np.cos(angle) - val[1][1] * np.sin(angle) for val in odom_data]), axis=1)
    y = np.expand_dims(np.array([val[1][0] * np.sin(angle) + val[1][1] * np.cos(angle) for val in odom_data]), axis=1)
            
    reg = LinearRegression().fit(x, y)
    heading = math.atan2(reg.coef_[0] * (x[-1, 0] - x[0, 0]), x[-1, 0] - x[0, 0])
    if heading < 0:
        heading += 2 * np.pi

    print(heading)

    plt.scatter(x, y)
    plt.plot(x, reg.predict(x))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    latitude_first, longitude_first = get_gps_data(bag_path, 5.0)[0][1][-2:]

    return [latitude_first, longitude_first, heading]
        
          
if __name__ == "__main__":
    datum = estimate_datum('/media/jakub/Extreme SSD/2022-07-28-12-27-52_filare_3')
    print(datum)
