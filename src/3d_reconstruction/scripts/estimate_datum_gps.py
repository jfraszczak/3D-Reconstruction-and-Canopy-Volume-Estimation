from rosbags.rosbag2 import Reader
from pyproj import Proj
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import NavSatFix
import math
import yaml

with open('src/3d_reconstruction/conf/config.yaml', 'r') as file:
        config = yaml.safe_load(file)


def estimate_datum(bag_path: str) -> list:
    '''
    Function used to estimate initial orientation of the robot and returning datum [latitude, longitude, heading]
    required by navsat_transform node from robot_localization package.

    Args:
        bag_path (str): Path to the Bag file from which data is to be read.

    Returns:
        list: Datum [latitude, longitude, heading] where heading is expressed in radians.
    '''

    def distance(point1: tuple[float, float], point2: tuple[float, float]) -> float:
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    with Reader(bag_path) as reader:
        latitude_first = None
        longitude_first = None
        x = []
        y = []
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == config['bag']['topics']['gps']:
                msg = deserialize_message(rawdata, NavSatFix)
                p = Proj(proj='utm', zone=32, ellps='WGS84', preserve_units=False)
                x_utm, y_utm = p(msg.longitude, msg.latitude)

                if latitude_first is None and longitude_first is None:
                    latitude_first = msg.latitude
                    longitude_first = msg.longitude
                else:
                    if distance((x[0], y[0]), (x_utm, y_utm)) > 0.2:
                        break
                    
                x.append(x_utm)
                y.append(y_utm)


        x = np.expand_dims(np.array(x), axis=1)
        y = np.array(y)

        reg = LinearRegression().fit(x, y)
        heading = math.atan2(reg.coef_[0] * (x[-1, 0] - x[0, 0]), x[-1, 0] - x[0, 0])
        if heading < 0:
            heading += 2 * np.pi

        plt.scatter(x, y)
        plt.plot(x, reg.predict(x))
        plt.xlabel('x_utm')
        plt.ylabel('y_utm')
        plt.show()

        return [latitude_first, longitude_first, heading]
                
    
if __name__ == "__main__":
    datum = estimate_datum('/media/jakub/Extreme SSD/2022-07-28-12-27-52_filare_3')
    print("DATUM:", datum)