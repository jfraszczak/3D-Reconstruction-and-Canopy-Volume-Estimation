from rosbags.rosbag2 import Reader
from sensor_msgs_py.point_cloud2 import read_points
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import yaml

from .reconstructor import Reconstructor
from .point_cloud_preprocessor import remove_distant_points, remove_outliers, remove_points_below_ground
from .utils_reconstruction import sensor_transformation, get_timestamp

with open('src/3d_reconstruction/gps_reconstruction/conf/config.yaml', 'r') as file:
        config = yaml.safe_load(file)


class LidarReconstructor(Reconstructor):

    def __init__(self, output_dir: str, test_mode: bool=False):
        """
        Initializes all the necessary attributes for the reconstructor object and reads data from the Bag file.

        Args:
            test_mode (bool): Indication whether to visualize intermediate computations or not.
        """
        super().__init__(output_dir=output_dir, test_mode=test_mode)
        self.sensor = config['tf_static']['lidar_link']


    def _collect_point_clouds(self, time_start: float = 0.0, time_end: float = -1.0, every_k: int = 1) -> None:
        """
        Computes consecutive point clouds based on the depth maps, intrinsic matrix of the RGBD camera and color images.

        Args:
            time_start (float): Value specifying point cloud with the earliest time stamp which should be taken into account.
            time_start (float): Value specifying point cloud with the latest time stamp which should be taken into account.
            every_k (int): Value specifying that only every k-th point cloud will be considered.
        """
        self.clouds = []
        i = -1
        with Reader(config['bag']['path']) as reader:
            for connection, _, rawdata in reader.messages():
                if connection.topic == config['bag']['topics']['lidar_points']:
                    msg = deserialize_message(rawdata, PointCloud2)
                    timestamp = get_timestamp(msg)

                    # Disregard measurements outside of given time range
                    if time_end == -1.0:
                        if time_start > timestamp:
                            continue
                    else:
                        if not (time_start <= timestamp <= time_end):
                            continue
                    
                    # Take into account only every k-th measurement
                    i += 1
                    if i % every_k != 0:
                        continue
                    
                    # Create point cloud
                    cloud_data = list(read_points(msg, skip_nans=True, field_names=['x', 'y', 'z']))
                    cloud = o3d.geometry.PointCloud()
                    xyz = [(x, y, z) for x, y, z in cloud_data]
                    cloud.points = o3d.utility.Vector3dVector(np.array(xyz))

                    # Transform point cloud to the base link frame before performing preprocessing
                    t = sensor_transformation(self.sensor)
                    cloud.transform(t)

                    # Preprocess point cloud
                    cloud = cloud.voxel_down_sample(voxel_size=0.05)
                    cloud = remove_distant_points(cloud, threshold=3.0)
                    cloud = remove_points_below_ground(cloud, threshold=0.1, test_mode=self.test_mode)
                    cloud = remove_outliers(cloud)
                    
                    # Reverse transformation to store point clouds in sensor frame
                    cloud.transform(np.linalg.inv(t))

                    self.clouds.append((timestamp, cloud))
