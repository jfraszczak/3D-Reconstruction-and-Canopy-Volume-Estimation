import open3d as o3d
import os
import numpy as np
from scipy.spatial.transform import Rotation
import copy
from rosbags.rosbag2 import Reader
from sensor_msgs_py.point_cloud2 import read_points
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
import laser_geometry.laser_geometry as lg
from scipy.spatial.transform import Slerp
from ..gps_reconstruction import get_timestamp


class CloudInterpolator:
    """
    Class used to perform interpolation of transformations between poses computed by RTABMap.
    Applied to reconstructions using scan topic as they are too sparse. As a result denser 
    cloud can be obtained.

    Attributes:
        dir (str): Path to the directory containing scan reconstruction which is to be made denser.
        scan_poses (list[list[float]]): List containing consecutive poses estimated by RTABMap.
        clouds (list[o3d.geometry.PointCloud]): List of all scan point clouds from Bag file.
        reconstructed_cloud (o3d.geometry.PointCloud): Point cloud reconstructed by RTABMap.

    Methods:
        interpolate(): Creates more dense reconstruction based on scan point clouds from Bag file and linear interpolations between poses estimated by RTABMap.
    """
        
    def __init__(self, dir: str) -> None:
        """
        Initializes required attributes.

        Args:
            dir (str): Path to the directory containing scan reconstruction which is to be made denser.
            scan_poses (list[list[float]]): List containing consecutive poses estimated by RTABMap.
            clouds (list[tuple[float, o3d.geometry.PointCloud]]): List of all scan point clouds from Bag file with corresponding timestamps stored in tuple (timestamp, cloud).
            reconstructed_cloud (o3d.geometry.PointCloud): Point cloud reconstructed by RTABMap.
        """
        self.dir = dir
        self.scan_poses = []
        self.clouds = []
        self.reconstructed_cloud = None

    def _read_reconstructed_point_cloud(self) -> None:
        """
        Reads point cloud reconstructed by RTABMap.
        """
        path = os.path.join(self.dir, 'rtabmap_cloud.ply')
        self.reconstructed_cloud = o3d.io.read_point_cloud(path)
        o3d.visualization.draw_geometries([self.reconstructed_cloud])

    def _read_poses(self) -> None:
        """
        Reads consecutive scan poses estimated by RTABMap.
        """
        def read_poses_file(file_path: str) -> []:
            file = open(file_path, 'r')
            poses = []
            header = True
            
            for line in file:
                if header:
                    header = False
                    continue

                values = line[:-1].split(' ')
                timestamp, x, y, z, qx, qy, qz, qw, id = [float(val) for val in values]
                id = int(id)
                poses.append([timestamp, x, y, z, qx, qy, qz, qw, id])
            
            return poses

        self.scan_poses = read_poses_file(os.path.join(self.dir, "rtabmap_scan_poses.txt"))

    def _collect_point_clouds(self, bag_path: str) -> None:
        """
        Reads point clouds from the specified Bag file.

        Args:
            bag_path (str): Path to the Bag file containing all the collected point clouds. 
        """
        lp = lg.LaserProjection()
        self.clouds = []
        i = -1
        with Reader(bag_path) as reader:
            for connection, _, rawdata in reader.messages():
                if connection.topic == '/scan':
                    msg = deserialize_message(rawdata, LaserScan)
                    timestamp = get_timestamp(msg)

                    msg = lp.projectLaser(msg)
                    
                    # Create point cloud
                    cloud_data = list(read_points(msg, skip_nans=True, field_names=['x', 'y', 'z']))
                    cloud = o3d.geometry.PointCloud()
                    xyz = [(x, y, z) for x, y, z in cloud_data]
                    cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
                    self.clouds.append((timestamp, cloud))


    def _get_slerp(self, idx: int) -> Slerp:
        """
        Returns Slerb object which interpolates transformations between idx-th and (idx + 1)-th pose.

        Returns:
            Slerp: Slerp object which is to perform interpolation.
        """
        timestamp1, x1, y1, z1, qx1, qy1, qz1, qw1, id1 = self.scan_poses[idx]
        timestamp2, x2, y2, z2, qx2, qy2, qz2, qw2, id2 = self.scan_poses[idx + 1]
        rot1 = Rotation.from_quat([qx1, qy1, qz1, qw1])
        rot2 = Rotation.from_quat([qx2, qy2, qz2, qw2])
        rotations = np.vstack((rot1.as_euler('xyz', degrees=False), rot2.as_euler('xyz', degrees=False)))
        rotations = Rotation.from_euler('xyz', rotations, degrees=False)
        slerp = Slerp([0, timestamp2 - timestamp1], rotations)

        return slerp

    def interpolate(self) -> None:
        """
        Creates more dense reconstruction based on scan point clouds from Bag file 
        and linear interpolations between poses estimated by RTABMap.
        """
        interpolated_cloud = None

        i = 0
        timestamp1, x1, y1, z1 = self.scan_poses[i][:4]
        timestamp2, x2, y2, z2 = self.scan_poses[i + 1][:4]
        slerp = self._get_slerp(i)

        for timestamp, cloud in self.clouds:
            if timestamp < timestamp1:
                continue

            if timestamp > timestamp2:
                i += 1

                if i >= len(self.scan_poses) - 1:
                    break

                timestamp1, x1, y1, z1 = self.scan_poses[i][:4]
                timestamp2, x2, y2, z2 = self.scan_poses[i + 1][:4]
                slerp = self._get_slerp(i)

            # Perform interpolation of transformation
            scl = (timestamp - timestamp1) / (timestamp2 - timestamp1)
            x = scl * (x2 - x1) + x1
            y = scl * (y2 - y1) + y1
            z = scl * (z2 - z1) + z1
            qx, qy, qz, qw = slerp([timestamp - timestamp1]).as_quat()[0]
            
            # Compute transformation matrix
            rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.eye(4)
            t[:3, :3] = np.array(rot)
            t[0, 3] = x
            t[1, 3] = y
            t[2, 3] = z

            cloud.transform(t)

            if interpolated_cloud is None:
                interpolated_cloud = copy.deepcopy(cloud)
            else:
                interpolated_cloud += cloud

        o3d.io.write_point_cloud(os.path.join(self.dir, 'interpolated_cloud.ply'), interpolated_cloud)
        o3d.visualization.draw_geometries([interpolated_cloud])
        


if __name__ == "__main__":
    interpolator = CloudInterpolator('/media/jakub/Extreme SSD/row_1/scan')
    interpolator._read_reconstructed_point_cloud()
    interpolator._read_poses()
    interpolator._collect_point_clouds('/media/jakub/Extreme SSD/2022-07-28-11-29-56_filare_1_lungo')
    interpolator.interpolate()
