from rosbags.rosbag2 import Reader
from pyproj import Proj
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, NavSatFix, Image
import math
import open3d as o3d
import copy
from abc import ABC, abstractmethod
import os
import shutil
from sklearn.preprocessing import PolynomialFeatures
import yaml
from scipy.spatial.transform import Rotation
from .utils_reconstruction import transformation_matrix, get_timestamp, sensor_transformation

with open('src/3d_reconstruction/gps_reconstruction/conf/config.yaml', 'r') as file:
        config = yaml.safe_load(file)


class Reconstructor(ABC):
    """
    Abstract class used to the baseline 3d reconstruction of the vineyard row
    based on the ROS messages containing gps data and lidar or rgbd point clouds.

    Attributes:
        sensor (str): Name of a sensor from tf_static_manual.json file used to collect point clouds.
        test_mode (bool): Indication whether to visualize intermediate computations or not.
        gps_data: List containing tuples (timestamp, (x, y, z)) representing time stamp and coordinates of a robot at the given time stamp recorded by gps.
        clouds: List containing tuples (timestamp, o3d.geometry.PointCloud).
        transformations: List containing numpy 2D matrices representing following tranformations which need to be applied to the corresponding point clouds with the same index.
        intrinsic (o3d.camera.PinholeCameraIntrinsic): Intrinsic matrix of the RGBD camera.
        camera_projection_matrix (np.ndarray): Numpy 3x4 projection matrix.
        img_width (float): Width of the images captured by the RGBD camera.
        img_height (float): Height of the images captured by the RGBD camera.
        merged_cloud (o3d.geometry.PointCloud): Merged point clouds after applying corresponding transformations.

    Methods:
        perform_reconstruction(): Collects point clouds from the Bag file and merges them all together.
        visualize_reconstruction(): Visualize reconstructed point cloud together with subsequent coordinate frames of the robot.
        get_reconstructed_point_cloud(): Returns reconstructed point cloud.
    """

    def __init__(self, output_dir: str, test_mode: bool=False):
        """
        Initializes all the necessary attributes for the reconstructor object and reads data from the Bag file.

        Args:
            test_mode (bool): Indication whether to visualize intermediate computations or not.
        """
        self.sensor = None
        self.output_dir = output_dir
        self.test_mode = test_mode

        self.gps_data = []
        self.clouds = []
        self.transformations = []
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.camera_projection_matrix = None
        self.img_width = 0
        self.img_height = 0
        self.merged_cloud = o3d.geometry.PointCloud()
        self._load_data_from_bag()
        self._make_output_dir()
    
    def _load_data_from_bag(self) -> None:
        """
        Load RGBD camera information, gps coordinates and odometry data from the BAG file.
        """
        with Reader(config['bag']['path']) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == config['bag']['topics']['camera_info']:
                    self._load_camera_information(rawdata)
                elif connection.topic == config['bag']['topics']['gps']:
                    self._load_gps_data(rawdata)

        # UTM coordinates relative to the first position
        for i in range(len(self.gps_data) - 1, -1, -1):
            x = self.gps_data[i][1][0] - self.gps_data[0][1][0]
            y = self.gps_data[i][1][1] - self.gps_data[0][1][1]
            z = self.gps_data[i][1][2] - self.gps_data[0][1][2]
            self.gps_data[i] = (self.gps_data[i][0], (x, y, z))
    
    def _load_camera_information(self, rawdata: bytes) -> None:
        """
        Loads intrinsic matrix, projection matrix and width and height of captured images by the RGBD camera.

        Args:
            rawdata (bytes): Data from the topic in raw format.
        """
        if self.camera_projection_matrix is None:
            msg = deserialize_message(rawdata, CameraInfo)
            fx = msg.k[0]
            fy = msg.k[4]
            cx = msg.k[2]
            cy = msg.k[5]

            self.intrinsic.set_intrinsics(msg.width, msg.height, fx, fy, cx, cy)
            self.camera_projection_matrix = np.reshape(np.array(msg.p), (3, 4))
            self.img_width = msg.width
            self.img_height = msg.height
                
    def _load_gps_data(self, rawdata: bytes) -> None:
        """
        Loads all the gps coordinates from the specified Bag file and project them onto the UTM system.
        Gathered data are stored in gps_data attribute as a list of tuples (timestamp, (x, y, z)).

        Args:
            rawdata (bytes): Data from the topic in raw format.
        """
        msg = deserialize_message(rawdata, NavSatFix)
        p = Proj(proj='utm', zone=32, ellps='WGS84', preserve_units=False)
        x, y = p(msg.longitude, msg.latitude)
        z = msg.altitude
        self.gps_data.append((get_timestamp(msg), (x, y, z)))
                    
    def _get_index_of_nearest_coordinates(self, timestamp: float) -> int:
        """
        Returns index of coordinates from gps_data with the timestamp closest to the one specified as an argument.

        Args:
            timestamp (float): Time stamp of the measurement.

        Returns:
            int: index of coordinates from gps_data with the closest timestamp.
        """
        gps_index = self.gps_data.index(min(self.gps_data, key=lambda x: abs(x[0] - timestamp))) 
        return gps_index

    def _gps_coords_to_numpy(self) -> np.ndarray:
        """
        Returns 2D numpy array containing following coordinates of the robot.

        Returns:
            np.ndarray: 2D numpy array with shape (n, 3) where n represents number of gps measurements and each row contains coordinates x, y, z.
        """
        gps_data_numpy = np.array([], dtype=np.float64).reshape(0, 3)
        for _, coords in self.gps_data:
            x, y, z = coords
            gps_data_numpy = np.vstack([gps_data_numpy, np.asarray([[x, y, z]])])
        return gps_data_numpy

    def _get_robot_transformation(self, timestamp: float) -> np.ndarray:
        """
        Returns 4x4 numpy matrix representing transformation of the robot at a given timestamp.
        Rotation of the robot is estimated based on a couple of consecutive gps measurements before
        and after specified timestamp. 

        Args:
            timestamp (float): Time stamp of the measurement.

        Returns:
            np.ndarray: 4x4 numpy transformation matrix.
        """

        # Internal function to compute distance beetween coordinates with index1 and index2
        def dist(index1, index2):
            x1, y1, z1 = self.gps_data[index1][1]
            x2, y2, z2 = self.gps_data[index2][1]
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        
        # Collect gps measurements within range specified by variable length
        gps_index = self._get_index_of_nearest_coordinates(timestamp)
        length = 2.5
        start_index = gps_index - 1
        end_index = gps_index + 1

        while start_index > 0:
           if dist(start_index, gps_index) > length / 2:
                break
           start_index -= 1
        start_index += 1

        while end_index < len(self.gps_data):
           if dist(end_index, gps_index) > length / 2:
                break
           end_index += 1
        end_index -= 1

        timestamp_start = self.gps_data[start_index][0]
        timestamps = []
        for idx in range(start_index, end_index + 1):
            timestamps.append(self.gps_data[idx][0] - timestamp_start)
        timestamps = np.expand_dims(np.array(timestamps), axis=1)

        # Compute yaw and pitch of the robot by fitting polynomial models to the given coordinates
        coords = self._gps_coords_to_numpy()[start_index:end_index + 1]
        xs = np.expand_dims(coords[:, 0], axis=1)
        ys = np.expand_dims(coords[:, 1], axis=1)
        zs = np.expand_dims(coords[:, 2], axis=1)

        timestamps_polynomial = PolynomialFeatures(4).fit_transform(timestamps)
        reg_x = LinearRegression().fit(timestamps_polynomial, xs)
        reg_y = LinearRegression().fit(timestamps_polynomial, ys)
        reg_z = LinearRegression().fit(timestamps_polynomial, zs)

        delta_x = 0
        for i in range(1, np.shape(reg_x.coef_)[1]):
            delta_x += i * reg_x.coef_[0][i] * (timestamp - timestamp_start) ** (i - 1)
        
        delta_y = 0
        for i in range(1, np.shape(reg_y.coef_)[1]):
            delta_y += i * reg_y.coef_[0][i] * (timestamp - timestamp_start) ** (i - 1)
        
        delta_z = 0
        for i in range(1, np.shape(reg_z.coef_)[1]):
            delta_z += i * reg_z.coef_[0][i] * (timestamp - timestamp_start) ** (i - 1)

        yaw = math.atan2(delta_y, delta_x)
        roll = 0
        pitch = -math.atan2(delta_z, np.sqrt(delta_x ** 2 + delta_y ** 2))

        timestamp_poly = PolynomialFeatures(4).fit_transform(np.array([timestamp - timestamp_start]).reshape(1, -1))
        x = reg_x.predict(timestamp_poly)
        y = reg_y.predict(timestamp_poly) 
        z = reg_z.predict(timestamp_poly)  

        # Visualize estimated linear models in a test mode
        if self.test_mode:
            plt.scatter([coord[1][0] for coord in self.gps_data], [coord[1][1] for coord in self.gps_data])
            plt.scatter(xs, ys)
            plt.plot(reg_x.predict(timestamps_polynomial), reg_y.predict(timestamps_polynomial))
            plt.scatter(x, y, c='red', marker='x')
            plt.show()

            ax = plt.figure().add_subplot(projection='3d')
            ax.scatter(xs, ys, zs)
            ax.plot3D(reg_x.predict(timestamps_polynomial), reg_y.predict(timestamps_polynomial), reg_z.predict(timestamps_polynomial))
            plt.show()

        t = transformation_matrix(roll, pitch, yaw, x, y, z)

        return t
        
    def _merge_point_clouds_no_icp(self) -> None:
        """
        Merges all the collected point clouds into one reconstructed point cloud and saves it into merged_cloud atttribute.
        Furthermore, it saves all the estimated transformations.
        """
        points = np.array([], dtype=np.float64).reshape(0, 3)
        colors = np.array([], dtype=np.float64).reshape(0, 3)

        x_gps = []
        y_gps = []
        times1 = []
        times2 = []

        for i in range(len(self.clouds)):
            timestamp, _ = self.clouds[i]
            transformation_matrix = self._get_robot_transformation(timestamp)

            # Compute tranformation from sensor to world frame
            t = transformation_matrix @ sensor_transformation(self.sensor)
            self.transformations.append(t)

            x_gps.append([transformation_matrix[0, 3]])
            y_gps.append([transformation_matrix[1, 3]])
            times1.append(timestamp)
            idx = self._get_index_of_nearest_coordinates(timestamp)
            times2.append(self.gps_data[idx][0])

            cloud = copy.deepcopy(self.clouds[i][1])
            cloud.transform(t)

            points = np.vstack([points, np.asarray(cloud.points)])
            colors = np.vstack([colors, np.asarray(cloud.colors)])

            if self.test_mode:
                if i == 0:
                    points1 = np.asarray(cloud.points)[:, 0:2]
                if i == 1:
                    points2 = np.asarray(cloud.points)[:, 0:2]
                    plt.scatter(points1[:, 0], points1[:, 1])
                    plt.scatter(points2[:, 0], points2[:, 1])
                    plt.show()
                if i > 1:
                    points1 = np.copy(points2)
                    points2 = np.asarray(cloud.points)[:, 0:2]  
                    plt.scatter(points1[:, 0], points1[:, 1])
                    plt.scatter(points2[:, 0], points2[:, 1])
                    plt.show()
                
                plt.scatter(x_gps, y_gps)
                plt.show()

                plt.scatter(times1, times2)
                plt.show()

        self.merged_cloud.points = o3d.utility.Vector3dVector(points)
        self.merged_cloud.colors = o3d.utility.Vector3dVector(colors)

    def _merge_point_clouds_icp(self) -> None:
        """
        Merges all the collected point clouds into one reconstructed point cloud, 
        uses ICP to improve estimated transformations and saves it into merged_cloud atttribute.
        Furthermore, it saves all the estimated transformations. 
        """
        timestamp, target = self.clouds[0]
        transformation_target = self._get_robot_transformation(timestamp)
        target.transform(transformation_target)

        points = np.array(target.points)
        colors = np.array(target.colors)
        for i in range(1, len(self.clouds)):
            timestamp, source = self.clouds[i]
            transformation_source = self._get_robot_transformation(timestamp)

            # Estimated transformations are used to compute initial transformation between source and target point cloud
            trans_init = np.linalg.inv(transformation_target) @ transformation_source

            threshold = 0.05
            reg_p2p = o3d.pipelines.registration.registration_icp(source, 
                                                                    target, 
                                                                    threshold, 
                                                                    trans_init, 
                                                                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
            
            t = transformation_target @ reg_p2p.transformation
            source.transform(t)
            self.transformations.append(t)

            points = np.vstack([points, np.asarray(source.points)])
            colors = np.vstack([colors, np.asarray(source.colors)])

            target.points = o3d.utility.Vector3dVector(points)
            target.colors = o3d.utility.Vector3dVector(colors)

            if self.test_mode:
                points1 = np.asarray(target.points)[:, 0:2]  
                points2 = np.asarray(source.points)[:, 0:2]  
                plt.scatter(points1[:, 0], points1[:, 1])
                plt.scatter(points2[:, 0], points2[:, 1])
                plt.show()

        self.merged_cloud.points = o3d.utility.Vector3dVector(points)
        self.merged_cloud.colors = o3d.utility.Vector3dVector(colors)

    @abstractmethod
    def _collect_point_clouds(self, time_start: float=0.0, time_end: float=-1.0, every_k: int=1) -> None:
        """
        Abstract method. It's implementations should extract point clouds from the Bag file and 
        store them in clouds attribute together with time stamps at which measurements were taken.

        Args:
            time_start (float): Value specifying point cloud with the earliest time stamp which should be taken into account.
            time_start (float): Value specifying point cloud with the latest time stamp which should be taken into account.
            every_k (int): Value specifying that only every k-th point cloud will be considered.
        """
        pass

    def _merge_point_clouds(self, icp: bool=False) -> None:
        """
        Merges all the collected point clouds into one reconstructed point cloud and saves it into merged_cloud atttribute.
        Furthermore, it saves all the estimated transformations.

        Args:
            icp (bool): Indicates whether to apply ICP registration or not.
        """
        if icp:
            self._merge_point_clouds_icp()
        else:
            self._merge_point_clouds_no_icp()

    def perform_reconstruction(self, time_start: float=0.0, time_end: float=-1.0, every_k: int=1, icp: bool=False) -> None:
        '''
        Collects point clouds from the Bag file and merges them all together.

        Args:
            time_start (float): Value specifying point cloud with the earliest time stamp which should be taken into account.
            time_start (float): Value specifying point cloud with the latest time stamp which should be taken into account.
            every_k (int): Value specifying that only every k-th point cloud will be considered.
            icp (bool): Indicates whether to apply ICP registration or not.
        '''
        self._collect_point_clouds(every_k=every_k, time_start=time_start, time_end=time_end)
        self._merge_point_clouds(icp=icp)
        self._save_reconstruction()
        
    def visualize_reconstruction(self, show_poses: bool=True) -> None:
        """
        Visualize reconstructed point cloud together with subsequent coordinate frames of the robot.

        Args:
            show_poses (bool): Indicates whether to visualize consecutive orientations of sensor which captured point clouds.
        """
        geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)]

        t_x, t_y, t_z = self.gps_data[0][1]
        if show_poses:
            for t in self.transformations:
                mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)
                mesh_frame.transform(t)
                mesh_frame.translate((-t_x, -t_y, -t_z))
                geometries.append(mesh_frame)
        cloud = copy.deepcopy(self.merged_cloud)
        cloud.translate((-t_x, -t_y, -t_z))
        geometries.append(cloud)
        o3d.visualization.draw_geometries(geometries)

    def get_reconstructed_point_cloud(self) -> o3d.geometry.PointCloud:
        """
        Returns reconstructed point cloud.

        Returns:
            o3d.geometry.PointCloud: Reconstructed point cloud.
        """
        return self.merged_cloud
    
    def _make_output_dir(self) -> None:
        '''
        Creates empty directory where reconstruction is to be saved.
        '''
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        os.makedirs(os.path.join(self.output_dir, 'clouds'))
    
    def _save_reconstruction(self) -> None:
        """
        Saves reconstructed point cloud and sensor poses into a directory.
        """
        file = open(os.path.join(self.output_dir, "sensor_poses.txt"), "w")
        file.write("#timestamp x y z qx qy qz qw id\n")

        for i in range(len(self.clouds)):
            id = i + 1
            timestamp, cloud = self.clouds[i]
            transformation_matrix = self._get_robot_transformation(timestamp)

            # Compute tranformation from camera to world frame
            t = transformation_matrix @ sensor_transformation(self.sensor)
            x = t[0, 3]
            y = t[1, 3]
            z = t[2, 3]
            quat = Rotation.from_matrix(t[:3, :3]).as_quat()
            qx, qy, qz, qw = quat
            line = [timestamp, x, y, z, qx, qy, qz, qw, id]
            line = [str(val) for val in line]
            file.write(' '.join(line) + '\n')
            o3d.io.write_point_cloud(os.path.join(self.output_dir, 'clouds', "{}.ply".format(id)), cloud)
            
        file.close()
        o3d.io.write_point_cloud(os.path.join(self.output_dir, "reconstructed_cloud.ply"), self.merged_cloud)
