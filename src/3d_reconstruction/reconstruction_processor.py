import open3d as o3d
import os
import sys
import shutil
from PIL import Image
from ..segmentation.models import predict
import torch
from matplotlib import pyplot as plt
from transformers import UperNetForSemanticSegmentation
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from .gps_reconstruction import remove_distant_points, remove_points_below_ground, sensor_transformation
import copy
from colordict import ColorDict
import yaml
from sklearn.linear_model import LinearRegression
from .volume_estimator import VolumeEstimator
from numba import njit
import time
import math
from abc import ABC, abstractmethod
from scipy.optimize import curve_fit
from pyproj import Proj


with open('src/3d_reconstruction/conf/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

@njit
def _compute_occlusions(vectors: np.ndarray, lengths: np.ndarray, norms: np.ndarray) -> np.ndarray:
    """
    Returns list with number of points by which consecutive points are occluded.
    """
    n = np.shape(vectors)[1]
    num_collisions = np.zeros((n,))

    for i in range(0, n, 1000):
        n_start = i
        n_stop = np.min(np.array([i + 1000, n]))

        norms_mat = norms * norms.T[0, n_start:n_stop]
        cosines = np.divide(np.dot(vectors.T, vectors[:, n_start:n_stop]), norms_mat)
        sines = np.sqrt(1 - cosines ** 2)
        d = sines * lengths.T[:, n_start:n_stop]
        d_proj = cosines * lengths.T[:, n_start:n_stop]
        collisions = (d < 0.05) & (0 < d_proj) & (d_proj < lengths)
        num_collisions += np.sum(collisions, axis=1)
    
    return num_collisions

def _compute_occlusions_naive(vectors: np.ndarray) -> np.ndarray:
    """
    Returns list with number of points by which consecutive points are occluded. Naive, loop-based implementation.
    """
    n = np.shape(vectors)[1]
    num_collisions = np.zeros((n,))

    for i in range(n):
        v1 = vectors[:, i]
        for j in range(n):
            v2 = vectors[:, j]
            cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            sine = np.sqrt(1 - cosine ** 2)
            d = sine * np.linalg.norm(v2)
            d_proj = cosine * np.linalg.norm(v2)
            if (d < 0.2) and (0 < d_proj) and (d_proj < np.linalg.norm(v1)):
                num_collisions[i] = num_collisions[i] + 1

    return num_collisions

class ReconstructionProcessor:
    """
    Class used to segment reconstruction obtained from RTABMap and extract locations of individual trunks.

    Attributes:
        camera_poses (dict[np.ndarray]): Transformation matrices applied to obtain following cameras poses. Key identified by id of a node.
        images (dict[o3d.geometry.Image]): Images from each node.
        depth_images (dict[o3d.geometry.Image]): Corresponding depth images from each node.
        segmentations (dict[np.ndarray]): Segmentation maps for each rgb image.
        projection_matrix (np.ndarray): Projection matrix of the RGBD camera.
        intrinsic (o3d.camera.PinholeCameraIntrinsic): Intrinsic matrix of the RGBD camera.
        reconstructed_cloud (o3d.geometry.Pointcloud): Reconstructed point cloud.
        reconstructed_segmented_cloud (o3d.geometry.Pointcloud): Reconstructed point cloud with colors indicating to which class particular point belongs.
    
        Methods:
              save_segmented_reconstruction(): Apply segmentation to reconstructed point clouds and save obtained results.
              show(): Visualize obtained reconstructions.
              cluster(): Perform clustering of points from rgbd point cloud classified as trunks.
    """

    def __init__(self, data_source: str, reconstruction_type: str, dir: str) -> None:
        self.data_source = data_source # rgbd / lidar / scan
        self.reconstruction_type = reconstruction_type # rtabmap / baseline
        self.dir = dir

        self.camera_poses = dict()
        self.images = dict()
        self.depth_images = dict()
        self.segmentations = dict()
        self.projection_matrix = None
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.reconstructed_cloud = None
        self.reconstructed_segmented_cloud = None

    def _read_reconstructed_point_cloud(self) -> None:
        """
        Reads reconstructed point cloud.
        """
        if self.data_source == 'rgbd':
            reconstructed_cloud = o3d.geometry.PointCloud()
            for id in sorted(self.camera_poses.keys()):
                rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(self.images[id], self.depth_images[id], convert_rgb_to_intensity=False)
                cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, self.intrinsic)

                # cloud.transform(sensor_transformation('d435i_link'))
                # cloud = remove_points_below_ground(cloud, threshold=0.15, test_mode=False)
                # cloud.transform(np.linalg.inv(sensor_transformation('d435i_link')))

                cloud.transform(self.camera_poses[id])
                
                cloud = cloud.voxel_down_sample(voxel_size=0.02)
                reconstructed_cloud += cloud
               
                print(np.shape(np.array(reconstructed_cloud.points)), np.shape(np.array(cloud.points)))

            #reconstructed_cloud = reconstructed_cloud.voxel_down_sample(voxel_size=0.2)
            self.reconstructed_cloud = copy.deepcopy(reconstructed_cloud)
        else:
            if self.reconstruction_type == 'rtabmap':
                path = os.path.join(os.getcwd(), config['path'], self.dir, self.data_source, 'interpolated_cloud.ply')
            elif self.reconstruction_type == 'baseline':
                path = os.path.join(os.getcwd(), config['path'], self.dir, self.data_source, 'reconstructed_cloud.pcd')
            elif self.reconstruction_type == 'art_slam':
                path = os.path.join(os.getcwd(), config['path'], self.dir, self.data_source, 'custom_map.pcd')

            self.reconstructed_cloud = o3d.io.read_point_cloud(path)
        
        self.reconstructed_cloud = self._extract_row_of_interest(self.reconstructed_cloud)

    def _read_intrinsic_matrix(self) -> None:
        """
        Reads intrinsic and projection matrix of RGBD camera.
        """
        if self.reconstruction_type == 'rtabmap':
            calib_path = os.path.join(os.getcwd(), config['path'], self.dir, 'rgbd', 'rtabmap_calib')
        elif self.reconstruction_type == 'baseline':
            calib_path = os.path.join(os.getcwd(), config['path'], self.dir, 'rgbd', 'calib')
        elif self.reconstruction_type == 'art_slam':
            calib_path = os.path.join(os.getcwd(), config['path'], self.dir, self.data_source, 'calib')

        with open(os.path.join(calib_path, os.listdir(calib_path)[0]), 'r') as file:
            if self.reconstruction_type == 'rtabmap':
                for i in range(2):
                    _ = file.readline()
            calib = yaml.safe_load(file)

        rows = calib["projection_matrix"]["rows"]
        cols = calib["projection_matrix"]["cols"]
        self.projection_matrix = np.array(calib["projection_matrix"]["data"]).reshape((rows, cols))

        fx = calib["camera_matrix"]["data"][0]
        fy = calib["camera_matrix"]["data"][4]
        cx = calib["camera_matrix"]["data"][2]
        cy = calib["camera_matrix"]["data"][5]

        self.intrinsic.set_intrinsics(calib["image_width"], calib["image_height"], fx, fy, cx, cy)

    def _read_poses(self) -> None:
        """
        Reads consequtive poses of camera and transform them into transformation matrices.
        """
        def read_poses_file(file_path: str) -> dict():
            file = open(file_path, 'r')
            poses = dict()
            header = True
            
            for line in file:
                if header:
                    header = False
                    continue

                values = line[:-1].split(' ')
                timestamp, x, y, z, qx, qy, qz, qw, id = [float(val) for val in values]
                id = int(id)

                rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                transformation_matrix = np.eye(4)
                transformation_matrix[:3, :3] = np.array(rot)
                transformation_matrix[0, 3] = x
                transformation_matrix[1, 3] = y
                transformation_matrix[2, 3] = z

                poses[id] = transformation_matrix
            
            return poses

        if self.reconstruction_type == 'rtabmap':
            camera_poses_path = os.path.join(os.getcwd(), config['path'], self.dir, 'rgbd', "rtabmap_camera_poses.txt")
        elif self.reconstruction_type == 'baseline':
            camera_poses_path = os.path.join(os.getcwd(), config['path'], self.dir, 'rgbd', "sensor_poses.txt")
        elif self.reconstruction_type == 'art_slam':
            camera_poses_path = os.path.join(os.getcwd(), config['path'], self.dir, self.data_source, "camera_poses.txt")

        self.camera_poses = read_poses_file(camera_poses_path)

    def _get_segmentations(self, read: bool=False) -> None:
        """
        Reads RGB images, corresponding depth images and compute segmentation maps using specified segmentation model.
        Afterwards saves all the segmentation maps to reduce computation times for future runs.

        Args:
            read (bool): Specifies whether to read already computed segmentation maps from saved files or predict new ones.
        """
        checkpoint = os.path.join(os.getcwd(), config['segmentation_model_checkpoint'])
        model = UperNetForSemanticSegmentation.from_pretrained(checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        depth_dir = None
        if self.reconstruction_type == 'rtabmap':
            rgb_dir = os.path.join(os.getcwd(), config['path'], self.dir, self.data_source,'rtabmap_rgb')
            if self.data_source == 'rgbd':
                depth_dir = os.path.join(os.getcwd(), config['path'], self.dir, self.data_source, 'rtabmap_depth')
        elif self.reconstruction_type == 'baseline':
            rgb_dir = os.path.join(os.getcwd(), config['path'], self.dir, 'rgbd', 'rgb')
            depth_dir = os.path.join(os.getcwd(), config['path'], self.dir, 'rgbd', 'depth')
        elif self.reconstruction_type == 'art_slam':
            rgb_dir = os.path.join(os.getcwd(), config['path'], self.dir, self.data_source, 'rgb')
            if self.data_source == 'rgbd':
                depth_dir = os.path.join(os.getcwd(), config['path'], self.dir, self.data_source, 'depth')
   
        img_names = sorted(os.listdir(rgb_dir))

        # Create new, empty folder for segmentation maps
        if not read:
            dir = os.path.join(os.getcwd(), config['path'], self.dir, 'segmentations')
            if os.path.exists(dir):
                shutil.rmtree(dir)
            os.makedirs(dir)

        i = 1
        for img_name in img_names:
            print('SEG', i)
            i += 1
            id = int(img_name.split('.')[0])

            if not read: 
                img = Image.open(os.path.join(rgb_dir, img_name))
                img_rotated = img.rotate(-90, Image.NEAREST, expand=1)
                logits, upsampled_logits = predict.predict_pretrained_model(model, config['segmentation_pretrained_model'], img_rotated)
                self.segmentations[id] = torch.rot90(upsampled_logits.argmax(dim=1)[0]).numpy()
            else:
                with open(os.path.join(os.getcwd(), config['path'], self.dir, 'segmentations', '{}.npy'.format(id)), 'rb') as f:
                    self.segmentations[id] = np.load(f)

            if depth_dir is not None:
                self.depth_images[id] = o3d.io.read_image(os.path.join(depth_dir, img_name.split('.')[0] + '.png'))
            self.images[id] = o3d.io.read_image(os.path.join(rgb_dir, img_name))

            if not read:
                with open(os.path.join(os.getcwd(), config['path'], self.dir, 'segmentations', '{}.npy'.format(id)), 'wb') as f:
                    np.save(f, self.segmentations[id])
            
            # if id > 283:
            #     color_seg = np.zeros((self.segmentations[id].shape[0], self.segmentations[id].shape[1], 3), dtype=np.uint8)
            #     color_palette = [[0, 255, 50], [139, 69, 19], [255, 255, 255]]
            #     for label, color in enumerate(color_palette):
            #         color_seg[self.segmentations[id] == label, :] = color

            #     img = np.array(self.images[id]) * 0.5 + color_seg * 0.5
            #     img = img.astype(np.uint8)
            #     img = np.rot90(img, 1, (1,0))

            #     plt.figure(figsize=(15, 10))
            #     plt.imshow(img)
            #     plt.show()

    def _project_segmentation_on_rgbd_cloud(self, one_side: bool=False):
        """
        Performs projection of segmentation maps on reconstructed point cloud obtained from RGBD images to determine whether this is canopy, trunk or background.

        Args:
            one_side (bool): Indicates whether apply segmentation only to one side of the row.
        """
        reconstructed_segmented_cloud = o3d.geometry.PointCloud()
        ids_to_include = []
        if one_side:
            ids_to_include = self._get_one_side_ids()

        for id in sorted(self.camera_poses.keys()):
            print("SEG", id)

            seg = np.array(self.images[id])
            seg[self.segmentations[id] == 0] = np.array([0, 255, 0])
            seg[self.segmentations[id] == 1] = np.array([255, 0, 0])
            seg[self.segmentations[id] == 2] = np.array([0, 0, 0])
            
            if id not in ids_to_include:
                seg[:] = np.array([0, 0, 0])

            seg = o3d.geometry.Image((np.ascontiguousarray(seg).astype(np.uint8)))

            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(seg, self.depth_images[id], convert_rgb_to_intensity=False)
            cloud_segmented = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, self.intrinsic)

            # cloud_segmented.transform(sensor_transformation('d435i_link'))
            # cloud_segmented = remove_points_below_ground(cloud_segmented, threshold=0.15, test_mode=False)
            # cloud_segmented.transform(np.linalg.inv(sensor_transformation('d435i_link')))

            cloud_segmented.transform(self.camera_poses[id])

            cloud_segmented = cloud_segmented.voxel_down_sample(voxel_size=0.02)
            reconstructed_segmented_cloud += cloud_segmented
            
        #reconstructed_segmented_cloud = reconstructed_segmented_cloud.voxel_down_sample(voxel_size=0.2)
        self.reconstructed_segmented_cloud = self._extract_row_of_interest(reconstructed_segmented_cloud)


    def _extract_row_of_interest(self, cloud: o3d.geometry.PointCloud()) -> None:
        """
        Extract only row of interest from the point cloud. Linear regression is estimated to capture lower and upper path
        and only points between those lines are considered.
        """

        x = []
        y = []
        for id in sorted(self.camera_poses.keys()):
            x.append(self.camera_poses[id][0, 3])
            y.append(self.camera_poses[id][1, 3])

        x = np.expand_dims(np.array(x), axis=1)
        y = np.expand_dims(np.array(y), axis=1)

        reg = LinearRegression().fit(x, y)
        yaw = np.arctan2(reg.coef_[0] * (x[-1, 0] - x[0, 0]), x[-1, 0] - x[0, 0])

        res = np.abs(reg.predict(x) - y) / np.sqrt(reg.coef_[0] ** 2 + 1)
        d = np.mean(res)
        #d = 0

        plt.scatter(x, y)
        plt.plot(x, reg.predict(x) - d / np.cos(yaw), color='r')
        plt.plot(x, reg.predict(x) + d / np.cos(yaw), color='r')
        plt.show()

        x_min = np.min(x)
        x_max = np.max(x)

        points = np.asarray(cloud.points)
        x = np.expand_dims(points[:, 0], axis=1)
        y = np.expand_dims(points[:, 1], axis=1)
        dists = np.abs(reg.predict(x) - y) / np.sqrt(reg.coef_[0] ** 2 + 1)
        #dists = (reg.predict(x) - y) / np.sqrt(reg.coef_[0] ** 2 + 1)
        indices = np.where((dists <= d) & (x_min <= x) & (x <= x_max))[0].tolist()
        #indices = np.where((dists <= 0) & (dists >= -3) & (x_min <= x) & (x <= x_max))[0].tolist()
        cloud = cloud.select_by_index(indices)

        return cloud

    def _get_one_side_ids(self) -> list[int]:
        """
        Returns ids of poses and images representing only one side of the reconstructed row.

        Returns:
            list[int]: List of ids of one side of the row.
        """
        ids = sorted(self.camera_poses.keys())
        v = []
        for i in range(1, len(ids)):
            id1 = ids[i - 1]
            id2 = ids[i]

            x1 = self.camera_poses[id1][0, 3]
            y1 = self.camera_poses[id1][1, 3]
            x2 = self.camera_poses[id2][0, 3]
            y2 = self.camera_poses[id2][1, 3]
            v.append(math.atan2(y2 - y1, x2 - x1))
        
        v = np.array(v)
        v -= np.min(v)
        plt.scatter(np.linspace(-1, 1, len(v)), v)
        plt.show()

        def sigmoid(x, L, x0, k, b):
            y = L / (1 + np.exp(-k*(x-x0))) + b
            return y
        
        xdata, ydata = np.linspace(-1, 1, len(v)), v
        popt, pcov = curve_fit(sigmoid, xdata[20:], ydata[20:], maxfev=10000, method='dogbox')
        print(popt)
        y = sigmoid(xdata, *popt)

        plt.scatter(xdata, ydata)
        plt.plot(xdata, y)
        plt.show()

        one_side_ids = []
        for i, x in enumerate(xdata):
            if x < popt[1]:
                one_side_ids.append(ids[i])

        return one_side_ids

    def _project_segmentation_on_reconstructed_cloud(self, one_side: bool=False) -> None:
        """
        Project points from LiDAR point cloud on segmentation maps to determine whether this is canopy, trunk or background.

        Args:
            one_side (bool): Indicates whether apply segmentation only to one side of the row.
        """
        self.reconstructed_cloud = self.reconstructed_cloud.voxel_down_sample(voxel_size=0.05)
        points = np.array(self.reconstructed_cloud.points)
        colors = np.zeros(points.shape)

        points = np.hstack((points, np.ones((points.shape[0], 1))))
        points = points.T

        ids_to_include = []
        if one_side:
            ids_to_include = self._get_one_side_ids()

        for id in sorted(self.camera_poses.keys()):
            print(id)

            if id not in ids_to_include:
                continue

            # Compute transformation from the camera frame to the image frame
            t_world_camera = np.linalg.inv(self.camera_poses[id])
            t_camera_img = self.projection_matrix @ t_world_camera

            # Project points onto the image
            points_projected = t_camera_img @ points
            z = np.copy(points_projected[-1, :])
            points_projected /= points_projected[-1, :]
            points_projected = points_projected.astype(int)
            height, width = np.shape(self.segmentations[id])

            indices = np.where((0 <= points_projected[0, :]) & (points_projected[0, :] < width) & (0 <= points_projected[1, :]) & (points_projected[1, :] < height) & (z > 0))[0].tolist()

            # Disregard photos taken at the top of the row as it requires projection of vast majority of points
            if np.shape(points_projected)[1] * 0.1 < len(indices):
                continue

            points_filtered = points[0:3, indices]
            camera_pose = np.expand_dims(self.camera_poses[id][0:3, 3], axis=1)

            vectors = points_filtered - camera_pose
            lengths = np.expand_dims(np.linalg.norm(vectors, axis=0), axis=1)
            norms = np.expand_dims(np.linalg.norm(vectors, axis=0), axis=1)

            start = time.time()
            num_collisions = _compute_occlusions(vectors, lengths, norms)
            end = time.time()
            print("Elapsed optimized = %s" % (end - start))

            # Iterate over all the points in the point cloud already projected onto the image
            for i in range(len(indices)):
                idx = indices[i]
                if num_collisions[i] > 10:
                    continue

                if self.segmentations[id][points_projected[1, idx], points_projected[0, idx]] == 0.0:
                    colors[idx, :] = np.array([0, 255, 0]) / 255.0
                elif self.segmentations[id][points_projected[1, idx], points_projected[0, idx]] == 1.0:
                    colors[idx, :] = np.array([255, 0, 0]) / 255.0

        self.reconstructed_segmented_cloud = copy.deepcopy(self.reconstructed_cloud)
        self.reconstructed_segmented_cloud.colors = o3d.utility.Vector3dVector(colors)

    def _perform_cloud_segmentation(self, one_side: bool=False) -> None:
        """
        Performs appropriate segmentation depending on the source of point cloud.

        Args:
            one_side (bool): Indicates whether apply segmentation only to one side of the row.
        """
        if self.data_source == 'rgbd':
            self._project_segmentation_on_rgbd_cloud(one_side=one_side)
        else:
            self._project_segmentation_on_reconstructed_cloud(one_side=one_side)

    def _show(self, cloud: o3d.geometry.PointCloud) -> None:
        """
        Visualizes a point cloud with respect to the first pose at which measurements were collected.

        Args:
            cloud (o3d.geometry.PointCloud): Point cloud to be visualized.
        """
        t_x = self.camera_poses[1][0, 3]
        t_y = self.camera_poses[1][1, 3]
        t_z = self.camera_poses[1][2, 3]

        cloud_centered = copy.deepcopy(cloud)
        cloud_centered.translate((-t_x, -t_y, -t_z))
        o3d.visualization.draw_geometries([cloud_centered])

    def _generate_point_clouds(self) -> None:
         """
         Create point clouds for each node reconstructed from RGBD camera.
         """
         for id in sorted(self.camera_poses.keys()):
            print('XD', id)
            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(self.images[id], self.depth_images[id], convert_rgb_to_intensity=False)
            cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, self.intrinsic)
            self.point_clouds_rgbd[id] = cloud

    def _apply_icp(self):
        """
        Apply color ICP registration and modify camera_poses accordingly.
        """
        clouds = []
        ts = []

        ids = sorted(self.camera_poses.keys())
        trans = self.camera_poses[ids[0]]
        cloud = copy.deepcopy(self.point_clouds_rgbd[ids[0]])
        cloud.transform(trans)
        clouds.append(cloud)
        ts.append(trans)

        previous = False

        reconstructed_cloud = o3d.geometry.PointCloud()
        for i in range(1, len(ids)):
            id1 = ids[i - 1]
            id2 = ids[i]

            t, flag = self._icp(id1, id2)

            if flag and previous:
                trans = ts[-1] @ t @ np.linalg.inv(self.camera_poses[id1]) @ self.camera_poses[id2]
            else:
                trans = self.camera_poses[id1] @ t @ np.linalg.inv(self.camera_poses[id1]) @ self.camera_poses[id2]

            if flag:
                previous = True
            else:
                previous = False

            cloud = copy.deepcopy(self.point_clouds_rgbd[id2])
            cloud.transform(trans)
            clouds.append(cloud)
            ts.append(trans)

            reconstructed_cloud += cloud

            self.camera_poses[id2] = trans

    def _icp(self, id1: int, id2: int) -> np.ndarray:
        """
        Performs color ICP registration between rgbd point cloud with id equal to id1 and id2.
        If fitness level of registration satisfies specified contraint obtained transformation is returned,
        otherwise identity matrix is returned.

        Args:
            id1 (int): Id of first rgbd point cloud.
            id2 (int): Id of second rgbd point cloud.

        Returns:
            np.ndarray: Transformation obtained between point clouds with id1 and id2 using ICP registration.
        """
        def draw_registration_result_original_color(source, target, transformation):
            source_temp = copy.deepcopy(source)
            source_temp.transform(transformation)
            o3d.visualization.draw_geometries([source_temp, target, o3d.geometry.TriangleMesh.create_coordinate_frame()])

        target = copy.deepcopy(self.point_clouds_rgbd[id1])
        source = copy.deepcopy(self.point_clouds_rgbd[id2])

        source.transform(np.linalg.inv(self.camera_poses[id1]) @ self.camera_poses[id2])

        max_iter = 50
        radius = 0.05
        current_transformation = np.identity(4)

        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        try:
            result_icp = o3d.pipelines.registration.registration_colored_icp(
                source, target, radius, current_transformation,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                relative_rmse=1e-6,
                                                                max_iteration=max_iter))
            current_transformation = result_icp.transformation
            #draw_registration_result_original_color(source, target, result_icp.transformation)

            if result_icp.fitness < 0.65:
                return np.identity(4), False
            
            return current_transformation, True
        
        except:
            return np.identity(4), False
        
    def _save(self) -> None:
        """
        Save obtained reconstructions.
        """
        path = os.path.join(os.getcwd(), config['path'], self.dir, 'segmented_reconstruction')

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        o3d.io.write_point_cloud(os.path.join(path, "reconstructed_cloud.ply"), self.reconstructed_cloud)
        o3d.io.write_point_cloud(os.path.join(path, "reconstructed_segmented_cloud.ply"), self.reconstructed_segmented_cloud)

    def get_trunks_positions(self) -> np.ndarray:
        """
        Perform clustering of points from rgbd point cloud classified as trunks.

        Returns:
            np.ndarray: 2d numpy array [[x, y, z], ...] containing estimated locations of trunks.
        """
        trunks = self.get_trunks_cloud()
        x = np.array(trunks.points)
        trunks_locations = []

        clustering = DBSCAN(eps=0.25, min_samples=8).fit(x[:, :2])
        for label in np.unique(clustering.labels_):
            plt.scatter(x[clustering.labels_ == label, 0], x[clustering.labels_ == label, 1])
            x_cluster, y_cluster, z_cluster = np.mean(x[clustering.labels_ == label, 0], axis=0), np.mean(x[clustering.labels_ == label, 1], axis=0), np.mean(x[clustering.labels_ == label, 2], axis=0)
            trunks_locations.append([x_cluster, y_cluster, z_cluster])
            plt.scatter(x_cluster, y_cluster, s=100, c='red', marker='x')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

        colors = ColorDict()
        new_colors = np.array(trunks.colors)
        for label in np.unique(clustering.labels_):
            if label == -1:
                continue
            new_colors[clustering.labels_ == label] = np.array(colors[list(colors.keys())[label]]) / 255.

        trunks.colors = o3d.utility.Vector3dVector(new_colors)
        self._show(trunks)

        return np.array(trunks_locations)
            
    def process(self, apply_icp: bool=False, one_side: bool=False, read_segmentations: bool=False) -> None:
        """
        Apply segmentation to reconstructed point clouds and save obtained results.

        Args:
            apply_icp (bool): Determines whether to perform ICP registration to correct estimated camera poses.
        """
        self._read_intrinsic_matrix()
        self._read_poses()
        self._get_segmentations(read=read_segmentations)
        self._read_reconstructed_point_cloud()
        
        if apply_icp:
            self._apply_icp()
 
        self._perform_cloud_segmentation(one_side=one_side)
        self._save()

        self._show(self.reconstructed_cloud)
        self._show(self.reconstructed_segmented_cloud)

        print(np.shape(np.array(self.reconstructed_cloud.points)), np.shape(np.array(self.reconstructed_segmented_cloud.points)))

    def read(self):
        self._read_intrinsic_matrix()
        self._read_poses()
        path = os.path.join(os.getcwd(), config['path'], self.dir, 'segmented_reconstruction')
        self.reconstructed_cloud = o3d.io.read_point_cloud(os.path.join(path, 'reconstructed_cloud.ply'))
        self.reconstructed_segmented_cloud = o3d.io.read_point_cloud(os.path.join(path, 'reconstructed_segmented_cloud.ply'))

    def get_filtered_cloud(self) -> o3d.geometry.PointCloud:
        """
        Returns reconstructed point cloud after filtering out points belonging to neither canopy nor trunk class.

        Returns:
            o3d.geometry.PointCloud: Filtered point cloud.
        """
        indices = np.where((np.array(self.reconstructed_segmented_cloud.colors) != [0, 0, 0]).any(axis=1))[0].tolist()
        filtered_cloud = self.reconstructed_cloud.select_by_index(indices)
        self._show(filtered_cloud)
        return filtered_cloud

    def get_canopy_cloud(self) -> o3d.geometry.PointCloud:
        """
        Returns reconstructed point cloud with points representing only canopy part.

        Returns:
            o3d.geometry.PointCloud: Canopy point cloud.
        """
        indices = np.where((np.array(self.reconstructed_segmented_cloud.colors) == [0, 1, 0]).all(axis=1))[0].tolist()
        canopy_cloud = self.reconstructed_cloud.select_by_index(indices)
        self._show(canopy_cloud)
        return canopy_cloud
    
    def get_trunks_cloud(self) -> o3d.geometry.PointCloud:
        """
        Returns reconstructed point cloud with points representing only trunks part.

        Returns:
            o3d.geometry.PointCloud: Trunks point cloud.
        """
        indices = np.where((np.array(self.reconstructed_segmented_cloud.colors) == [1, 0, 0]).all(axis=1))[0].tolist()
        trunks_cloud = self.reconstructed_cloud.select_by_index(indices)
        return trunks_cloud
    
    def get_segmented_cloud(self) -> o3d.geometry.PointCloud:
        """
        Returns reconstructed point cloud after filtering out points belonging to neither canopy nor trunk class.
        Remain green and red colors indicating canopy and trunks. Mainly for visualization purposes.

        Returns:
            o3d.geometry.PointCloud: Filtered point cloud with colors indicating canopy and trunks.
        """
        indices = np.where((np.array(self.reconstructed_segmented_cloud.colors) != [0, 0, 0]).any(axis=1))[0].tolist()
        segmented_cloud = self.reconstructed_segmented_cloud.select_by_index(indices)
        self._show(segmented_cloud)
        return segmented_cloud
