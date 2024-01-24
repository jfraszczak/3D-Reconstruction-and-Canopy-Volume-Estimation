from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import math
import open3d as o3d
import os
import shutil
from .gps_reconstruction import transformation_matrix
from pyproj import Proj


class VolumeEstimator:
    """
    Class used to volume estimation of a given vineyard row. 

    Attributes:
        row (o3d.geometry.PointCloud): Point cloud representing vineyard row whose volume is to be computed.
        applied_transformation (np.ndarray): Transformation that has been applied to align row with x axis.

    Methods:
        load_cloud(): Loads point cloud whose volume is to be computed.
        load_cloud_from_file(): Loads point cloud whose volume is to be computed from the specified file.
        compute_volumes(): Generates 3D models of trees in a row segment by segment using different methods and compute their volumes.
    """

    def __init__(self):
        """
        Initializes all the necessary attributes for the volume estimation.
        """
        self.row = None
        self.applied_transformation = None
        self.volume_density_function = None

    def load_cloud(self, row: o3d.geometry.PointCloud) -> None:
        """
        Loads cloud whose volume is to be computed.

        Args:
            row (o3d.geometry.PointCloud): Point cloud representing row whose volume is to be computed.
        """
        self.row = row
        self._align_row_with_axis()

    def load_cloud_from_file(self, file_path: str) -> None:
        """
        Loads cloud whose volume is to be computed from the specified file.

        Args:
            file_path (str): Path to the file containing point cloud of interest.
        """
        self.row = o3d.io.read_point_cloud(file_path)
        self._align_row_with_axis()

    def _align_row_with_axis(self) -> None:
        """
        Apply appropriate transformation aligning the row with X-axis 
        in order to make the computation of volume more convenient.
        """

        # Rotate row to go along x axis.
        points = np.array(self.row.points)
        x = np.expand_dims(points[:, 0], axis=1)
        y = np.expand_dims(points[:, 1], axis=1)

        reg = LinearRegression().fit(x, y)
        yaw = math.atan2(reg.coef_[0], 1)
        t1 = transformation_matrix(0, 0, -yaw, 0, 0, 0)
        self.row.transform(t1)

        x_min = np.min(np.array(self.row.points)[:, 0])
        z_min = np.min(np.array(self.row.points)[:, 2])
        t2 = transformation_matrix(0, 0, 0, -x_min, 0, -z_min)
        self.row.transform(t2)

        self.applied_transformation = t2 @ t1

        #mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
        #o3d.visualization.draw_geometries([mesh_frame, self.row])
    
    def _crop_row(self, start: float, end: float) -> o3d.geometry.PointCloud:
        """
        Crop specified segmet of the row.

        Args:
            start (float): Value specifying at what distance to start cropping.
            stop (float): Value specifying at what distance to stop cropping.
        """
        points = np.copy(np.array(self.row.points))
        colors = np.copy(np.array(self.row.colors))
        to_select = (points[:, 0] > start) & (points[:, 0] < end)
        points = points[to_select]
        if np.shape(colors)[0] > 0:
            colors = colors[to_select]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        return cloud

    def _alpha_shape_with_minimal_alpha(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        '''
        Applies alpha-shapes algorithm with minimal alpha value not introducing voids.

        Args:
            cloud (o3d.geometry.PointCloud): Point cloud from which triangle mesh is to be created.

        Returns:
            o3d.geometry.TriangleMesh: Resulting triangle mesh after application of alpha-shapes algorithm.
        '''
        hull, _ = cloud.compute_convex_hull()
        return hull
    
        alphas = list(np.linspace(0.1, 3.0, num=30))
        for alpha in alphas:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud, alpha)
            if mesh.is_watertight():
                print("ALHPA", alpha)
                mesh.compute_vertex_normals()
                return mesh
        hull, _ = cloud.compute_convex_hull()
        return hull

    def compute_volumes(self, measurements_gps: list[tuple[float, float]], datum: list[float], voxel_size: float=0.1, trunks_locations: np.ndarray=None) -> None:
        """
        Generates 3D models of trees in a row segment by segment using different methods and compute their volumes. Bounding box is applied for every tree to crop
        the point cloud, then voxel grid, alpha-shapes, bounding box and convex hull are computed and based on it volumes are estimated.

        Args:
            measurements_gps (list[tuple[float, float]]): List of GPS locations of measured trees for which volumes are to be collected.
            datum (list[float]): List [longitude, latitude, yaw] describing initial position of the robot.
            voxel_size (float): Size of the voxels used to compute voxel grid.
            trunks_locations (np.ndarray): Locations of trunks. Volume will be computed between consecutive trunks.
        """
        volumes_voxel = []
        volumes_mesh = []
        volumes_bounding_box = []
        volumes_convex_hull = []
        volumes_density_function = []
        line_sets = []
        voxel_grids = []
        meshes = []
        bounding_boxes = []
        convex_hulls = []
        widths = []

        if trunks_locations is not None:
            trunks_locations = np.hstack((trunks_locations, np.ones((trunks_locations.shape[0], 1))))
            print(self.applied_transformation, trunks_locations)
            trunks_locations = self.applied_transformation @ trunks_locations.T
            print(trunks_locations)
            meters = sorted(list(trunks_locations[0, :]))
            meters = list(range(0, 70, 1))
        else:
            meters = []
            p = Proj(proj='utm', zone=32, ellps='WGS84', preserve_units=False)
            x_0, y_0 = p(datum[1], datum[0])
            geometries = []
            for lat, long in measurements_gps:
                x, y = p(long, lat)
                x -= x_0
                y -= y_0
                coords = self.applied_transformation @ np.array([x, y, 0, 1]).T
                x = coords[0]
                y = coords[1]
                if y > 10.0 or y < -10.0:
                    continue
                meters.append(x - 0.5)
                meters.append(x + 0.5)
            meters = sorted(meters)

        zs = []
        for i in range(len(meters) - 1):
            # Crop point cloud
            cropped_cloud = self._crop_row(meters[i], meters[i + 1])
            _, indices = cropped_cloud.remove_radius_outlier(nb_points=300, radius=0.5)
            cropped_cloud = cropped_cloud.select_by_index(indices)
            color = np.random.uniform(0, 1, size=(3, ))
            cropped_cloud.colors = o3d.utility.Vector3dVector([color] * np.shape(np.array(cropped_cloud.points))[0])

            if np.shape(np.array(cropped_cloud.points))[0] == 0:
                if len(zs) > 0:
                    zs.append(zs[-1])
                else:
                    zs.append(0)
            else:
                zs.append(np.min(np.array(cropped_cloud.points)[:, 2]))

            # Compute voxel grid
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cropped_cloud, voxel_size)
            voxel_grids.append(voxel_grid)

            # Compute triangle mesh
            try:
                mesh = self._alpha_shape_with_minimal_alpha(cropped_cloud)
                meshes.append(mesh)
                volumes_mesh.append(mesh.get_volume())
            except:
                volumes_mesh.append(0)

            # Compute oriented bounding box
            bounding_box = cropped_cloud.get_axis_aligned_bounding_box()
            bounding_boxes.append(bounding_box)
            bounding_box.color = color

            # Compute convex hull
            try:
                hull, _ = cropped_cloud.compute_convex_hull()
                hull.paint_uniform_color(color)
                convex_hulls.append(hull)
                volumes_convex_hull.append(hull.get_volume())
            except:
                volumes_convex_hull.append(0)
            
            # Compute volume
            volume = len(voxel_grid.get_voxels()) * voxel_size ** 3
            volumes_voxel.append(volume)
            volumes_mesh.append(mesh.get_volume())
            volumes_bounding_box.append(bounding_box.volume())
            print(bounding_box.get_extent())
            widths.append(bounding_box.get_extent()[1])
            volumes_convex_hull.append(hull.get_volume())
            volumes_density_function.append(self._get_function_approximated_volume(start=meters[i], end=meters[i + 1], voxel_size=0.05, increment=0.25))

            corners = np.array([
                [meters[i], 10, -10],
                [meters[i + 1], 10, -10],
                [meters[i + 1], -10, -10],
                [meters[i], -10, -10],
                [meters[i], 10, 50],
                [meters[i + 1], 10, 50],
                [meters[i + 1], -10, 50],
                [meters[i], -10, 50],
            ])

            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0], 
                [4, 5], [5, 6], [6, 7], [7, 4], 
                [0, 4], [1, 5], [2, 6], [3, 7]
            ]
            colors = [[1, 0, 0] for _ in range(len(lines))]
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            line_sets.append(line_set)

        p = Proj(proj='utm', zone=32, ellps='WGS84', preserve_units=False)
        x_0, y_0 = p(datum[1], datum[0])
        geometries = [self.row]
        for lat, long in measurements_gps:
            x, y = p(long, lat)
            x -= x_0
            y -= y_0
            coords = self.applied_transformation @ np.array([x, y, 0, 1]).T
            x = coords[0]
            y = coords[1]

            z = 0
            for i in range(len(meters) - 1):
                if x > meters[i]:
                    z = zs[i]

            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
            sphere.paint_uniform_color([1, 0, 0])
            sphere.translate((x, y, z))
            geometries.append(sphere)

        visualization(bounding_boxes, widths, geometries)
        visualization(voxel_grids, volumes_voxel, geometries)
        visualization(voxel_grids, volumes_density_function, geometries)
        visualization(meshes, volumes_mesh, geometries)
        visualization(bounding_boxes, volumes_bounding_box, geometries)
        visualization(convex_hulls, volumes_convex_hull, geometries)
        
    def _get_volumes_along_row(self, voxel_size: float=0.1, increment: float=0.25) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        volumes_voxel = []
        volumes_mesh = []
        volumes_bounding_box = []
        volumes_convex_hull = []
        line_sets = []
        voxel_grids = []
        meshes = []
        bounding_boxes = []
        convex_hulls = []
        cropped_clouds = []

        points = np.copy(np.array(self.row.points))
        x_min = np.min(points[:, 0])
        x_max = np.max(points[:, 0])

        meters = np.arange(x_min, x_max, increment)
        for i in range(len(meters) - 1):
            # Crop point cloud
            cropped_cloud = self._crop_row(meters[i], meters[i + 1])
            # _, indices = cropped_cloud.remove_radius_outlier(nb_points=10, radius=0.2)
            # cropped_cloud = cropped_cloud.select_by_index(indices)

            color = np.random.uniform(0, 1, size=(3, ))
            cropped_cloud.colors = o3d.utility.Vector3dVector([color] * np.shape(np.array(cropped_cloud.colors))[0])
            cropped_clouds.append(cropped_cloud)

            # Compute voxel grid
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cropped_cloud, voxel_size)
            voxel_grids.append(voxel_grid)

            # Compute triangle mesh
            try:
                mesh = self._alpha_shape_with_minimal_alpha(cropped_cloud)
                meshes.append(mesh)
                volumes_mesh.append(mesh.get_volume())
            except:
                volumes_mesh.append(0)

            # Compute oriented bounding box
            bounding_box = cropped_cloud.get_axis_aligned_bounding_box()
            bounding_boxes.append(bounding_box)
            bounding_box.color = color

            # Compute convex hull
            try:
                hull, _ = cropped_cloud.compute_convex_hull()
                hull.paint_uniform_color(color)
                convex_hulls.append(hull)
                volumes_convex_hull.append(hull.get_volume())
            except:
                volumes_convex_hull.append(0)
            
            # Compute volume
            volume = len(voxel_grid.get_voxels()) * voxel_size ** 3
            volumes_voxel.append(volume)
            volumes_bounding_box.append(bounding_box.volume())

            corners = np.array([
                [meters[i], 10, -10],
                [meters[i + 1], 10, -10],
                [meters[i + 1], -10, -10],
                [meters[i], -10, -10],
                [meters[i], 10, 50],
                [meters[i + 1], 10, 50],
                [meters[i + 1], -10, 50],
                [meters[i], -10, 50],
            ])

            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0], 
                [4, 5], [5, 6], [6, 7], [7, 4], 
                [0, 4], [1, 5], [2, 6], [3, 7]
            ]
            colors = [[1, 0, 0] for _ in range(len(lines))]
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            line_sets.append(line_set)

        # Visualize reconstructions
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.row, voxel_size)
        # o3d.visualization.draw_geometries([voxel_grid, mesh_frame] + line_sets)
        # o3d.visualization.draw_geometries(cropped_clouds)
        # o3d.visualization.draw_geometries(voxel_grids)
        # o3d.visualization.draw_geometries(meshes, mesh_show_wireframe=True)
        # o3d.visualization.draw_geometries(bounding_boxes, mesh_show_wireframe=True)
        # o3d.visualization.draw_geometries(convex_hulls, mesh_show_wireframe=True)

        return meters[1:], volumes_voxel, volumes_mesh, volumes_bounding_box, volumes_convex_hull

    def generate_volume_distribution_plots(self, result_dir: str, voxel_size: float=0.1, increment: float=0.25) -> None:
        """
        Generates plots representing volumes of a row segment by segment using different methods. Bounding box is applied every givien increment to crop
        the point cloud, then voxel grid, alpha-shapes, bounding box and convex hull are computed and based on it volumes are estimated.

        Args:
            result_dir (str): Name of a directory where generated plots shall be saved.
            voxel_size (float): Size of the voxels used to compute voxel grid.
            increment (float): Value specifying sizes of segments of the row. Volume of each segment is to be computed.
            trunks_locations (np.ndarray): Locations of trunks. Volume will be computed between consecutive trunks.
        """
        meters, volumes_voxel, volumes_mesh, volumes_bounding_box, volumes_convex_hull = self._get_volumes_along_row(voxel_size=voxel_size, increment=increment)

        path = os.path.join(os.getcwd(), 'data', '3d_reconstruction', result_dir)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        # Save plots with volumes
        plt.scatter(meters, volumes_voxel)
        plt.xlabel('Distance along the row (meters)')
        plt.ylabel('Volume (meters ^ 3)')
        title = 'Volumes computed at the increment of {}m with voxel size = {}'.format(increment, voxel_size)
        plt.title(title)
        plt.savefig(os.path.join(path, title + '.png'))
        #plt.show()
        plt.clf()

        plt.hist(volumes_voxel)
        plt.savefig(os.path.join(path, "volumes_voxel_grid_histogram.png"))
        plt.xlabel('Volume (meters ^ 3)')
        plt.ylabel('Number')
        title = 'Distribution of volumes computed at the increment of {}m with voxel size = {}'.format(increment, voxel_size)
        plt.title(title)
        plt.savefig(os.path.join(path, title + '.png'))
        #plt.show()
        plt.clf()

        plt.scatter(meters, volumes_mesh)
        plt.xlabel('Distance along the row (meters)')
        plt.ylabel('Volume (meters ^ 3)')
        title = 'Volumes computed at the increment of {}m with alpha-shapes'.format(increment)
        plt.title(title)
        plt.savefig(os.path.join(path, title + '.png'))
        #plt.show()
        plt.clf()

        plt.hist(volumes_mesh)
        plt.savefig(os.path.join(path, "volumes_alpha_shapes_histogram.png"))
        plt.xlabel('Volume (meters ^ 3)')
        plt.ylabel('Number')
        title = 'Distribution of volumes computed at the increment of {}m with alpha-shapes'.format(increment)
        plt.title(title)
        plt.savefig(os.path.join(path, title + '.png'))
        #plt.show()
        plt.clf()

        plt.scatter(meters, volumes_bounding_box)
        plt.xlabel('Distance along the row (meters)')
        plt.ylabel('Volume (meters ^ 3)')
        title = 'Volumes computed at the increment of {}m with bounding boxes'.format(increment)
        plt.title(title)
        plt.savefig(os.path.join(path, title + '.png'))
        #plt.show()
        plt.clf()

        plt.hist(volumes_bounding_box)
        plt.savefig(os.path.join(path, "volumes_oriented_bounding_box_histogram.png"))
        plt.xlabel('Volume (meters ^ 3)')
        plt.ylabel('Number')
        title = 'Distribution of volumes computed at the increment of {}m with bounding boxes'.format(increment)
        plt.title(title)
        plt.savefig(os.path.join(path, title + '.png'))
        #plt.show()
        plt.clf()

        plt.scatter(meters, volumes_convex_hull)
        plt.xlabel('Distance along the row (meters)')
        plt.ylabel('Volume (meters ^ 3)')
        title = 'Volumes computed at the increment of {}m with convex hull'.format(increment)
        plt.title(title)
        plt.savefig(os.path.join(path, title + '.png'))
        #plt.show()
        plt.clf()

        plt.hist(volumes_convex_hull)
        plt.savefig(os.path.join(path, "volumes_oriented_bounding_box_histogram.png"))
        plt.xlabel('Volume (meters ^ 3)')
        plt.ylabel('Number')
        title = 'Distribution of volumes computed at the increment of {}m with convex hull'.format(increment)
        plt.title(title)
        plt.savefig(os.path.join(path, title + '.png'))
        #plt.show()
        plt.clf()
        
    def _get_function_approximated_volume(self, start: float, end: float, voxel_size: float=0.1, increment: float=0.25) -> float:
        degree = 5
        
        if self.volume_density_function is None:
            meters, volumes_voxel, volumes_mesh, volumes_bounding_box, volumes_convex_hull = self._get_volumes_along_row(voxel_size=voxel_size, increment=increment)
            meters = np.array(meters)
            
            x = np.zeros(shape=(np.shape(meters)[0], degree))
            for i in range(1, degree + 1):
                x[:, i - 1] = 1 / i * (meters ** i - (meters - increment) ** i)

            self.volume_density_function = LinearRegression(fit_intercept=False).fit(x, np.array(volumes_voxel))
            print(self.volume_density_function.intercept_, self.volume_density_function.coef_)

            x_poly = np.zeros(shape=(np.shape(meters)[0], degree))
            for i in range(degree):
                x_poly[:, i] = meters ** i

            plt.scatter(meters, volumes_voxel)
            plt.plot(meters, self.volume_density_function.predict(x))
            plt.plot(meters, self.volume_density_function.predict(x_poly))
            plt.show()

        x = np.zeros(shape=(1, degree))
        for i in range(1, degree + 1):
            x[:, i - 1] = 1 / i * (end ** i - start ** i)

        print('RESULT', self.volume_density_function.predict(x))
        
        return self.volume_density_function.predict(x)[0]



def visualization(trees: list[o3d.geometry.Geometry], volumes: list[float], geometries: list[o3d.geometry.Geometry]) -> None:
    """
    Creates visualised 3D reconstruction with estimated volumes for each particular tree.

    Args:
        trees (list[o3d.geometry.Geometry]): List of geometries representing particular trees.
        volumes (list[float]): List of corresponding estimated volumes.
        geometries (list[o3d.geometry.Geometry]): List of geometries indicating GPS positions of measured trees.
    """
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    for i, tree in enumerate(trees):
        bounds = tree.get_axis_aligned_bounding_box()
        print(bounds.get_center())
        vis.add_geometry("tree_{}".format(i), tree)
        vis.add_3d_label(np.array(bounds.get_center()) + np.array([0, 0, 1.5]), "{}".format(round(volumes[i], 2)))

    for i, geometry in enumerate(geometries):
        vis.add_geometry("geometry_{}".format(i), geometry)

    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()

