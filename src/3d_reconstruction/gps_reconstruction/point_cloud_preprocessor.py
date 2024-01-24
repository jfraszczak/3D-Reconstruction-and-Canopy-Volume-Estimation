import open3d as o3d
import numpy as np
import math
from sklearn.linear_model import RANSACRegressor


def remove_outliers(cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Applies outliers removal.

    Args:
        cloud (o3d.geometry.PointCloud): Point cloud to be preprocessed.

    Returns:
        o3d.geometry.PointCloud: Preprocessed point cloud.
    
    """
    _, indices = cloud.remove_radius_outlier(nb_points=5, radius=0.3)
    cloud = cloud.select_by_index(indices)

    return cloud

def remove_distant_points(cloud: o3d.geometry.PointCloud, threshold: float=5.0) -> o3d.geometry.PointCloud:
    """
    Remove distant points along the x axis of the row.

    Args:
        cloud (o3d.geometry.PointCloud): Point cloud from which points are to be removed.
        threshold (float): Value specifying up tho which distance points should be remained.

    Returns:
        o3d.geometry.PointCloud: Point cloud after removal of points.
    """
    points = np.array(cloud.points)
    filtered_points = np.where((points[:, 0] >= -threshold) & (points[:, 0] <= threshold))[0].tolist()
    cloud = cloud.select_by_index(filtered_points)
    
    return cloud

def _fit_ground_plane(cloud: o3d.geometry.PointCloud) -> RANSACRegressor:
    """
    Fit plane to the points.

    Args:
        cloud (o3d.geometry.PointCloud): Point cloud containing the ground.

    Returns:
        RANSACRegressor: RANSACRegressor after being fitted to the points representing the ground.
    """
    points = np.array(cloud.points)
    points_below_ground = points[points[:, 2] < 0]

    if np.shape(points_below_ground)[0] > 0:
        ransac = RANSACRegressor(random_state=0).fit(points_below_ground[:, 0:2], points_below_ground[:, 2])
        return ransac
    else:
        return None
    
def _remove_points_below_ground(cloud: o3d.geometry.PointCloud, ransac: RANSACRegressor, threshold: float=0.0) -> o3d.geometry.PointCloud:
    """
    Remove points below the ground + threshold.

    Args:
        cloud (o3d.geometry.PointCloud): Point cloud from which points are to be removed.
        ransac (RANSACRegressor): RANSACRegressor after being fitted to the points representing the ground.
        threshold (float): Value specifying up to which height above the ground to remove points.

    Returns:
        o3d.geometry.PointCloud: Point cloud after removal of points below given threshold.
    """
    a, b = ransac.estimator_.coef_
    c = -1
    d = ransac.estimator_.intercept_

    points = np.array(cloud.points)
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    plane = np.expand_dims(np.array([a, b, c, d]), axis=1)
    dists = np.abs(points @ plane) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
    filtered_points = np.where((ransac.predict(points[:, 0:2]) <= points[:, 2]) & (dists > threshold).flatten())[0].tolist()
    cloud = cloud.select_by_index(filtered_points)
    return cloud

def _get_plane_as_triangle_mesh(ransac: RANSACRegressor) -> o3d.geometry.TriangleMesh:
    """
    Returns triangle mesh representing given plane.

    Args:
        ransac (RANSACRegressor): RANSACRegressor after being fitted to the points representing the ground.

    Returns:
        o3d.geometry.TriangleMesh: Triangle mesh representing given plane.
    """
    corners = np.array([[-1, -1],
                        [1, -1],
                        [1, 1],
                        [-1, 1]]) * 5.0
    
    z = ransac.predict(corners)
    z = np.expand_dims(z, axis=1)
    corners = np.concatenate((corners, z), axis=1)

    plane = o3d.geometry.TriangleMesh()
    plane.vertices = o3d.utility.Vector3dVector(corners)
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    plane.triangles = o3d.utility.Vector3iVector(triangles)
    plane.paint_uniform_color([0, 1, 0])

    return plane

def remove_points_below_ground(cloud: o3d.geometry.PointCloud, threshold: float=0.0, test_mode: bool=False) -> o3d.geometry.PointCloud:
    """
    Remove points below the ground + threshold.

    Args:
        cloud (o3d.geometry.PointCloud): Point cloud from which points are to be removed.
        threshold (float): Value specifying up to which height above the ground to remove points.
        test_mode (bool): Specifies whether to visualize intermediate results.

    Returns:
        o3d.geometry.PointCloud: Point cloud after removal of points below given threshold.
    """
    ransac = _fit_ground_plane(cloud)
    if ransac is not None:
        cloud_preprocessed = _remove_points_below_ground(cloud, ransac, threshold=threshold)

    if test_mode and ransac is not None:
        plane = _get_plane_as_triangle_mesh(ransac)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)
        o3d.visualization.draw_geometries([plane, cloud, mesh_frame], mesh_show_back_face=True)
        o3d.visualization.draw_geometries([plane, cloud_preprocessed, mesh_frame], mesh_show_back_face=True)  

    return cloud_preprocessed
