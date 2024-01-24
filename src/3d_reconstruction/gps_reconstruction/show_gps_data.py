from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import math
import open3d as o3d
from .rgbd_reconstructor import RGBDReconstructor

def verify_gps_data(reconstructor: RGBDReconstructor):
    timestamps = [coord[0] for coord in reconstructor.gps_data]
    coords = reconstructor._gps_coords_to_numpy()
    x = np.expand_dims(coords[:, 0], axis=1)
    y = np.expand_dims(coords[:, 1], axis=1)
    z = np.expand_dims(coords[:, 2], axis=1)

    reg = LinearRegression().fit(x, y)
    res = np.abs(reg.predict(x) - y) / math.sqrt(reg.coef_[0] ** 2 + 1)
    d = np.mean(res)
    plt.hist(res)
    plt.show()

    alpha = math.atan2(reg.coef_[0] * (x[-1, 0] - x[0, 0]), x[-1, 0] - x[0, 0])
    c = d / math.cos(alpha)
    print(c)
    
    print(np.shape(x))
    plt.scatter(x, y)
    plt.plot(x, reg.predict(x))
    plt.plot(x, reg.predict(x) + c)
    plt.plot(x, reg.predict(x) - c)
    plt.show()

    # Plot x and y
    plt.scatter(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    # Plot timestamps and z
    plt.scatter(timestamps, z)
    plt.xlabel('timestamp')
    plt.ylabel('Z')
    plt.show()

    # Plot timestamps and x
    plt.scatter(timestamps, x)
    plt.xlabel('timestamp')
    plt.ylabel('X')
    plt.show()

    # Plot timestamps and y
    plt.scatter(timestamps, y)
    plt.xlabel('timestamp')
    plt.ylabel('Y')
    plt.show()

def show_point_clouds(reconstructor: RGBDReconstructor):
    for _, cloud in reconstructor.clouds:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)
        o3d.visualization.draw_geometries([mesh_frame, cloud])

reconstructor = RGBDReconstructor('d435i_link')
verify_gps_data(reconstructor)
