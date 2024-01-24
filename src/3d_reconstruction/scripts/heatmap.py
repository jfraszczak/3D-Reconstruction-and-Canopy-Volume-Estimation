import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import open3d as o3d

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
cloud = o3d.io.read_point_cloud('C:/Users/jakub/OneDrive/Documents/Master Thesis/canopy-volume-estimation/data/3d_reconstruction/reconstructions/2022/row_1/art_slam_reconstruction/segmented_reconstruction/reconstructed_segmented_cloud.ply')
o3d.visualization.draw_geometries([mesh_frame, cloud])

# for i in range(10):
#     data = np.zeros((100, 100))
#     data[:, i * 10:(i + 1) * 10] = 100
#     ax = sns.heatmap(data, cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True))
#     ax.tick_params(left = False, right = False , labelleft = False , 
#                 labelbottom = False, bottom = False) 
#     ax.set_xlabel('n')
#     ax.set_ylabel('n')
#     plt.show()