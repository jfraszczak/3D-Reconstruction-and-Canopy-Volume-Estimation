from matplotlib import pyplot as plt
import numpy as np

path = 'C:/Users/jakub/OneDrive/Documents/Master Thesis/canopy-volume-estimation/data/3d_reconstruction/reconstructions/2022/row_1/art_slam_reconstruction/rgbd/gps.txt'
file = open(path, "r")

xs = []
ys = []
for line in file:
    id, x, y, _ = line.split(' ')
    xs.append(float(x))
    ys.append(float(y))

xs = np.array(xs)
ys = np.array(ys)

xs -= xs[0]
ys -= ys[0]

plt.scatter(xs, ys)
plt.show()