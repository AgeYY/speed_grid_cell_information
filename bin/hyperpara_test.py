import os
import numpy as np
from grid_cell.gkr import GKR_Fitter
from grid_cell.manifold_fitter import label_mesh
import hickle as hkl
import grid_cell.torus_geometry as tg
import matplotlib.pyplot as plt
from global_setting import *

mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'
fr_smooth_sigma = 1
downsample_rate_gen = 5
downsample_rate_fit = 10
pca_component = 6
n_epoch = 20
adaptive_fr_sigma = False
n_inducing = 500

gkr_file_name = 'gkr_{}_{}_{}_{}_pca{}_downsample{}_{}_{}_{}_{}.hkl'.format(mouse_name, module, day, session, str(pca_component), str(downsample_rate_fit), fr_smooth_sigma, downsample_rate_gen, adaptive_fr_sigma, n_inducing)
model_path = os.path.join(DATAROOT, gkr_file_name)
print(model_path)

model = hkl.load(model_path)

mesh_size = 100
x_bound, y_bound, speed_bound = [-0.75, 0.75], [-0.75, 0.75], [0.05, 0.60]
# get random mesh of points
query_mesh_space = label_mesh([x_bound, y_bound], mesh_size=mesh_size, grid=False, random=True)

speed_list = np.arange(speed_bound[0], speed_bound[1], 0.05)
radius_list = []
for speed_value in speed_list:
    query_mesh = np.concatenate([query_mesh_space, np.ones([mesh_size, 1]) * speed_value], axis=1)
    feamap, _ = model.predict(query_mesh)

    # radius
    radius = tg.torus_avg_radius(feamap)
    radius_list.append(radius)

speed_list = np.array(speed_list) * 100 # convert to cm
radius_list = np.array(radius_list)
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(speed_list, radius_list, marker='o')
ax.set_xlabel('Speed (cm/s)')
ax.set_ylabel('Radius (cm)')
plt.show()
