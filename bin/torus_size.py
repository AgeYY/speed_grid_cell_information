# run fit_gkr_models.py first
import os
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
from grid_cell.manifold_fitter import label_mesh
import grid_cell.ploter as ploter
import grid_cell.torus_geometry as tg
from grid_cell.util import remove_outliers, clean_and_mean_se, gram_schmidt, project_to_orthogonal_subspace, angle_two_vec
from global_setting import *

#################### Hyperparameters ####################
dataset_name_list = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
eg_dn = dataset_name_list[0]
mesh_size = 200
# n_pca = None
n_pca = 6
metric_type = 'ALL'
x_bound, y_bound, speed_bound = [-0.75, 0.75], [-0.75, 0.75], [0.05, 0.45]

for dn in dataset_name_list:
    print(f'Analyzing {dn}...')
    analyzer = tg.TorusShapeAnalyzer(mesh_size, x_bound, y_bound, speed_bound)
    model_path = f'gkr_models_{dn}_pca{n_pca}.hkl'
    result = analyzer.analyze(model_path, metric_type=metric_type, iid_mode=False)
    tg.save_torus_shape_results(DATAROOT,  f'torus_shape_speed_{dn}_pca{n_pca}_iid{False}.hkl', result)

    print(f'Analyzing i.i.d. {dn}...')
    analyzer = tg.TorusShapeAnalyzer(mesh_size, x_bound, y_bound, speed_bound)
    model_path = f'gkr_models_{dn}_pca{n_pca}.hkl'
    result = analyzer.analyze(model_path, metric_type=metric_type, iid_mode=True)
    tg.save_torus_shape_results(DATAROOT,  f'torus_shape_speed_{dn}_pca{n_pca}_iid{True}.hkl', result)

    analyzer = tg.TorusShapeAnalyzer(mesh_size, x_bound, y_bound, speed_bound)
    print(f'Analyzing shuffle {dn}...')
    model_path = f'shuffle_gkr_models_{dn}_pca{n_pca}.hkl'
    result = analyzer.analyze(model_path, metric_type=metric_type, iid_mode=False)
    tg.save_torus_shape_results(DATAROOT,  f'shuffle_torus_shape_speed_{dn}_pca{n_pca}_iid{False}.hkl', result)

torus_shape_speed = hkl.load(os.path.join(DATAROOT, f'torus_shape_speed_{eg_dn}_pca{n_pca}_iid{False}.hkl'))
speed_list = torus_shape_speed['speed']
lattice_area_list = torus_shape_speed['lattice_area']
radius_list = torus_shape_speed['radius']
center_dist_list = torus_shape_speed['center_dist']

fig, ax = plt.subplots(2, 1, figsize=(3, 5))
ax[0].scatter(speed_list.flatten(), lattice_area_list.flatten())
ax[0].set_ylabel('Lattice Area ' + r'($a.u. / cm^2$)')

ax[1].scatter(speed_list.flatten(), radius_list.flatten())
ax[1].set_ylabel('Radius ' + r'($a.u.$)')
ax[1].set_xlabel('Speed (cm/s)')
fig.savefig(os.path.join(FIGROOT, 'torus_size_speed_pca{n_pca}.svg'))

plt.show()
