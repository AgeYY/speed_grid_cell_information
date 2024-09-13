'''
reproducing the torus manifold by umap, from the Gardner et al. 2022.
'''
import numpy as np
import matplotlib.pyplot as plt
from grid_cell.util import select_arr_by_t, apply_each_col
from sklearn.decomposition import PCA
from umap import UMAP
import grid_cell.tuning as gct
from scipy.ndimage import gaussian_filter # for smoothing grid cell tuning
import os
from global_setting import *

########## Hyperparameters ##########
file_name = 'rat_r_day1_grid_modules_1_2_3.npz'
cell_module = 'spikes_mod2'
mouse_name, day = 'R2', 'day1'

file_name = 'rat_r_day2_grid_modules_1_2_3.npz'
cell_module = 'spikes_mod3'
mouse_name, day = 'R3', 'day2'

session = 'open_field_1'
qgrid_path = os.path.join(EXP_DATAROOT, file_name)

is_conj_file_name = 'is_conjunctive_all.npz'
conj_path = os.path.join(EXP_DATAROOT, is_conj_file_name)

spatial_n_bin = 50
speed_thre = 2.5 # cm/s
fr_smooth_sigma = 5 # smoothing in the temporal domain
downsample_rate = 25 # downsample time points
pca_pp_n_components = 6 # number of components for pca preprocessing
bin_fire_rate = False # whether to bin the firing rate
visualize_method = 'umap'

bin_fire_rate_sigma = 1 # smoothing in the spatial domain. Only used when bin_fire_rate is True

# umap parameters for dimension reduction. This is identical to the paper
umap_components = 3
umap_n_neighbors = 1000
umap_min_dist = 0.6
umap_metric = 'cosine'
umap_init = 'spectral'

########## Main ##########
qgrid_data = np.load(qgrid_path, allow_pickle=True)
is_conj = np.load(conj_path, allow_pickle=True)

# load data and select session
cell_module = qgrid_data[cell_module].item()
cell_module = {key: select_arr_by_t(value, value, session=session, file_name=file_name) for key, value in cell_module.items()}
x, y, t = (select_arr_by_t(qgrid_data[key], qgrid_data['t'], session=session, file_name=file_name) for key in ['x', 'y', 't'])

# compute firing rate
fire_rate_matrix = gct.compute_all_firing_rates(cell_module, t, fr_smooth_sigma)

# filter by speed
speed = gct.compute_speed(x, y, t) # speed at each time point
speed = speed * 100 # convert to cm/s
# plt.figure()
# plt.hist(speed, bins=100)
# plt.show()
# exit()
fire_rate_matrix = gct.filter_xyt_by_speed(fire_rate_matrix, speed, speed_thre)
x, y, t = gct.filter_xyt_by_speed(x, speed, speed_thre), gct.filter_xyt_by_speed(y, speed, speed_thre), gct.filter_xyt_by_speed(t, speed, speed_thre)

# preprocessing: bin, downsample, pure grid cells, pca
if bin_fire_rate:
    n_cell = fire_rate_matrix.shape[1]
    fire_rate_matrix = apply_each_col(fire_rate_matrix, gct.spatial_bin_fire_rate, x=x, y=y, n_bin=spatial_n_bin)
    fire_rate_matrix = np.nan_to_num(fire_rate_matrix)
    fire_rate_matrix = gaussian_filter(fire_rate_matrix, sigma=bin_fire_rate_sigma, axes=[0, 1])
    fire_rate_matrix = fire_rate_matrix.reshape(-1, n_cell)
else:
    fire_rate_matrix = fire_rate_matrix[::downsample_rate] # downsample time points

# fire_rate_matrix = fire_rate_matrix[:, np.logical_not(is_conj['is_conj_{}_{}'.format(mouse_name, day)])] # select non-conjunctive cells here

# pca preprocessing pca
pca = PCA(n_components=pca_pp_n_components)
feamap = pca.fit_transform(fire_rate_matrix) # row: time, col: pcaed features

# umap visualization
print('{} visualization...'.format(visualize_method))
if visualize_method == 'umap':
    fitter = UMAP(n_components=umap_components, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, metric=umap_metric, init=umap_init)
elif visualize_method == 'pca':
    fitter = PCA(n_components=umap_components)
feamap_umap = fitter.fit_transform(feamap)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(feamap_umap[:, 0], feamap_umap[:, 1], feamap_umap[:, 2], c=feamap[:, 0], s=1)
plt.show()
