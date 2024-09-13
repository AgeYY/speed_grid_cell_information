'''
compute grid cell tuning by averaging firing rate. No speed filtering.
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from grid_cell.util import select_arr_by_t
import grid_cell.tuning as gct
from scipy.ndimage import gaussian_filter # for smoothing grid cell tuning
import os
from global_setting import *

########## Hyperparameters ##########
########## Hyperparameters ##########
mouse_name, module, day = 'R', '2', 'day1'
session = 'open_field_1'
file_name = 'preprocessed_data_{}_{}_{}_{}.npz'.format(mouse_name, module, day, session)
data_path = os.path.join(DATAROOT, 'preprocessed/', file_name)

n_bin = 50
tuning_smooth_sigma = 2 # smoothing in the spatial domain
pca_pp_n_components = 10 # number of components for pca preprocessing

########## Main ##########
data = np.load(data_path, allow_pickle=True)

fire_rate_matrix = data['fire_rate_matrix']
x, y, t, dt = data['x'], data['y'], data['t'], data['dt']

pca = PCA(n_components=pca_pp_n_components)
fire_rate_matrix = pca.fit_transform(fire_rate_matrix) # row: time, col: pcaed features

for cell_id in range(pca_pp_n_components):
    tuning = gct.spatial_bin_fire_rate(fire_rate_matrix[:, cell_id], x, y, n_bin=n_bin)
    tuning = np.nan_to_num(tuning)
    tuning = gaussian_filter(tuning, sigma=tuning_smooth_sigma)

    plt.figure()
    plt.imshow(tuning)
    plt.show()
