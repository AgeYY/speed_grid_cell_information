'''
compute grid cell tuning by averaging firing rate. No speed filtering.
'''
import numpy as np
import matplotlib.pyplot as plt
from grid_cell.util import select_arr_by_t
import grid_cell.tuning as gct
from scipy.ndimage import gaussian_filter # for smoothing grid cell tuning
import os
from global_setting import *

########## Hyperparameters ##########
file_name = 'rat_r_day1_grid_modules_1_2_3.npz'
qgrid_path = os.path.join(EXP_DATAROOT, file_name)
cell_module = 'spikes_mod2'
session = 'open_field_1'

cell_id = 10 # tuning of this cell
n_bin = 50
fr_smooth_sigma = 2 # smoothing in the temporal domain
tuning_smooth_sigma = 2 # smoothing in the spatial domain

########## Main ##########
qgrid_data = np.load(qgrid_path, allow_pickle=True)

# load data and select session
cell_module = qgrid_data[cell_module].item()
cell_module = {key: select_arr_by_t(value, value, session=session, file_name=file_name) for key, value in cell_module.items()}
x, y, t = (select_arr_by_t(qgrid_data[key], qgrid_data['t'], session=session, file_name=file_name) for key in ['x', 'y', 't'])

fire_rate_matrix = gct.compute_all_firing_rates(cell_module, t, fr_smooth_sigma)

tuning = gct.spatial_bin_fire_rate(fire_rate_matrix[:, cell_id], x, y, n_bin=n_bin)
tuning = np.nan_to_num(tuning)
tuning = gaussian_filter(tuning, sigma=tuning_smooth_sigma)

plt.figure()
plt.imshow(tuning)
plt.show()
