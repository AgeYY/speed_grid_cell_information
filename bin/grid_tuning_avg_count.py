'''
get example grid cell tuning by averaging spike count
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
speed_thre = 2.5 # cm/s, same as the paper
smooth_sigma = 2.75 # smoothing in the spatial domain

########## Main ##########
qgrid_data = np.load(qgrid_path, allow_pickle=True)

# load data and select session
cell0 = qgrid_data[cell_module].item()[cell_id]
cell0 = select_arr_by_t(cell0, cell0, session=session, file_name=file_name)
x, y, t = (select_arr_by_t(qgrid_data[key], np.array(qgrid_data['t']), session=session, file_name=file_name) for key in ['x', 'y', 't'])

dt = t[1] - t[0] # before filtered by speed, store the time spacing

# filter by speed
speed = gct.compute_speed(x, y, t) # speed at each time point
speed = speed * 100 # convert to cm/s
cell0 = gct.filter_spikes_by_speed(cell0, t, speed, speed_thre) # in practice, filtering by speed make not much difference, only a little spikes will be filtered out
x, y, t = gct.filter_xyt_by_speed(x, speed, speed_thre), gct.filter_xyt_by_speed(y, speed, speed_thre), gct.filter_xyt_by_speed(t, speed, speed_thre)

stay_time = gct.compute_bin_times(x, y, dt, n_bin) # amount of time staying in each bin
unvisited_mask = stay_time > 0.01

# obtained number of spikes in each bin
spike_count = gct.spatial_bin_spikes_count(cell0, t, x, y, n_bin)

# compute the firing rate
spike_rate = spike_count / stay_time
spike_rate = np.nan_to_num(spike_rate)

spike_rate = gaussian_filter(spike_rate, sigma=smooth_sigma) # smooth the firing rate
unvisited_mask = gaussian_filter(unvisited_mask.astype(float), sigma=smooth_sigma) # smooth the firing rate

spike_rate = spike_rate / unvisited_mask

plt.figure()
plt.imshow(spike_rate)
plt.show()
