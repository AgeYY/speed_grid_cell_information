'''
get example grid cell tuning by averaging spike count
'''
import numpy as np
import matplotlib.pyplot as plt
from grid_cell.util import select_arr_by_t, remove_ax_frame
import grid_cell.tuning as gct
from scipy.ndimage import gaussian_filter # for smoothing grid cell tuning
from grid_cell.util import get_data_filename_by_keywords
from scipy.ndimage import rotate
from numpy.ma import corrcoef
import os
from global_setting import *

########## Hyperparameters ##########
# file_name = 'rat_r_day1_grid_modules_1_2_3.npz'
# qgrid_path = os.path.join(EXP_DATAROOT, file_name)
# cell_module = 'spikes_mod2'
# session = 'open_field_1'

mouse_name, module, day = 'R', '2', 'day1'
file_name = get_data_filename_by_keywords(mouse_name.lower(), day)
cell_module = 'spikes_mod{}'.format(module)
session = 'open_field_1'

qgrid_path = os.path.join(EXP_DATAROOT, file_name)

# is_conj_file_name = 'is_conjunctive_all.npz'
# conj_path = os.path.join(EXP_DATAROOT, is_conj_file_name)

n_bin = 50
speed_thre = 2.5 # cm/s, same as the paper
smooth_sigma = 2.75 # smoothing in the spatial domain
x_bound, y_bound = None, None # boundary of the arena

auto_corr_padding = 10

########## Main ##########
qgrid_data = np.load(qgrid_path, allow_pickle=True)
# is_conj = np.load(conj_path, allow_pickle=True)
# is_pure = np.logical_not(is_conj['is_conj_{}_{}'.format(mouse_name + module, day)])

# load data and select session
cell_module = qgrid_data[cell_module].item()
# cell_module = {key: cell0 for key, cell0 in cell_module.items() if is_pure[key]} # select pure grid cells
cell_module = {key: select_arr_by_t(value, value, session=session, file_name=file_name) for key, value in cell_module.items()}
x, y, t = (select_arr_by_t(qgrid_data[key], qgrid_data['t'], session=session, file_name=file_name) for key in ['x', 'y', 't'])

dt = t[1] - t[0] # before filtered by speed, store the time spacing

# filter by speed
speed = gct.compute_speed(x, y, t) # speed at each time point
speed = speed * 100 # convert to cm/s
cell_module = {key: gct.filter_spikes_by_speed(cell0, t, speed, speed_thre) for key, cell0 in cell_module.items()} # in practice, filtering by speed make not much difference, only a little spikes will be filtered out
x, y, t = gct.filter_xyt_by_speed(x, speed, speed_thre), gct.filter_xyt_by_speed(y, speed, speed_thre), gct.filter_xyt_by_speed(t, speed, speed_thre)

stay_time = gct.compute_bin_times(x, y, dt, n_bin, x_bound=x_bound, y_bound=y_bound) # amount of time staying in each bin
unvisited_mask = stay_time > 0.01
unvisited_mask = gaussian_filter(unvisited_mask.astype(float), sigma=smooth_sigma) # smooth the firing rate

spike_rate_dict = {}
for key, cell0 in cell_module.items():
    # obtained number of spikes in each bin
    spike_count = gct.spatial_bin_spikes_count(cell0, t, x, y, n_bin, x_bound=x_bound, y_bound=y_bound)

    # compute the firing rate
    spike_rate = spike_count / stay_time
    spike_rate = np.nan_to_num(spike_rate)

    spike_rate = gaussian_filter(spike_rate, sigma=smooth_sigma) # smooth the firing rate

    spike_rate = spike_rate / unvisited_mask

    spike_rate_dict[key] = spike_rate

n_cell = len(spike_rate_dict)

# compute gridness
gridness_score = np.zeros(n_cell)
for key, val in spike_rate_dict.items():
    gridness_score[key] = gct.gridness(val, padding=auto_corr_padding)
score_rank = np.argsort(-gridness_score).argsort() # sort by gridness score
# print(np.where(score_rank == 140))
# print(np.argmin(gridness_score))
# print(np.argmax(score_rank))
# exit()
# print(score_rank)

n_row = np.ceil(np.sqrt(n_cell)).astype(int)
fig, axes = plt.subplots(n_row, n_row, figsize=(10, 10))
ax_flat = axes.flatten()
for key, spike_rate in spike_rate_dict.items():
    # print(score_rank[key])
    ax_flat[score_rank[key]].imshow(spike_rate)
    # ax_flat[score_rank[key]].set_title('{}'.format(key))


for ax in ax_flat:
    remove_ax_frame(ax)

plt.tight_layout()
plt.show()

rate_map_path = os.path.join(DATAROOT, 'rate_map.npz')
np.savez(rate_map_path, spike_rate_dict=spike_rate_dict, gridness_score=gridness_score)
