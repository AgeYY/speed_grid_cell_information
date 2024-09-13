import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os
from global_setting import *

#################### Hyperparameters ####################
mouse_name = 'S'
day = 'day1'
module = '1'
session = 'open_field_1'
# cell_id_list = [38, 7, 157, 12, 20, 145] # These are the correct ids for cell shown in the figure, when gridness = -inf. Please go to generate_speed_neuron_info.py to get the correct ids
# cell_id_list = [25, 5, 106, 9, 15, 98] # new id in after gridness > 0.3
cell_id_list = [25, 5, 30, 9, 15, 20] # new id in after gridness > 0.3
# cell_id_list = [38, 38, 38, 38, 48, 48] # these placeholder
cell_title_list = ['linear growth', 'flat then grow',  'linear decay', 'decay then flat', 'u shape', 'weak speed \n modulation']

speed_temporal_file_name = 'speed_temporal_{}_{}_{}_{}.npz'.format(mouse_name, module, day, session)
speed_temporal_path = os.path.join(DATAROOT, speed_temporal_file_name)
#################### Main ####################
data = np.load(speed_temporal_path, allow_pickle=True)
speed_list, mfr = data['speed_bins'][1:], data['mean_firing_rate']

# Placeholder initializations
r_values = np.zeros(mfr.shape[1])
slopes = np.zeros_like(r_values)
intercepts = np.zeros_like(r_values)

# Loop for each cell, get the slope, intercept and r value
for i in range(mfr.shape[1]):
    slope, intercept, r_value, p_value, std_err = stats.linregress(speed_list, mfr[:,i])

    r_values[i] = r_value
    slopes[i] = slope
    intercepts[i] = intercept
r_values = np.round(r_values, 2)
slopes = np.round(slopes, 2)
intercepts = np.round(intercepts, 2)

# plot mean fr as a function of speed
fig = plt.figure(figsize=(12,4), constrained_layout=True)
spec = gridspec.GridSpec(ncols=6, nrows=2, figure=fig)

for i in range(2):
  for j in range(0, 6, 2):
    ax1 = fig.add_subplot(spec[i, j])
    ax2 = fig.add_subplot(spec[i, j + 1])

    cid = cell_id_list[i+j]
    ax1.imshow(data['rate_map'][:, :, cid])
    ax1.axis('off')
    ax1.set_title(cell_title_list[i+j], fontsize=13)

    x, y = data['speed_bins'][1:], data['mean_firing_rate'][:, cid]
    ax2.scatter(x, y, s=3)


    xmf, xMf = 0, 0.5
    ymf, yMf = np.round(y.min(), 1) - 0.1, np.round(y.max(), 1) + 0.1
    xr, yr = [xmf, xMf], [ymf, yMf]

    ax2.set_ylim(yr)
    ax2.set_xticks(xr)
    ax2.set_yticks(yr)

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    if i == 1:
        ax2.set_xlabel('speed (m/s)', fontsize=13)
    ax2.set_ylabel('firing rate (Hz)', fontsize=13)

fig.tight_layout()

fig = plt.figure(figsize=(8,4), constrained_layout=True)
spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)
for i in range(2):
    for j in range(0, 3):
        ax = fig.add_subplot(spec[i, j])
        cid = cell_id_list[i+j*2]
        ax.text(0.2, 0.2, 'k = {} \n $y_0$ = {} \n r = {}'.format(slopes[cid], intercepts[cid], r_values[cid]), verticalalignment='bottom', fontsize=13)
plt.show()

fig, ax = plt.subplots(13, 11, figsize=(10, 10))
ax = ax.flatten()
for cell_id in np.arange(0, data['mean_firing_rate'].shape[1]):
    ax[cell_id].scatter(data['speed_bins'][1:], data['mean_firing_rate'][:, cell_id], s=2)
    ax[cell_id].set_title(cell_id)
plt.show()
