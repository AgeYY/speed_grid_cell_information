# the change of rate map with speed
import numpy as np
import matplotlib.pyplot as plt
from grid_cell.grid_cell_processor import Grid_Cell_Processor, Speed_Processor
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from global_setting import *

#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'
# session = 'wagon_wheel_1'
speed_win_list = [[i, i + 0.025] for i in np.arange(0.025, 0.5, 0.1)]
gridness_thre = 0.3
eg_neuron_id = 25

#################### Main ####################
# '''
gcp = Grid_Cell_Processor()
gcp.load_data(mouse_name, day, module, session, fr_smooth_sigma=10)
fire_rate, x, y, t, speed = gcp.preprocess(gridness_thre=gridness_thre, pca_components=None, use_zscore=False, return_speed=True)
label = np.array([x, y, t, speed]).T

sp = Speed_Processor()
sp.load_data(fire_rate, label)

### compute the rate map for each speed window
rate_map = []
speed_list = []
for speed_win in speed_win_list:
    rate_map.append(sp.compute_rate_map_speedwin(speed_win))
    speed_list.append(speed_win[0])

### get the example neuron
eg_rate_map = [rate_map[i][:, :, eg_neuron_id] for i in range(len(rate_map))]

np.savez('eg_rate_map_speed.npz', eg_rate_map=eg_rate_map, speed_list=speed_list)
# '''

data = np.load('eg_rate_map_speed.npz')
eg_rate_map = data['eg_rate_map']
speed_list = data['speed_list']

#################### Plot ####################
n_speed = len(eg_rate_map)
fig = plt.figure(figsize=(n_speed * 2, 2))
gs = fig.add_gridspec(1, n_speed+1, width_ratios=[1]*n_speed + [0.4])

vmin, vmax = np.min(eg_rate_map), np.max(eg_rate_map)

for i, im, s in zip(range(n_speed), eg_rate_map, speed_list):
    ax = fig.add_subplot(gs[0, i])
    img = ax.imshow(im, vmin=vmin, vmax=vmax, aspect='equal', cmap='seismic')
    ax.axis('off')
    if i == 0:
        ax.set_title(f'Speed: {s:.3f} m/s', fontsize=13)
    else:
        ax.set_title(f'{s:.3f}', fontsize=13)

cax = fig.add_subplot([.92, .1, .01, .8])
cbar = fig.colorbar(img, cax=cax)
cbar.set_label('Firing Rate (Hz)', rotation=270, labelpad=20, fontsize=13)
fig.tight_layout()
plt.show()
