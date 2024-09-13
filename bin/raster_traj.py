import os
import numpy as np
import matplotlib.pyplot as plt
from grid_cell.util import select_arr_by_t, get_data_filename_by_keywords
from global_setting import *

mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'
pca_str = '6' # none or number as a string


## raw data
file_name = get_data_filename_by_keywords(mouse_name.lower(), day)
cell_module_name = 'spikes_mod{}'.format(module)
data_path = os.path.join(EXP_DATAROOT, file_name)

## preprocessed data
preprocessed_file_name = 'preprocessed_data_{}_{}_{}_{}_pca{}.npz'.format(mouse_name, module, day, session, pca_str)
preprocessed_dir = os.path.join(DATAROOT, 'preprocessed/')
if not os.path.exists(preprocessed_dir): os.makedirs(preprocessed_dir)
preprocessed_path = os.path.join(preprocessed_dir, preprocessed_file_name)

data = np.load(data_path, allow_pickle=True)
spike = data[cell_module_name].item()
x, y, t = data['x'], data['y'], data['t']
# data = np.load(preprocessed_path, allow_pickle=True)
# print(data['fire_rate'].shape)

def draw_raster_plot(neural_data):
    fig, ax = plt.subplots()
    for x_i, (neuron_id, spike_times) in enumerate(neural_data.items()):
        # ax.vlines(spike_times, ymin=x_i + .5, ymax=x_i + 1.5)
        ax.scatter(spike_times, [x_i + 1] * len(spike_times), color='k', s=0.1, marker='o')
    ax.set_ylim(.5, len(neural_data) + .5)
    ax.set_yticks([])
    ax.set_xlabel('Time (sec)', fontsize=15)
    ax.set_ylabel('Neuron ID', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

def filter_spikes_within_window(neural_data, start_time, end_time):
    filtered_data = {neuron_id: [t for t in spike_times if start_time <= t <= end_time] 
                     for neuron_id, spike_times in neural_data.items()}
    return filtered_data

def draw_trajectory(x, y, t, t_win):
    mask = (t >= t_win[0]) & (t <= t_win[1])  # build a mask based on t_win
    x_filt, y_filt, t_filt = x[mask], y[mask], t[mask]

    fig, ax = plt.subplots()
    sc = ax.scatter(x_filt, y_filt, c=t_filt, alpha=1, s=1)  # color based on time
    cbar = plt.colorbar(sc, label='Time (sec)', ticks=[t_win[0] + 1, t_win[1] - 1])
    cbar.ax.set_yticklabels([str(t_win[0] + 1), str(t_win[1] - 1)])
    cbar.ax.yaxis.label.set_size(14)
   
    # remove ticks and tick labels
    ax.set_xticks([])
    ax.set_yticks([])
   
    # make spines thicker
    [i.set_linewidth(2) for i in ax.spines.values()]

    ax.set_xlim(x_filt.min(), x_filt.max())
    ax.set_ylim(y_filt.min(), y_filt.max())

t_win = [7457, 7477]
filtered_spike = filter_spikes_within_window(spike, t_win[0], t_win[1])
draw_raster_plot(filtered_spike)
draw_trajectory(x, y, t, t_win)
plt.show()
