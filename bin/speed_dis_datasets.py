import numpy as np
import matplotlib.pyplot as plt
from grid_cell.grid_cell_processor import Grid_Cell_Processor, Speed_Processor
from matplotlib.ticker import FuncFormatter
import os
from global_setting import *

#################### Hyperparameters ####################
# dataset_name = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
dataset_name = ['r1m1', 'r2m1', 's1m1', 'q1m1']
binwidth = 0.05
preprocessed_file_name = 'preprocessed_data_OF.hkl'

fig, axes = plt.subplots(2, len(dataset_name), figsize=(10, 5))
for i_dn, dn in enumerate(dataset_name):
    mouse_name, day, module = dn[0], 'day' + dn[1], dn[-1]

    #################### Main ####################
    gcp = Grid_Cell_Processor()
    gcp.load_data(mouse_name, day, module, 'open_field_1', speed_estimation_method='finite_diff')
    fire_rate = gcp.fire_rate
    x, y, t, dt = gcp.x, gcp.y, gcp.t, gcp.dt

    speed = gcp.compute_speed()
    label = np.array([x, y, t, speed]).T
    speed_bins = np.arange(min(speed), max(speed) + binwidth, binwidth)
    x_min, x_max = 0.05, 0.8

    ### Speed distribution
    # fig, ax = plt.subplots(figsize=(3, 3))
    ax_speed = axes[0, i_dn]
    speed_data_idx = (speed > x_min) & (speed < x_max)
    ax_speed.hist(speed[speed_data_idx] * 100, bins=speed_bins * 100, facecolor='tab:blue', color='k') # * 100 to convert to cm/s
    if i_dn == 0: ax_speed.set_ylabel('Number of Data Points')
    ax_speed.set_xlim([(x_min - 0.02) * 100, x_max * 100])
    ax_speed.axvline(x=45, color='tab:red', linestyle='--')
    ax_speed.axvline(x=5, color='tab:red', linestyle='--')
    ax_speed.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax_speed.set_title(dn.upper())

    ### OF covering rate
    sp = Speed_Processor()
    sp.load_data(fire_rate, label, dt=dt)
    bound = [-0.75, 0.75]
    cover_rate = sp.compute_OF_cover_rate(speed_bins, x_bound=bound, y_bound=bound, n_spatial_bin=30) # bin the space into 5 cm to compute occupancy

    ax_cover = axes[1, i_dn]
    ax_cover.scatter(speed_bins[1:] * 100, cover_rate[1:], color='tab:blue', s=10) # * 100 to convert to cm/s
    ax_cover.plot(speed_bins[1:] * 100, cover_rate[1:], color='tab:blue')
    ax_cover.set_xlabel('Speed (cm/s)')
    if i_dn == 0: ax_cover.set_ylabel('Fraction of OF area visited \n within each speed bin')
    ax_cover.set_xlim([(x_min - 0.02) * 100, x_max * 100])
    ax_cover.axvline(x=45, color='tab:red', linestyle='--')
    ax_cover.axvline(x=5, color='tab:red', linestyle='--')

fig.tight_layout()
fig.savefig(os.path.join(FIGROOT, 'speed_dis_datasets.svg'))
plt.show()
