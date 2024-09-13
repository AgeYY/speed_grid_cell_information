# similar to plot_torus_size, but instead of data and shuffle_data, this plot data and ideal iid grid cell data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from grid_cell.slope_ploter import SlopeAnalysis, draw_two_line_data, add_small_noise, draw_two_line_data_boot
import hickle as hkl
import os
from global_setting import *

##################################################
# Plot Results for eg_dn dataset                  #
##################################################
ykeys = ['lattice_area', 'radius', 'center_dist', 'total_noise', 'noise_proj', 'noise_ratio', 'total_fisher']
y_label = [
    'Lattice Area' + r'($\mathrm{Hz}^2/\mathrm{cm}^2$)',
    'Torus Radius' + r'($\mathrm{Hz}$)',
    'Torus Center Distance' + r'($\mathrm{Hz}$)',
    'Total Noise' + r'($\mathrm{Hz}^{2}$)',
    'Projected Noise' + r'($\mathrm{Hz}^{2}$)',
    'Noise Ratio (a.u.)',
    'Total Fisher' + r'($\mathrm{cm}^{-2}$)',
]
dn = 'r1m2'
n_pca = None
EPS = 1e-8 # std of the small noise will be added to the y data for numerical stability

torus_size_path = os.path.join(DATAROOT, f'torus_shape_speed_{dn}_pca{n_pca}_iid{False}.hkl')
size_data = hkl.load(torus_size_path)
shuffle_torus_size_path = os.path.join(DATAROOT, f'torus_shape_speed_{dn}_pca{n_pca}_iid{True}.hkl')
shuffle_size_data = hkl.load(shuffle_torus_size_path)

for i, key in enumerate(ykeys):
    print(f'Processing {key}')
    speed = size_data['speed']
    y_shape = size_data[key].shape
    y = add_small_noise(size_data[key].flatten(), EPS).reshape(y_shape)
    shuffle_y = add_small_noise(shuffle_size_data[key].flatten(), EPS).reshape(y_shape)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax = draw_two_line_data_boot(speed, y, shuffle_y, ax, data_label='Original', shuffle_label='IFGC', color_data='tab:blue', color_shuffle='tab:orange', mode='BBLR')
    ax.set_xlabel('Speed (cm/s)')
    ax.set_ylabel(y_label[i])
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-1,3))
    ax.legend()
    fig.savefig(os.path.join(FIGROOT, f'{dn}_{key}_speed_IFGC.svg'))

##################################################
# Plot Slope Graph for all datasets              #
##################################################
# dataset_names = ['r1m1', 'r1m2']
dataset_names = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
y_keys = [ 'lattice_area', 'radius', 'center_dist', 'total_noise', 'noise_proj', 'noise_ratio', 'total_fisher']
do_not_draw = ['fisher']

####################
# Create meta data #
####################
meta_data = {}
for dn in dataset_names:
    torus_size_path = os.path.join(DATAROOT, f'torus_shape_speed_{dn}_pca{n_pca}_iid{False}.hkl')
    size_data = hkl.load(torus_size_path)
    shuffle_size_data_path = os.path.join(DATAROOT, f'torus_shape_speed_{dn}_pca{n_pca}_iid{True}.hkl')
    shuffle_size_data = hkl.load(shuffle_size_data_path)

    meta_data[dn] = {}
    meta_data[dn]['speed'] = size_data['speed']

    for yk in y_keys:
        meta_data[dn][yk] = add_small_noise(size_data[yk], EPS)
        meta_data[dn][f'shuffle_{yk}'] = add_small_noise(shuffle_size_data[yk], EPS)


####################
# Plot Slope Graph #
####################

for i, yk in enumerate(y_keys):
    print(f'Processing {yk}')
    if yk in do_not_draw:
        continue
    sa = SlopeAnalysis(meta_data, dataset_names, x_key='speed', y_key=yk, shuffle_y_key=f'shuffle_{yk}')
    fig, ax = sa.analyze_value(y_text_offset=0, data_color='tab:blue', shuffle_data_color='tab:orange', x_shift=0.1, mode='bootstrap')
    ax.set_ylabel('Speed-Averaged ' + y_label[i], fontsize=16)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,3))
    fig.savefig(os.path.join(FIGROOT, f'{yk}_IFGC.svg'))

plt.show()
