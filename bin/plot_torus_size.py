import numpy as np
import matplotlib.pyplot as plt
from grid_cell.linear_regression import fit_a_line, output_params, draw_line
import matplotlib.ticker as ticker
from grid_cell.slope_ploter import SlopeAnalysis, draw_two_line_data, add_small_noise, draw_two_line_data_boot
import hickle as hkl
import os
from global_setting import *

##################################################
# Plot Results for eg_dn dataset                  #
##################################################
dataset_names = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
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
dn = dataset_names[1]
n_pca = None
iid_mode = False
EPS = 1e-5 # std of the small noise will be added to the y data for numerical stability

torus_size_path = os.path.join(DATAROOT, f'torus_shape_speed_{dn}_pca{n_pca}_iid{iid_mode}.hkl')
size_data = hkl.load(torus_size_path)
shuffle_torus_size_path = os.path.join(DATAROOT, f'shuffle_torus_shape_speed_{dn}_pca{n_pca}_iid{False}.hkl')
shuffle_size_data = hkl.load(shuffle_torus_size_path)

for i, key in enumerate(ykeys):
    print(f'Processing {key}')
    speed = size_data['speed']
    y_shape = size_data[key].shape
    y = add_small_noise(size_data[key].flatten(), EPS).reshape(y_shape)
    # shuffle_y_flat = add_small_noise(shuffle_size_data[key].flatten(), EPS)
    shuffle_y = None # You can uncomment the above line to show the results of label-shuffled data sets

    fig, ax = plt.subplots(figsize=(3, 3))
    ax = draw_two_line_data_boot(speed, y, shuffle_y, ax, mode='BBLR')
    ax.set_xlabel('Speed (cm/s)')
    ax.set_ylabel(y_label[i])
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,3))
    fig.savefig(os.path.join(FIGROOT, f'{dn}_{key}_speed.svg'))

##################################################
# Plot Slope Graph for all datasets              #
##################################################
y_keys = [ 'lattice_area', 'radius', 'center_dist', 'total_noise', 'noise_proj', 'noise_ratio', 'fisher', 'total_fisher']
y_label = [
    'Lattice Area-Speed Slope' + r'($\mathrm{Hz}/\mathrm{cm}^3$)',
    'Torus Radius-Speed Slope' + r'($\mathrm{cm}^{-1}$)',
    'Torus Center Distance-Speed Slope' + r'($\mathrm{cm}^{-1}$)',
    'Total Noise-Speed Slope' + r'($\mathrm{Hz} / \mathrm{cm}$)',
    'Projected Noise-Speed Slope' + r'($\mathrm{Hz} / \mathrm{cm}$)',
    'Noise Ratio-Speed Slope (a.u.)',
    'Fisher-Speed Slope' + r'($\mathrm{s} / \mathrm{cm}^2$)',
    'Total Fisher-Speed Slope' + r'($\mathrm{s} / \mathrm{cm}^2$)',
]
do_not_draw = ['fisher']

####################
# Create meta data #
####################
meta_data = {}
for dn in dataset_names:
    torus_size_path = os.path.join(DATAROOT, f'torus_shape_speed_{dn}_pca{n_pca}_iid{iid_mode}.hkl')
    size_data = hkl.load(torus_size_path)
    shuffle_size_data_path = os.path.join(DATAROOT, f'shuffle_torus_shape_speed_{dn}_pca{n_pca}_iid{False}.hkl')
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
    fig, ax = sa.analyze(y_text_offset=0.2, mode='BBLR')
    ax.set_ylabel(y_label[i], fontsize=16)
    ax.hlines(0, -1, len(dataset_names), color='k', linestyle='--')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,3))
    fig.savefig(os.path.join(FIGROOT, f'{yk}_slope.svg'))

plt.show()
