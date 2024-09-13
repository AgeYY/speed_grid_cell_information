# run lca_iid first
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
import os
from sklearn.linear_model import LogisticRegression
from grid_cell.grid_cell_processor import Speed_Processor
from grid_cell.locf_grid_cell import compute_locf
from grid_cell.toy_locf import calculate_locf_accuracy_upper_bound
from grid_cell.slope_ploter import SlopeAnalysis, draw_two_line_data, add_small_noise, draw_two_line_data_boot
import scipy.stats as stats
import matplotlib.ticker as ticker
from global_setting import *

##################################################
#################### With iid data ####################
##################################################
n_pca = None
#################### Plot example dataset's result ####################
dn = 'r1m2'
data = hkl.load(os.path.join(DATAROOT, f'accuracy_iid_{dn}_pca{n_pca}.hkl'))
speed_bins = data['speed_bins']; accuracy = data['accuracy']; accuracy_iid = data['accuracy_iid']
n_boot = accuracy.shape[0]
speed_bins = speed_bins[:-1]
speed = np.tile(speed_bins, (n_boot, 1)) * 100

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax = draw_two_line_data_boot(speed, accuracy_iid, accuracy, ax, data_label='IFGC', shuffle_label='Original', color_data='tab:orange', color_shuffle='tab:blue', mode='BBLR')

ax.set_xlabel('Speed (cm/s)')
ax.set_ylabel('Spatial Classification Accuracy (SCA)')
ax.legend()
fig.savefig(os.path.join(FIGROOT, f'lca_iid_pca{n_pca}.svg'))
plt.show()

#################### All datasets ####################
dataset_names = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
meta_data = {}
# combine all datasets
for dn in dataset_names:
    data = hkl.load(os.path.join(DATAROOT, f'accuracy_iid_{dn}_pca{n_pca}.hkl'))
    n_boot = data['accuracy'].shape[0]
    data['speed_bins'] = np.tile(data['speed_bins'][:-1], (n_boot, 1)) * 100
    meta_data[dn] = data

sa = SlopeAnalysis(meta_data, dataset_names, x_key='speed_bins', y_key='accuracy', shuffle_y_key=f'accuracy_iid')
fig, ax = sa.analyze_value(y_text_offset=0, data_color='tab:blue', shuffle_data_color='tab:orange', x_shift=0.1, mode='bootstrap')
ax.set_ylabel('Speed-Averaged SCA', fontsize=18)
fig.savefig(os.path.join(FIGROOT, f'accuracy_iid_all_dataset_pca{n_pca}.svg'))


##################################################
#################### With Shuffle data ####################
##################################################

#################### Plot example dataset's result ####################
dn = 'r1m1'
data = hkl.load(os.path.join(DATAROOT, f'accuracy_iid_{dn}_pca{n_pca}.hkl'))
speed_bins = data['speed_bins']; accuracy = data['accuracy']; accuracy_shuffle = data['accuracy_shuffle']
n_boot = accuracy.shape[0]
speed_bins = speed_bins[:-1] * 100
speed = np.tile(speed_bins, (n_boot, 1))

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax = draw_two_line_data_boot(speed, accuracy, None, ax, data_label='Original', shuffle_label='Label-Shuffled', color_data='tab:blue', color_shuffle='tab:grey', mode='BBLR')
# ax.hlines(0.5, speed_flat[0], speed_flat[-1], color='k', linestyle='--')

ax.set_xlabel('Speed (cm/s)')
ax.set_ylabel('Spatial Coding Accuracy')
ax.legend(fontsize=14)
fig.savefig(os.path.join(FIGROOT, f'lca_shuffle_pca{n_pca}.svg'))

#################### All datasets ####################
dataset_names = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
meta_data = {}
# combine all datasets
for dn in dataset_names:
    data = hkl.load(os.path.join(DATAROOT, f'accuracy_iid_{dn}_pca{n_pca}.hkl'))
    n_boot = data['accuracy'].shape[0]
    data['speed_bins'] = np.tile(data['speed_bins'][:-1], (n_boot, 1)) * 100
    meta_data[dn] = data

sa = SlopeAnalysis(meta_data, dataset_names, x_key='speed_bins', y_key='accuracy', shuffle_y_key=f'accuracy_shuffle')
fig, ax = sa.analyze(y_text_offset=0, data_color='tab:blue', shuffle_data_color='tab:grey', x_shift=0., add_connecting_line=False, mode='BBLR')
ax.set_ylabel('SCA-Speed Slope (a.u.)', fontsize=16)
ax.hlines(0., -0.2, len(dataset_names), color='k', linestyle='--')
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

fig.savefig(os.path.join(FIGROOT, f'accuracy_shuffle_all_dataset_pca{n_pca}.svg'))

plt.show()
