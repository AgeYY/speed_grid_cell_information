# run num_time_point_gp.py first
import numpy as np
import hickle as hkl
from grid_cell.ploter import error_bar_plot
import matplotlib.pyplot as plt
import os
from global_setting import *

#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'
pca_str = '6' # none or number as a string

it_gp_timepoint_file_name = 'it_timepoint_gp_{}_{}_{}_{}_pca{}.hkl'.format(mouse_name, module, day, session, pca_str)
it_gp_timepoint_path = os.path.join(DATAROOT, it_gp_timepoint_file_name)
it_avg_timepoint_file_name = 'it_timepoint_avg_{}_{}_{}_{}_pca{}.hkl'.format(mouse_name, module, day, session, pca_str)
it_avg_timepoint_path = os.path.join(DATAROOT, it_avg_timepoint_file_name)
it_none_timepoint_file_name = 'it_timepoint_none_{}_{}_{}_{}_pca{}.hkl'.format(mouse_name, module, day, session, pca_str)
it_none_timepoint_path = os.path.join(DATAROOT, it_none_timepoint_file_name)

data_gp = hkl.load(it_gp_timepoint_path)
data_avg = hkl.load(it_avg_timepoint_path)
data_none = hkl.load(it_none_timepoint_path)

nt_gp, it_list_gp = data_gp['n_time_point_list'], data_gp['it_list_gp']
nt_avg, it_list_avg = data_avg['n_time_point_list'], data_avg['it_list_avg']
nt_none, it_list_none = data_none['n_time_point_list'], data_none['it_list_none']

# find the range of broken axes
fig, ax = plt.subplots(figsize=(3, 3))
fig, ax = error_bar_plot(nt_gp, it_list_avg[:len(nt_gp)], color='tab:orange', fig=fig, ax=ax, label='Avg')
fig, ax = error_bar_plot(nt_gp, it_list_none[:len(nt_gp)], color='tab:green', fig=fig, ax=ax, label='PPData')
fig, ax = error_bar_plot(nt_gp, it_list_gp, ax=ax, fig=fig, color='tab:blue', label='GPR')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
for axis in ['bottom', 'left']:
  ax.spines[axis].set_linewidth(2)
ax.set_xlabel('Number of time points')
ax.set_ylabel('Probability of finding a torus')
plt.legend()
fig.tight_layout()

fig, ax = plt.subplots(figsize=(3, 3))
fig, ax = error_bar_plot(nt_avg[len(nt_gp):], it_list_avg[len(nt_gp):], color='tab:orange', fig=fig, ax=ax, label='Avg')
fig, ax = error_bar_plot(nt_none[len(nt_gp):], it_list_none[len(nt_gp):], color='tab:green', fig=fig, ax=ax, label='none')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
for axis in ['bottom', 'left']:
  ax.spines[axis].set_linewidth(2)
ax.set_xlabel('Number of time points')
ax.set_ylabel('Probability of finding a torus')
plt.legend()
fig.tight_layout()
plt.show()
