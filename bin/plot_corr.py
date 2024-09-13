# Please run generate_corr_data.py first
import numpy as np
import matplotlib.pyplot as plt
from grid_cell.ploter import error_bar_plot
import os
from global_setting import *

#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'
pca_str = '6' # none or number as a string

corr_file_name = 'corr_data_{}_{}_{}_{}_pca{}.npz'.format(mouse_name, module, day, session, pca_str)
corr_path = os.path.join(DATAROOT, corr_file_name)

data = np.load(corr_path)
print(data.files)

nt_gp, corr_gp = data['n_time_point_list_gp'], data['correlations_gp']
nt_avg, corr_avg = data['n_time_point_list_avg'], data['correlations_avg']

# find the range of broken axes
fig, ax = plt.subplots(figsize=(3, 3))
fig, ax = error_bar_plot(nt_gp, corr_avg[:len(nt_gp)], color='tab:orange', fig=fig, ax=ax, label='Avg')
fig, ax = error_bar_plot(nt_gp, corr_gp, ax=ax, fig=fig, color='tab:blue', label='GPR')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
for axis in ['bottom', 'left']:
  ax.spines[axis].set_linewidth(2)
ax.set_xlabel('Number of time points')
ax.set_ylabel('Pearson Correlation with \n the approximated ground truth \n (All PC units)')
ax.set_ylim([0, 1])
plt.legend()
fig.tight_layout()

fig, ax = plt.subplots(figsize=(3, 3))
fig, ax = error_bar_plot(nt_avg[len(nt_gp):], corr_avg[len(nt_gp):], color='tab:orange', fig=fig, ax=ax, label='Avg')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
for axis in ['bottom', 'left']:
  ax.spines[axis].set_linewidth(2)
ax.set_xlabel('Number of time points')
ax.set_ylabel('Pearson Correlation with \n the approximated ground truth \n (All PC units)')
ax.set_ylim([0, 1])
plt.legend()
fig.tight_layout()
plt.show()
