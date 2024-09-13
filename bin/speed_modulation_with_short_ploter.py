import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from grid_cell.util import get_line
import matplotlib.gridspec as gridspec
import os
from global_setting import *

#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'

speed_temporal_file_name = 'speed_temporal_{}_{}_{}_{}.npz'.format(mouse_name, module, day, session)
speed_temporal_path = os.path.join(DATAROOT, speed_temporal_file_name)
speed_temporal_short_file_name = 'speed_temporal_short_{}_{}_{}_{}.npz'.format(mouse_name, module, day, session)
speed_temporal_short_path = os.path.join(DATAROOT, speed_temporal_short_file_name)

cell_id_list = [25, 5, 106, 9, 10] # new id in after gridness > 0.3
cell_title_list = ['linear growth', 'flat then grow',  'linear decay', 'decay then flat', 'weak/no speed \n modulation']
#################### Main ####################
data = np.load(speed_temporal_path, allow_pickle=True)
data_short = np.load(speed_temporal_short_path, allow_pickle=True)

slopes, intercepts, r_values = data['slopes'], data['intercepts'], data['r_values'] # for drawing regression line
slopes_shuf = data['slopes_shuf']  # for computing the statistical significance of the regression line
slope_pval = np.mean(slopes_shuf >= slopes, axis=0)

slopes_short, intercepts_short, r_values_short = data_short['slopes'], data_short['intercepts'], data_short['r_values'] # for drawing regression line
slopes_shuf_short = data_short['slopes_shuf']  # for computing the statistical significance of the regression line
slope_pval_short = np.mean(slopes_shuf_short >= slopes_short, axis=0)

pv = np.array([stats.percentileofscore(slopes_shuf[:,cell], slopes[cell])/100 for cell in range(slopes.shape[0])])
slope_pval = 2 * np.minimum(pv, 1 - pv)

pv_short = np.array([stats.percentileofscore(slopes_shuf_short[:,cell], slopes_short[cell])/100 for cell in range(slopes_short.shape[0])])
slope_pval_short = 2 * np.minimum(pv_short, 1 - pv_short)
rate_map = data['rate_map']

#################### Visualization
# plot mean fr as a function of speed
fig = plt.figure(figsize=(12,4), constrained_layout=True)
spec = gridspec.GridSpec(ncols=6, nrows=2, figure=fig)

for i in range(2):
  for j in range(0, 6, 2):
    if i+j >= len(cell_id_list): continue
    ax1 = fig.add_subplot(spec[i, j])
    ax2 = fig.add_subplot(spec[i, j + 1])

    cid = cell_id_list[i+j]
    ax1.imshow(rate_map[:, :, cid])
    ax1.axis('off')
    # title indicating slope and p value
    ax1.text(0.5, 1.15, 's: {:.2f}, p: {:.2f}'.format(slopes[cid], slope_pval[cid]), transform=ax1.transAxes, fontsize=10, ha='center', color='tab:blue')
    ax1.text(0.5, 1.02, 's: {:.2f}, p: {:.2f}'.format(slopes_short[cid], slope_pval_short[cid]), transform=ax1.transAxes, fontsize=10, ha='center', color='tab:red')
    # ax1.set_title(str(cell_title_list[i+j]), fontsize=13)

    # results from temporal-bin-based regression
    ax2.scatter(data['speed_bins'][:-1], data['mean_firing_rate'][:, cid], s=3, color='tab:blue')
    ax2.fill_between(data['speed_bins'][:-1], data['low_ci'][:, cid], data['up_ci'][:, cid], color='gray', alpha=0.5)
    x_line, y_line = get_line(ax2, slopes[cid], intercept=intercepts[cid])
    ax2.plot(x_line, y_line, color='tab:blue', label='Origin')

    ax2.scatter(data_short['speed_bins'][:-1], data_short['mean_firing_rate'][:, cid], s=3, color='tab:red')
    ax2.fill_between(data_short['speed_bins'][:-1], data_short['low_ci'][:, cid], data_short['up_ci'][:, cid], color='gray', alpha=0.5)
    x_line, y_line = get_line(ax2, slopes_short[cid], intercept=intercepts_short[cid])
    ax2.plot(x_line, y_line, color='tab:red', label='Small')

    if (i == 0) and (j==0): ax2.legend(frameon=False)


    xmf, xMf = 0, 0.5
    ymin = data['mean_firing_rate'][:, cid].min()
    ymax = data['mean_firing_rate'][:, cid].max()
    ymf, yMf = np.round(ymin, 1) - 0.1, np.round(ymax, 1) + 0.1
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

slope_dis_range = np.min(slopes), np.max(slopes)
intercept_dis_range = np.min(intercepts), np.max(intercepts)
rvalue_dis_range = np.min(r_values), np.max(r_values)

fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.5))
ax = ax.flatten()
ax[0].hist(slopes, bins=40, facecolor='tab:blue', edgecolor='k', linewidth=0.3)
ax[0].set_ylabel('Number of Neurons', fontsize=13)
ax[0].set_xlabel('Slope', fontsize=13)
ax[0].axvline(0, color='r', linestyle='--')
ax[0].set_xlim(slope_dis_range)
ax[1].hist(intercepts, bins=40, facecolor='tab:blue', edgecolor='k', linewidth=0.3)
ax[1].set_xlabel('y intercept (Hz)', fontsize=13)
ax[1].axvline(0, color='r', linestyle='--')
ax[1].set_xlim(intercept_dis_range)
ax[2].hist(r_values, bins=40, facecolor='tab:blue', edgecolor='k', linewidth=0.3)
ax[2].set_xlabel('R value', fontsize=13)
ax[2].axvline(0, color='r', linestyle='--')
ax[2].set_xlim(rvalue_dis_range)
fig.tight_layout()

fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.5))
ax = ax.flatten()
ax[0].hist(slopes_short, bins=40, facecolor='tab:red', edgecolor='k', linewidth=0.3)
ax[0].set_ylabel('Number of Neurons', fontsize=13)
ax[0].set_xlabel('Slope', fontsize=13)
ax[0].axvline(0, color='r', linestyle='--')
ax[0].set_xlim(slope_dis_range)
ax[1].hist(intercepts_short, bins=40, facecolor='tab:red', edgecolor='k', linewidth=0.3)
ax[1].set_xlabel('y intercept (Hz)', fontsize=13)
ax[1].axvline(0, color='r', linestyle='--')
ax[1].set_xlim(intercept_dis_range)
ax[2].hist(r_values_short, bins=40, facecolor='tab:red', edgecolor='k', linewidth=0.3)
ax[2].set_xlabel('R value', fontsize=13)
ax[2].axvline(0, color='r', linestyle='--')
ax[2].set_xlim(rvalue_dis_range)
fig.tight_layout()

# fig = plt.figure(figsize=(8,4), constrained_layout=True)
# spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)
# for i in range(2):
#     for j in range(0, 3):
#         if i + j*2 >= len(cell_id_list): continue
#         ax = fig.add_subplot(spec[i, j])
#         cid = cell_id_list[i+j*2]
#         ax.text(0.2, 0.2, 'slope: {:.2f} \n p_value: {:.2f}'.format(slopes[cid], slope_pval[cid]), verticalalignment='bottom', fontsize=13, color='tab:blue')
#         ax.text(0.2, 0.8, 'slope: {:.2f} \n p_value: {:.2f}'.format(slopes_short[cid], slope_pval_short[cid]), verticalalignment='bottom', fontsize=13, color='tab:red')

print('Number of cells with significant speed modulation (max speed = 0.5): {}'.format(np.sum(slope_pval < 0.01)))
print('Number of cells with significant speed modulation (max speed = 0.3): {}'.format(np.sum(slope_pval_short < 0.01)))
n_cell = slope_pval.shape[0]
n_speed_cell_origin = np.sum(slope_pval < 0.01)
n_speed_cell_short = np.sum(slope_pval_short < 0.01)

labels = ['Origin', 'Small']
non_speed_cells = [n_cell - n_speed_cell_origin, n_cell- n_speed_cell_short]
speed_cells = [n_speed_cell_origin, n_speed_cell_short]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(3, 3))
rects1 = ax.bar(x, speed_cells, width, label='Speed Modulated Cells', color='tab:blue')
rects2 = ax.bar(x, non_speed_cells, width, bottom=speed_cells, label='Non Speed Modulated Cells', color='grey')

# Add labels for each bar
for i in range(len(non_speed_cells)):
    ax.text(i, speed_cells[i]/2, str(speed_cells[i]), ha='center', va='bottom', color='white')
    ax.text(i, speed_cells[i] + non_speed_cells[i]/2, str(non_speed_cells[i]), ha='center', va='bottom', color='white')

ax.set_ylabel('Number of Cells')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(frameon=False, loc='upper center')

fig.tight_layout()
plt.show()
# print(np.sum(slope_pval < 0.01))
# for i in range(len(slope_pval)):
#     print(i, slope_pval[i], slopes[i])

# print(slope_pval)
# print(slopes)
# print(len(slope_pval))

plt.show()
