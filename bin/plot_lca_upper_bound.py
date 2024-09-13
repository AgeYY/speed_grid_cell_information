# # measure the localling decoding performance on different data. Run generate_processed_exp_data.py first. Requires run lca_upper_bound.py and generate_processed_exp_data.py first
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
import os
from sklearn.linear_model import LogisticRegression
from grid_cell.slope_ploter import draw_two_line_data, add_small_noise, plot_error_bars_with_text, get_significant_text, draw_two_line_data_boot
import scipy.stats as stats
from global_setting import *

def compute_corr(up_flat, accuracy_flat):
    # compute correlation, p value of correlation and confidence interval
    correlation, p_value = stats.pearsonr(up_flat, accuracy_flat)
    n = len(up_flat)
    z = np.arctanh(correlation)
    z_se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - 0.05 / 2)
    ci = np.tanh([z - z_crit * z_se, z + z_crit * z_se])
    return correlation, p_value, ci

#################### Plot one example dataset ####################
n_pca = 6
broken_axis = (n_pca is None)
dn = 'r1m2'
data = hkl.load(os.path.join(DATAROOT, f'locf_fisher_speed_box_{dn}_pca{n_pca}.hkl'))
speed_bins = data['speed_bins']; accuracy = data['accuracy']; upper_bound = data['upper_bound']
print(accuracy.shape)
n_boot = upper_bound.shape[0]

speed_bins = speed_bins[:-1]
speed = np.tile(speed_bins, (n_boot, 1))

# broken axis
if broken_axis:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 4))
    ax1 = draw_two_line_data_boot(speed, upper_bound, accuracy, ax1, data_label='Upper Bound \n (derived from Fisher info)', shuffle_label='SCA from data', color_data='tab:red', color_shuffle='tab:blue', mode='BBLR')
    ax2 = draw_two_line_data_boot(speed, upper_bound, accuracy, ax2, data_label='Upper Bound \n (derived from Fisher info)', shuffle_label='SCA from data', color_data='tab:red', color_shuffle='tab:blue', mode='BBLR')
    ax1.set_ylim([0.85, 0.90])
    ax2.set_ylim([0.62, 0.70])

    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.tick_params(bottom=False)  # don't put tick labels at the top
    ax1.set_xticklabels([])

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    ax2.set_xlabel('Speed (cm/s)')
    ax1.set_ylabel('Spatial Coding Accuracy (SCA)')
    ax1.legend()
else:
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax = draw_two_line_data_boot(speed, upper_bound, accuracy, ax, data_label='Upper Bound \n (derived from Fisher info)', shuffle_label='SCA from data', color_data='tab:red', color_shuffle='tab:blue', mode='BBLR')
    
    ax.set_xlabel('Speed (cm/s)')
    ax.set_ylabel('Spatial Coding Accuracy (SCA)')
    ax.legend()

fig.savefig(os.path.join(FIGROOT, 'locf_fisher_speed_box.svg'))

##################################################
# Plot Correlation Graph for all datasets              #
##################################################
datasets_names = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
# datasets_names = ['r1m1', 'r1m2']
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
xlabel_list = []
for i, dn in enumerate(datasets_names):
    # load data
    data = hkl.load(os.path.join(DATAROOT, f'locf_fisher_speed_box_{dn}_pca{n_pca}.hkl'))
    accuracy = data['accuracy']; upper_bound = data['upper_bound']
    n_boot = upper_bound.shape[0]

    ####################
    # compute correlation #
    ####################

    # ########## bootstrap method for estimating correlation and ci
    # corr = np.array([stats.pearsonr(upper_bound[i], accuracy[i])[0] for i in range(n_boot)])
    # corr_mean = np.mean(corr)
    # corr_std = np.std(corr)
    # p_value = stats.norm.cdf(0, loc=corr_mean, scale=corr_std)
    # p_value = 2 * min(p_value, 1 - p_value)
    # ci = [corr_mean - 1.96 * corr_std, corr_mean + 1.96 * corr_std]

    ########## t-test method for estimating correlation and ci
    speed_bins = speed_bins[:-1]
    speed_flat = np.tile(speed_bins, (n_boot, 1)).flatten()
    up_flat = upper_bound.flatten()
    accuracy_flat = accuracy.flatten()
    corr_mean, p_value, ci = compute_corr(up_flat, accuracy_flat)

    yerr = np.array([[corr_mean - ci[0], ci[1] - corr_mean]]).T

    ####################
    # plot correlation #
    ####################
    lower_text = get_significant_text(p_value)
    ax = plot_error_bars_with_text(ax, i, corr_mean, yerr, 'k', upper_text='', lower_text=lower_text, y_text_offset=0.2)
    xlabel_list.append(f'{dn.upper()}')

ax.set_xticks(np.arange(len(datasets_names)))
ax.set_xticklabels(xlabel_list, fontsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.set_ylabel('Correlation between SCA and Upper Bound', fontsize=16)
ax.hlines(0, -1, len(datasets_names), linestyles='dashed', colors='gray')
fig.savefig(os.path.join(FIGROOT, 'up_lca_correlation.svg'))
plt.show()
