import hickle as hkl
import os
import numpy as np
import matplotlib.pyplot as plt
from grid_cell.ploter import error_bar_plot
from global_setting import *

# Assuming DATAROOT is defined and points to your data directory
tm_name = ['ori_tuning', 'place_cell']  # You can add more if needed
# model_names = ['gpr', 'kernel', 'avg', 'gwpr']  # List of method names
model_names = ['gkr', 'lw', 'avg']
label_dict = {'gpr': 'GGPR', 'kernel': 'Kernel', 'avg': 'Bin', 'gwpr': 'GWPR', 'gkr': 'GKR', 'lw': 'LW', 'gwm': 'GWM'}

# data = hkl.load(os.path.join(DATAROOT, "manifold_sim_score_n_data.hkl"))
# x_list = data['n_points_list']
# x_label = 'Number of Data Points'
# out_file_tail = 'n_data'

data = hkl.load(os.path.join(DATAROOT, "manifold_sim_score_n_dim.hkl"))
x_list = data['n_dim_list']
for tm in tm_name: data[f'{tm}_blanket_n_points'] = 0
x_label = 'Number of Neurons'
out_file_tail = 'n_dim'

def plot_sim_scores(x, score_lists, title_suffix, ax=None, fig=None, labels=[], colors=[], **plot_kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))

    for i, scores in enumerate(score_lists):
        # error_bar_plot(x, scores, label=labels[i], color=colors[i], error_mode='std', mean_mode='mean', fig=fig, ax=ax, **plot_kwargs)
        error_bar_plot(x, scores, label=labels[i], color=colors[i], error_mode='quantile', mean_mode='median', fig=fig, ax=ax, **plot_kwargs)

    ax.set_title(f'{title_suffix}')

for tm in tm_name:
    fig, axes = plt.subplots(1, 5, figsize=(12, 3.2))
    fig.suptitle(f'{tm.upper()} Manifold')
    blanket_n_points = data[f'{tm}_blanket_n_points']
    x = np.array(x_list) + blanket_n_points

    for i, metric in enumerate(['manifold_', 'cov_', 'metric_', 'fisher_', 'precision_']):
        print('Plotting', i)
        score_lists = [data[f'{tm}_{name}_{metric}sim_score_arr'] for name in model_names]
        labels = [label_dict[name] for name in model_names]
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'k']

        if metric == 'metric_':
            plot_sim_scores(x, score_lists, 'Riemannian Metric', ax=axes[i], labels=labels, colors=colors, fig=fig)
        else:
            plot_sim_scores(x, score_lists, metric.title().strip('_'), ax=axes[i], labels=labels, colors=colors, fig=fig)
        axes[i].set_xlabel(x_label)
        plt.legend()
        if i == 0:
            axes[i].set_ylabel('Reltaive Prediction Error (a.u.)')
        y_max = np.min([np.max(score_lists), 2.0])
        axes[i].set_ylim([0, y_max])
    fig.tight_layout()
    fig.savefig(os.path.join(FIGROOT, f'{tm}_manifold_metrics_{out_file_tail}.svg'))
plt.show()
