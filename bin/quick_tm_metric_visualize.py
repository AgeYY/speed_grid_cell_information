import hickle as hkl
import os
from grid_cell.ploter import error_bar_plot
import matplotlib.pyplot as plt
from global_setting import *

tm_name = 'place_cell'
data = hkl.load(os.path.join(DATAROOT, "manifold_sim_score_n_data.hkl"))
n_points_list = data['n_points_list']
print(data)
# data = hkl.load(os.path.join(DATAROOT, "manifold_sim_score_n_dim.hkl"))
# n_points_list = data['n_dim_list']

def plot_sim_scores(n_points, gpr_sim_score_arr, kernel_sim_score_arr, title_suffix, ax=None, fig=None, **plot_kwargs):
    if ax is None: fig, ax = plt.subplots(figsize=(3, 3))
    error_bar_plot(n_points, gpr_sim_score_arr, label='GGPR', color='tab:blue', error_mode='std', mean_mode='mean', fig=fig, ax=ax, **plot_kwargs)
    error_bar_plot(n_points, kernel_sim_score_arr, label='Kernel Avg.', color='tab:green', error_mode='std', mean_mode='mean', fig=fig, ax=ax, **plot_kwargs)
    ax.set_title(f'{title_suffix}')
    ax.set_xlabel('Number of Data Points')

fig, axes = plt.subplots(1, 4, figsize=(10, 3))
plot_sim_scores(n_points_list, data[f'{tm_name}_gpr_sim_score_arr'], data[f'{tm_name}_kernel_sim_score_arr'], 'Manifold', ax=axes[0], fig=fig)
axes[0].set_ylabel('Similarity to the Ground Truth')

plot_sim_scores(n_points_list, data[f'{tm_name}_gpr_cov_sim_score_arr'], data[f'{tm_name}_kernel_cov_sim_score_arr'], 'Covariance', ax=axes[1], fig=fig)
plot_sim_scores(n_points_list, data[f'{tm_name}_gpr_metric_sim_score_arr'], data[f'{tm_name}_kernel_metric_sim_score_arr'], 'Riemannian Metric', ax=axes[2], fig=fig)
plot_sim_scores(n_points_list, data[f'{tm_name}_gpr_fisher_sim_score_arr'], data[f'{tm_name}_kernel_fisher_sim_score_arr'], 'Fisher Information', ax=axes[3], fig=fig)
axes[0].legend()
fig.tight_layout()
plt.show()
