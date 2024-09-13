import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from grid_cell.manifold_fitter import Avg_Fitter
from grid_cell.ggpr import GGPR_Fitter
from grid_cell.gkr import GKR_Fitter
from grid_cell.gwpr import GWPR_Fitter
from grid_cell.manifold_fitter import label_mesh
from grid_cell.manifold_visualizer import fig_ax_by_dim, plot_scatter_or_line_by_dim, fit_plot_manifold_figure, plot_cov_ellipse
from grid_cell.ori_tuning_manifold import Ori_Tuning_Manifold, Place_Cell_Manifold
from global_setting import *

def set_ax_limits(ax, xlim, ylim, zlim, visualization_dim=3):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if visualization_dim == 3: ax.set_zlim(zlim)
    return ax

# initialize the manifold class
n_neuron = 10
pcm = Place_Cell_Manifold(n_cell=n_neuron, sigma=0.3, std_frac=0.5, random_sigma=True)

# initialize ground truth data, example data, and training data
field_range = [-0.8, 0.8]
n_x_pos = 50 # number of points in each dimension, for noiseless data only
pos = label_mesh([field_range, field_range], n_x_pos, grid=True)

x_noiseless = pos; r_noiseless = pcm.point_on_manifold(x_noiseless) # ground truth data
eg_x = np.array([[0.5, 0.5], [-0.5, -0.5]]) # example data
eg_r = pcm.point_on_manifold(eg_x) # example data
eg_cov = pcm.get_covariance_matrix(eg_x) # example data
blanket_spacing = 0.2

x_each_dim = [np.arange(a, b + blanket_spacing, blanket_spacing) for a, b in [field_range, field_range]]
x_blanket = list(itertools.product(*x_each_dim))
x_train = np.random.uniform(*field_range, (200, 2));
x_train = np.concatenate([x_train, x_blanket], axis=0)

r_train = pcm.random_point(x_train) # get the training data

# initialize the figure
visualization_dim = 3
if visualization_dim == 3: visual_projection = '3d'
else: visual_projection = None

fig = plt.figure(figsize=(12, 4))
axes = [fig.add_subplot(1, 6, i+1, projection=visual_projection) for i in range(6)]
for ax in axes: ax.set_axis_off()
ax_true, ax_data, ax_gkr, ax_lw, ax_bin, ax_ggpr, = axes

dim_reduce = PCA(n_components=visualization_dim)
dim_reduce.fit(r_noiseless)

############# Show rate map
n_rate_map = 3
fig_rate_map, ax_rate_map = plt.subplots(n_rate_map, 1, figsize=(1, 2.5))
rnr = r_noiseless.reshape(n_x_pos, n_x_pos, r_noiseless.shape[-1])
for i, idx in enumerate([1, 2, 4]): # show three examples. I selected this because a few example units are too close to the boundary which are not representative
    ax = ax_rate_map[i]
    ax.imshow(rnr[:,:,idx], cmap='viridis', interpolation='nearest')
    ax.set_axis_off()
fig_rate_map.tight_layout()
fig_rate_map.savefig(os.path.join(FIGROOT, "rate_map_2d.svg"), bbox_inches='tight')

################ plot the surface
r_noiseless_reduced = dim_reduce.transform(r_noiseless).reshape(n_x_pos, n_x_pos, 3)
plot_scatter_or_line_by_dim(r_noiseless, x_noiseless[:, 0], ax_true, dim_reduce, plot_mode='surface', theta_range=field_range)
plot_cov_ellipse(eg_r, eg_cov, ax=ax_true, zo=2, pca=dim_reduce) # plot the cov ellipses
ax_true.set_title('Ground Truth')

# get the limit
zoom_factor = 0.8
xlim = np.array(ax_true.get_xlim()) * zoom_factor; ylim = np.array(ax_true.get_ylim()) * zoom_factor;
if visualization_dim == 3: zlim = np.array(ax_true.get_zlim()) * zoom_factor
else: zlim = None
set_ax_limits(ax_true, xlim, ylim, zlim, visualization_dim)

################ plot the training data
plot_scatter_or_line_by_dim(r_train, x_train[:, 0], ax_data, dim_reduce, visualize_dim=3, plot_mode='scatter', theta_range=field_range)
set_ax_limits(ax_data, xlim, ylim, zlim, visualization_dim)
ax_data.set_title('Data')

########## fit the ggpr
af = GGPR_Fitter()
fit_plot_manifold_figure(af, r_train, x_train, x_noiseless, eg_x, model_name='ggpr', fig=fig, ax=ax_ggpr, title='GGPR', visualization_dim=visualization_dim, pca=dim_reduce, theta_range=field_range, plot_mode='surface')
set_ax_limits(ax_ggpr, xlim, ylim, zlim, visualization_dim)

# ########## fit the gwpr
# af = GWPR_Fitter(max_iter=1000)
# fit_plot_manifold_figure(af, r_train, x_train, x_noiseless, eg_x, fig=fig, ax=ax_gwpr, model_name='ggpr', title='GWPR', visualization_dim=visualization_dim, pca=dim_reduce, theta_range=field_range, plot_mode='surface')
# set_ax_limits(ax_gwpr, xlim, ylim, zlim, visualization_dim)

########## fit the gkr
af = GKR_Fitter(n_input=2, n_output=n_neuron, circular_period=None)
fit_plot_manifold_figure(af, r_train, x_train, x_noiseless, eg_x, fig=fig, ax=ax_gkr, model_name='ggpr', title='GKR', visualization_dim=visualization_dim, pca=dim_reduce, theta_range=field_range, plot_mode='surface')
set_ax_limits(ax_gkr, xlim, ylim, zlim, visualization_dim)

# ########## fit the avg-fitter, kernel method
# af = Avg_Fitter(avg_method='kernel', bin_search=True, hyper_bound=[(0.02, 1.5), (0.02, 1.5)]) # lower hyper_bound to avoid too less data within a bin
# fit_plot_manifold_figure(af, r_train, x_train, x_noiseless, eg_x, fig=fig, ax=ax_kernel, model_name='avg', title='Kernel Average', visualization_dim=visualization_dim, pca=dim_reduce, theta_range=field_range, plot_mode='surface')
# set_ax_limits(ax_kernel, xlim, ylim, zlim, visualization_dim)

########## fit the avg-fitter, bin method
af = Avg_Fitter(avg_method='bin', bin_search=False, bin_size_mean=blanket_spacing, bin_size_cov=blanket_spacing) # lower hyper_bound to avoid too less data within a bin
fit_plot_manifold_figure(af, r_train, x_train, x_noiseless, eg_x, fig=fig, ax=ax_bin, model_name='avg', title='Bin Average', visualization_dim=visualization_dim, pca=dim_reduce, theta_range=field_range, plot_mode='surface')
set_ax_limits(ax_bin, xlim, ylim, zlim, visualization_dim)

########## fit the avg-fitter, lw method
af = Avg_Fitter(avg_method='bin', bin_search=False, bin_size_mean=blanket_spacing, bin_size_cov=blanket_spacing, use_bin_lw=True) # lower hyper_bound to avoid too less data within a bin
fit_plot_manifold_figure(af, r_train, x_train, x_noiseless, eg_x, fig=fig, ax=ax_lw, model_name='avg', title='LW', visualization_dim=visualization_dim, pca=dim_reduce, theta_range=field_range, plot_mode='surface')
set_ax_limits(ax_bin, xlim, ylim, zlim, visualization_dim)

fig.tight_layout()
fig.savefig(os.path.join(FIGROOT, "test_on_toy_dataset_2d.pdf"), bbox_inches='tight')
plt.show()
