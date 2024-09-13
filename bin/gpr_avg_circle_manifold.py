import numpy as np
import matplotlib.pyplot as plt
import os
from grid_cell.manifold_fitter import label_mesh
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from grid_cell.sphere_manifold import Circle_Manifold, get_circle_mesh
from grid_cell.ori_tuning_manifold import Ori_Tuning_Manifold, Place_Cell_Manifold
from grid_cell.manifold_fitter import Avg_Fitter
from global_setting import *
from grid_cell.manifold_visualizer import fig_ax_by_dim, plot_scatter_or_line_by_dim, fit_plot_manifold_figure, plot_cov_ellipse
from grid_cell.ggpr import GGPR_Fitter
from grid_cell.gwpr import GWPR_Fitter
from grid_cell.gkr import GKR_Fitter

def set_ax_limits(ax, xlim, ylim, zlim, visualization_dim=3):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if visualization_dim == 3: ax.set_zlim(zlim)
    return ax

########## Hyperparameters
### universal configuration
visualization_dim = 2
manifold_plot_mode = 'line'
pca = PCA(n_components=visualization_dim)
n_train_points = 100
# xlim, ylim, zlim = None, None, None
lim_half_width = 0.9
xlim, ylim, zlim = [-lim_half_width, lim_half_width], [-lim_half_width, lim_half_width], [-lim_half_width, lim_half_width]

#################### circle manifold ####################
# n_neuron = 2
# sigma = 0.15
# center = [100, 100]
# x_scale = 2
# toy_manifold = Circle_Manifold(sigma=sigma, center=center, x_scale=x_scale)

# # get the ground truth data
# theta_noiseless = np.linspace(0, 2 * np.pi, 100)[..., None] # noiseless data
# r_noiseless = toy_manifold.point_on_circle(theta_noiseless) # ground truth circle
# eg_theta = np.array([0, np.pi/4, np.pi/2, np.pi, 5*np.pi/4])[..., None] # example theta

# # draw the noisy data
# # theta = np.linspace(0, 2 * np.pi, 100)
# # theta = np.random.choice(theta, size=n_train_points, replace=True)[..., None]
# theta = np.random.beta(a=0.5, b=0.5, size=n_train_points) * (2 * np.pi)
# theta = theta[..., None]
# r = random_points = toy_manifold.random_point(theta)
# theta_train = theta; r_train = r
########################################

#################### ori_tuning manifold ####################
n_neuron = 10
toy_manifold = Ori_Tuning_Manifold(n_neuron, random_gain=True, std_frac=0.2, sigma=0.3)
# get the ground truth data
theta_noiseless = np.linspace(0, 2 * np.pi, 100)[..., None] # noiseless data
r_noiseless = toy_manifold.point_on_manifold(theta_noiseless) # ground truth circle
# eg_theta = np.array([0, np.pi/4, np.pi/2, np.pi, 5*np.pi/4])[..., None] # example theta
eg_theta = np.linspace(0, 2*np.pi, 20)[..., None] # example theta

# draw the noisy data
# blanket theta to make sure in each blanket_spacing bin there's at least on data point so that avg method can be appropriately computed
blanket_spacing = 0.2
theta_blanket = np.arange(0, 2 * np.pi + blanket_spacing, blanket_spacing).reshape(-1, 1)
theta = np.random.uniform(size=(n_train_points, 1)) * (2 * np.pi)
theta = np.concatenate([theta, theta_blanket], axis=0) # with blanket points
r = random_points = toy_manifold.random_point(theta)
theta_train = theta; r_train = r
########################################

#### Create the figure
if visualization_dim == 3: visual_projection = '3d'
else: visual_projection = None

fig = plt.figure(figsize=(12, 4))
axes = [fig.add_subplot(1, 6, i+1, projection=visual_projection) for i in range(6)]
for ax in axes: ax.set_axis_off()
ax_true, ax_data, ax_gkr, ax_ggpr, ax_bin, ax_lw = axes

#### draw the tuning curves
fig_tuning, ax_tuning = plt.subplots(1, 1, figsize=(3, 1))
for i_neuron in range(r_noiseless.shape[1]):
    ax_tuning.plot(theta_noiseless[:, 0], r_noiseless[:, i_neuron], color='k')
ax_tuning.set_title('Tuning Curves')
ax_tuning.set_xlabel(r'$\theta$'); ax_tuning.set_ylabel(r'Firing Rate')
fig_tuning.savefig(os.path.join(FIGROOT, 'tuning_curves_1d.svg'))

##### a few example points' covariance ellipses
eg_r = toy_manifold.point_on_manifold(eg_theta)
eg_cov = toy_manifold.get_covariance_matrix(eg_theta)

##### show the trainning data
pca.fit(r_train) # PCA is only for visualization, which is only fitted here so that the prediction from different methods can have a universal coordinate system, easy for visual comparison
plot_scatter_or_line_by_dim(r_train, theta_train[:, 0], ax_data, pca, plot_mode='scatter')
ax_data.set_aspect('equal')

# get the limit
xlim = ax_data.get_xlim(); ylim = ax_data.get_ylim();
if visualization_dim == 3: zlim = ax_data.get_zlim()
else: zlim = None

ax_data.set_axis_off()
ax_data.set_title('Training Data')

##### plot ground truth circle
plot_scatter_or_line_by_dim(r_noiseless, theta_noiseless[:, 0], ax_true, pca, plot_mode=manifold_plot_mode) # plot the circle
plot_cov_ellipse(eg_r, eg_cov, ax=ax_true, zo=2, pca=pca) # plot the cov ellipses
ax_true.set_aspect('equal')
set_ax_limits(ax_true, xlim, ylim, zlim, visualization_dim)
ax_true.set_axis_off()
ax_true.set_title('Ground Truth')

### fit the manifold
# using avg-fitter, avg method
af = Avg_Fitter(circular_period=2 * np.pi, avg_method='bin', bin_search=False, bin_size_mean=blanket_spacing, bin_size_cov=blanket_spacing)
fit_plot_manifold_figure(af, r_train, theta_train, theta_noiseless, eg_theta, fig=fig, ax=ax_bin, model_name='avg', title='Bin Average', visualization_dim=visualization_dim, pca=pca)
set_ax_limits(ax_bin, xlim, ylim, zlim, visualization_dim)

# using avg-fitter, lw
af = Avg_Fitter(circular_period=2 * np.pi, avg_method='bin', bin_search=False, bin_size_mean=blanket_spacing, bin_size_cov=blanket_spacing, use_bin_lw=True)
fit_plot_manifold_figure(af, r_train, theta_train, theta_noiseless, eg_theta, fig=fig, ax=ax_lw, model_name='avg', title='LW', visualization_dim=visualization_dim, pca=pca)
set_ax_limits(ax_bin, xlim, ylim, zlim, visualization_dim)

# use GGPR
af = GGPR_Fitter(circular_period=[2 * np.pi])
fit_plot_manifold_figure(af, r_train, theta_train, theta_noiseless, eg_theta, fig=fig, ax=ax_ggpr, model_name='gpr', title='GGPR', visualization_dim=visualization_dim, pca=pca)
set_ax_limits(ax_ggpr, xlim, ylim, zlim, visualization_dim)

# use GKR
af = GKR_Fitter(n_input=1, n_output=n_neuron, circular_period=2 * np.pi, gpr_params={'n_inducing': None})
fit_plot_manifold_figure(af, r_train, theta_train, theta_noiseless, eg_theta, fig=fig, ax=ax_gkr, model_name='gpr', title='GKR', visualization_dim=visualization_dim, pca=pca)
set_ax_limits(ax_gkr, xlim, ylim, zlim, visualization_dim)

fig.tight_layout()
fig.savefig(os.path.join(FIGROOT, "test_on_toy_dataset_1d.svg"), bbox_inches='tight')

plt.show()
