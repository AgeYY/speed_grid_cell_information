import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from grid_cell.sphere_manifold import Circle_Manifold, get_circle_mesh, plot_cov_ellipse
from grid_cell.gpr_covnet import GPR_CovNet
from grid_cell.manifold_fitter import Avg_Fitter
from grid_cell.ori_tuning_manifold import Ori_Tuning_Manifold
from global_setting import *

def fit_plot_manifold_figure(af, r_train, theta_train, theta_noiseless, eg_theta, xlim, ylim, fig=None, ax=None, model_name='avg', title=None):
    if fig is None: fig, ax = plt.subplots(figsize=(3, 3))

    af.fit(r_train, theta_train[..., np.newaxis])

    if model_name == 'avg':
        r_pred, qm_valid, _, _ = af.predict_1d(theta_noiseless[..., np.newaxis], return_cov=True)
    elif model_name == 'gpr':
        r_pred, _ = model_gpr.predict(theta_noiseless[..., np.newaxis])
        qm_valid = theta_noiseless[..., np.newaxis]

    colors = plt.cm.hsv(qm_valid / (2 * np.pi))
    ax.set_aspect('equal')
    for i in range(len(qm_valid) - 1):
        ax.plot(r_pred[i:i+2, 0], r_pred[i:i+2, 1], color=colors[i], linewidth=4)
    ax.plot(r_pred[-1:, 0], r_pred[:1, 1], color=colors[-1], linewidth=4)

    if model_name == 'avg':
        eg_r, qm_eg, eg_cov, _ = af.predict_1d(eg_theta[..., np.newaxis], return_cov=True)
    elif model_name == 'gpr':
        eg_r, eg_cov = model_gpr.predict(eg_theta[..., np.newaxis])

    plot_cov_ellipse(eg_r, eg_cov, ax=ax, zo=2)

    ax.set_axis_off()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)

    return fig, ax

########## Hyperparameters
sigma = 0.15
center = [100, 100]
x_scale = 2
cm = Circle_Manifold(sigma=sigma, center=center, x_scale=x_scale)
# xlim, ylim = [-2.0, 2.0], [-2.0, 2.0]
xlim, ylim = None, None

theta_noiseless = np.linspace(0, 2 * np.pi, 100) # noiseless data
r_noiseless = cm.point_on_circle(theta_noiseless).T # ground truth circle

eg_theta = np.array([0, np.pi/4, np.pi/2, np.pi, 5*np.pi/4]) # example theta

# draw the noisy data
n_points = 200

theta = np.linspace(0, 2 * np.pi, 100)
theta = np.random.choice(theta, size=n_points, replace=True)
r = random_points = cm.random_point(theta)
theta_train = theta; r_train = r

# split data to train and test

#### draw the noiseless circle with covariance ellipse
# the manifold
# a few example points' covariance ellipses
eg_r = cm.point_on_circle(eg_theta).T
eg_cov = cm.get_covariance_matrix(eg_theta)

# plot ground truth circle
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_aspect('equal')
# plot circle
colors = plt.cm.hsv(theta_noiseless / (2 * np.pi))
for i in range(len(theta_noiseless) - 1):
    ax.plot(r_noiseless[i:i+2, 0], r_noiseless[i:i+2, 1], color=colors[i], linewidth=4)
# plot the cov ellipses
plot_cov_ellipse(eg_r, eg_cov, ax=ax, zo=2)
ax.set_axis_off()
ax.set_xlim(xlim); ax.set_ylim(ylim)
fig.savefig(os.path.join(FIGROOT, "circle_manifold.svg"), bbox_inches='tight')

# show the trainning data
fig, ax_data = plt.subplots(figsize=(3, 3))
ax_data.set_aspect('equal')
ax_data.scatter(r_train[:, 0], r_train[:, 1], alpha=0.6, s=10, c=theta_train, cmap='hsv')
ax_data.set_xlim(xlim)
ax_data.set_ylim(ylim)
ax_data.set_axis_off()
fig.savefig(os.path.join(FIGROOT, "circle_manifold_noisy.svg"), bbox_inches='tight')

### fit the manifold
# using gpr-covnet
model_gpr = GPR_CovNet(epochs=500, batch_size=9999, with_cov=True, circular_period=np.pi*2, standardize=True)
fig_gpr, ax_gpr = plt.subplots(figsize=(3, 3))
fig_gpr, ax_gpr = fit_plot_manifold_figure(model_gpr, r_train, theta_train, theta_noiseless, eg_theta, xlim, ylim, fig=fig_gpr, ax=ax_gpr, model_name='gpr', title='GPR-CovNet')
fig_gpr.savefig(os.path.join(FIGROOT, "circle_manifold_fit_gpr.svg"), bbox_inches='tight')

# using avg-fitter, kernel method
af = Avg_Fitter(circular_period=2 * np.pi, avg_method='kernel', bin_search=True)
fig_avg, ax_avg = plt.subplots(figsize=(3, 3))
fig_avg, ax_avg = fit_plot_manifold_figure(af, r_train, theta_train, theta_noiseless, eg_theta, xlim, ylim, fig=fig_avg, ax=ax_avg, model_name='avg', title='Kernel Average')
fig_avg.savefig(os.path.join(FIGROOT, "circle_manifold_fit_avg_kernel.svg"), bbox_inches='tight')

# using avg-fitter, bin method
af = Avg_Fitter(circular_period=2 * np.pi, avg_method='bin', bin_search=True)
fig_avg, ax_avg = plt.subplots(figsize=(3, 3))
fig_avg, ax_avg = fit_plot_manifold_figure(af, r_train, theta_train, theta_noiseless, eg_theta, xlim, ylim, fig=fig_avg, ax=ax_avg, model_name='avg', title='Bin Average')
fig_avg.savefig(os.path.join(FIGROOT, "circle_manifold_fit_avg_bin.svg"), bbox_inches='tight')

plt.show()
