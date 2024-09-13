# please run fit_torus_speed_cov.py first to obtain the g-covnet model
import os
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
from grid_cell.gpr_covnet import GPR_CovNet, save_gpr_covnet_all, load_gpr_covnet_all
from grid_cell.manifold_fitter import label_mesh
import grid_cell.ploter as ploter
import grid_cell.torus_geometry as tg
from grid_cell.util import remove_outliers, clean_and_mean_se
from global_setting import *

def mean_only_decorator(eg_model):
    def wrapper(*args):
        f, _ = eg_model.predict(*args)
        return f
    return wrapper
#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'

n_bins = 10
x_bound, y_bound = [-0.75, 0.75], [-0.75, 0.75]

speed_delta = 0.025
speed_win = [[i, i + speed_delta] for i in np.arange(speed_delta, 0.5, speed_delta)]

gpr_dir = os.path.join(DATAROOT, 'short_long_gpr_covnet_torus_speed_{}_{}_{}_{}/'.format(mouse_name, module, day, session))
data_path = os.path.join(DATAROOT, 'short_long_torus_geo_data_{}_{}_{}_{}.hkl'.format(mouse_name, module, day, session))
fig_dir = os.path.join(FIGROOT, 'shuffle_torus_geo_fig_{}_{}_{}_{}/'.format(mouse_name, module, day, session))
if not os.path.exists(fig_dir): os.makedirs(fig_dir)
#################### Main ####################
query_mesh = label_mesh([x_bound, y_bound], mesh_size=n_bins, grid=True)

data = load_gpr_covnet_all(gpr_covnet_dir)

speed_win, result = data['speed_win'], data['result']

n_speed, n_boot = len(result), len(result[0])

tot_noise = np.empty((n_speed, n_boot))
tot_noise_proj = np.empty((n_speed, n_boot))
noise_ratio = np.empty((n_speed, n_boot))
lattice_area = np.empty((n_speed, n_boot))
radius = np.empty((n_speed, n_boot))
noise_lattice_ratio = np.empty((n_speed, n_boot))
tot_fisher = np.empty((n_speed, n_boot))
tot_fisher_unit = np.empty((n_speed, n_boot))

for i in range(n_speed):
    for j in range(n_boot):
        model = result[i][j]
        f, cov = model.predict(query_mesh)
        _, prec = model.predict(query_mesh, cov_to_prec=True)

        mean_wrapper = mean_only_decorator(model)
        J = tg.compute_jacobian_central(mean_wrapper, query_mesh, 1.5/n_bins)

        # total noise
        trace = np.trace(cov, axis1=1, axis2=2)
        tot_noise_boot = np.mean(trace)
        tot_noise[i, j] = tot_noise_boot

        # # noise projection on the tangent plane
        noise_proj = tg.project_noise_to_plane(cov, J[:, :, 0], J[:, :, 1])
        tot_noise_proj[i, j] = np.trace(noise_proj, axis1=1, axis2=2).mean()

        # ratio of projected noise to total noise
        noise_ratio[i, j] = tot_noise_proj[i, j] / tot_noise[i, j]

        # lattice area
        lattice_area[i, j] = tg.compute_lattice_area(J).mean()

        # radius
        radius_boot = tg.torus_avg_radius(f)
        radius[i, j] = radius_boot

        # projected noise to lattice area ratio
        noise_lattice_ratio[i, j] = tot_noise_proj[i, j] / lattice_area[i, j]

        # fisher information
        fisher = tg.compute_fisher_info(J, prec)
        tot_fisher[i, j] = np.trace(fisher, axis1=1, axis2=2).mean()

        # fisher information for each axis
        n_neuron = f.shape[1]
        fisher_avg_units = []
        for unit_id in range(n_neuron):
            vec = np.zeros((f.shape[0], n_neuron))
            vec[:, unit_id] = 1
            fisher_unit = tg.compute_fisher_one_vec(vec, J, cov)
            fisher_unit = np.trace(fisher_unit, axis1=1, axis2=2).mean()
            fisher_avg_units.append(fisher_unit)
        tot_fisher_unit[i, j] = np.mean(fisher_avg_units)

var_list_name = ['tot_noise', 'tot_noise_proj', 'noise_ratio', 'lattice_area', 'radius', 'noise_lattice_ratio', 'tot_fisher', 'tot_fisher_unit']
var_list = [tot_noise, tot_noise_proj, noise_ratio, lattice_area, radius, noise_lattice_ratio, tot_fisher, tot_fisher_unit]
speed_var = [sp[0] for sp in speed_win]

data = {'speed_var': speed_var, 'var_list_name': var_list_name, 'var_list': var_list}
hkl.dump(data, data_path)
# exit()

data = hkl.load(data_path)
speed_var, var_list_name, var_list = data['speed_var'], data['var_list_name'], data['var_list']
ylabel = ['Total noise', 'Projected noise', 'Projected / Tot. Noise ratio \n (a.u.)', 'Lattice area', 'Torus Radius (a.u.)', 'Noise/lattice ratio', 'Total Fisher', 'Fisher per unit']

for i, (var_name, var) in enumerate(zip(var_list_name, var_list)):
    if var_name == 'noise_ratio':
        fig, ax = plt.subplots(figsize=(3, 2.5))
    else:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
    fig, ax = ploter.error_bar_plot(speed_var, var, mean_mode='median', error_mode='quantile', fig=fig, ax=ax)
    # ax.set_title(var_name)
    ax.set_xlabel('Speed (m/s)')
    ax.set_ylabel(ylabel[i])
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, '{}.svg'.format(var_name)), format='svg')

# draw the total noise, noise projected on the tangent plane and 
fig, ax = plt.subplots(figsize=(2.8, 2.8))
fig, ax = ploter.error_bar_plot(speed_var, var_list[0], mean_mode='median', error_mode='quantile', fig=fig, ax=ax, label='Total noise', color='tab:blue')
fig, ax = ploter.error_bar_plot(speed_var, var_list[1], mean_mode='median', error_mode='quantile', fig=fig, ax=ax, label='Projected noise', color='tab:orange')
ax.set_xlabel('Speed (m/s)')
ax.set_ylabel('Noise (a.u.)')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'noise.svg'), format='svg')

# the ratio of projected noise to total noise
plt.show()
