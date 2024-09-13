# compute the geometry of covariance matrix. Run fit_GKR_models.py first.
import os
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
from grid_cell.manifold_fitter import label_mesh
import grid_cell.ploter as ploter
import grid_cell.torus_geometry as tg
from grid_cell.util import remove_outliers, clean_and_mean_se, gram_schmidt, project_to_orthogonal_subspace, angle_two_vec
from global_setting import *

def mean_only_decorator(eg_model):
    def wrapper(*args):
        f, _ = eg_model.predict(*args)
        return f
    return wrapper
#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '1'
session = 'open_field_1'
pca_component = 6
downsample_rate_plot = 3

gkr_file_name = 'gkr_{}_{}_{}_{}_pca{}_dr{}.hkl'.format(mouse_name, module, day, session, str(pca_component), str(downsample_rate_plot))
model_path = os.path.join(DATAROOT, gkr_file_name)
model = hkl.load(model_path)

mesh_size = 300
x_bound, y_bound, speed_bound = [-0.75, 0.75], [-0.75, 0.75], [0.05, 0.60]

# get random mesh of points
query_mesh_space = label_mesh([x_bound, y_bound], mesh_size=mesh_size, grid=False, random=True)

def create_fitter_wrapper(model):
    def fitter_wrapper(query_mesh):
        return model.predict(query_mesh)[0]
    return fitter_wrapper

speed_list = np.arange(speed_bound[0], speed_bound[1], 0.05)
tot_noise_list = []
noise_proj_list = []
noise_ratio_list = []
fisher_list = []
for speed_value in speed_list:
    query_mesh = np.concatenate([query_mesh_space, np.ones([mesh_size, 1]) * speed_value], axis=1)

    _, cov = model.predict(query_mesh)

    model_wrap = create_fitter_wrapper(model)
    # compute the Jacobian matrix of these points
    J = tg.compute_jacobian_central(model_wrap, query_mesh, h=0.1)

    # total noise
    trace = np.trace(cov, axis1=1, axis2=2)
    tot_noise = trace

    # # noise projection on the tangent plane
    noise_proj = tg.project_noise_to_plane(cov, J[:, :, 0], J[:, :, 1])
    noise_proj = np.trace(noise_proj, axis1=1, axis2=2)

    noise_ratio = noise_proj / tot_noise

    # fisher information
    prec = np.linalg.inv(cov)
    fisher = tg.compute_fisher_info(J, prec)
    tot_fisher = np.trace(fisher, axis1=1, axis2=2)

    tot_noise_list.append(np.mean(tot_noise))
    noise_proj_list.append(noise_proj.mean())
    noise_ratio_list.append(noise_ratio.mean())
    fisher_list.append(tot_fisher.mean())

data = {'speed': np.array(speed_list), 'total_noise': np.array(tot_noise_list), 'noise_proj': np.array(noise_proj_list), 'noise_ratio': np.array(noise_ratio_list), 'fisher': np.array(fisher_list)}
hkl.dump(data, os.path.join(DATAROOT, 'noise_geo_data.hkl'))

data = hkl.load(os.path.join(DATAROOT, 'noise_geo_data.hkl'))
fig, ax = plt.subplots(figsize=(3, 3))
ax.plot(data['speed'] * 100, data['total_noise'], marker='o', color='k') # convert to cm/s
ax.set_xlabel('Speed (cm/s)')
ax.set_ylabel('Total Noise ' + r'(Hz$^2$)')
fig.savefig(os.path.join(FIGROOT, 'total_noise.svg'))

fig, ax = plt.subplots(figsize=(3, 3))
ax.plot(data['speed'] * 100, data['noise_proj'], label='Noise Projection', marker='o', color='k') # convert to cm/s
ax.set_xlabel('Speed (cm/s)')
ax.set_ylabel('Noise Projection ' + r'(Hz$^2$)')
fig.savefig(os.path.join(FIGROOT, 'proj_noise.svg'))

fig, ax = plt.subplots(figsize=(3, 3))
ax.plot(data['speed'] * 100, data['fisher'] / 10000, label='Total Fisher Info', marker='o', color='k') # convert to cm/s and 1 / cm^2
ax.set_xlabel('Speed (cm/s)')
ax.set_ylabel('Total Fisher Info ' + r'(cm$^{-2}$)')
fig.savefig(os.path.join(FIGROOT, 'fisher_info.svg'))
plt.show()
