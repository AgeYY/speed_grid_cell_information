# # measure the localling decoding performance on different data. Run generate_processed_exp_data.py first
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from grid_cell.linear_regression import fit_a_line, output_params, draw_line
import hickle as hkl
import os
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from grid_cell.manifold_fitter import label_mesh
from sklearn.model_selection import train_test_split
from sklearn.linear_model import MultiTaskLasso, LinearRegression
from grid_cell.grid_cell_processor import Speed_Processor
from sklearn.preprocessing import StandardScaler
from grid_cell.linear_regression import BayesianLinearRegression
from sklearn.pipeline import Pipeline
from grid_cell.lole import LOLE
from global_setting import *

def compute_mse_lole(label, feamap, x_bound, y_bound, speed_bound, model=MultiTaskLasso(alpha=0.1)):
    lole = LOLE(model)
    speed_list = np.arange(speed_bound[0], speed_bound[1], 0.05)
    mse_list = []
    for speed_value in speed_list:
        query_mesh_space = label_mesh([x_bound, y_bound], mesh_size=100, grid=False, random=True)
        query_mesh = np.concatenate([query_mesh_space, np.ones((query_mesh_space.shape[0], 1)) * speed_value], axis=1)
        mse, _ = lole.calculate_lole_mse(label, feamap, query_mesh)
        mse_list.append(mse)
    return speed_list, mse_list

#################### Hyperparameters ####################
x_bound, y_bound, speed_bound = [-0.75, 0.75], [-0.75, 0.75], [0.0, 0.57]
preprocessed_file_name = 'preprocessed_data_OF.hkl'
#################### Main ####################
# dataset_name = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
dataset_name = ['r2m3']
n_sample_data = 2500
data = hkl.load(os.path.join(DATAROOT, preprocessed_file_name))

lole_data = {}

for dn in dataset_name:
    lole_data[dn] = {}
    lole_data[dn]['speed_list'], lole_data[dn]['mse_list'], lole_data[dn]['shuffle_mse_list'] = [], [], []

    feamap = data[dn]['feamap']
    label = data[dn]['label']

    sp = Speed_Processor()
    sp.load_data(feamap, label)

    shuffle_label = np.random.permutation(label)
    shuffle_sp = Speed_Processor()
    shuffle_sp.load_data(feamap, shuffle_label)

    for _ in range(10):
        feamap, label = sp.sample_data(n_sample_data=n_sample_data, speed_min=0.05, replace=True, n_random_projection=15)
        label = label[:, [0, 1, 3]] # drop time
        speed_temp, mse_temp = compute_mse_lole(label, feamap, x_bound, y_bound, speed_bound)

        feamap, s_label = shuffle_sp.sample_data(n_sample_data=n_sample_data)
        s_label = s_label[:, [0, 1, 3]] # drop time
        speed_temp, s_mse_temp = compute_mse_lole(s_label, feamap, x_bound, y_bound, speed_bound)

        lole_data[dn]['speed_list'].append(speed_temp)
        lole_data[dn]['mse_list'].append(mse_temp)
        lole_data[dn]['shuffle_mse_list'].append(s_mse_temp)

    lole_data[dn]['speed_list'] = np.array(lole_data[dn]['speed_list'])
    lole_data[dn]['mse_list'] = np.array(lole_data[dn]['mse_list'])
    lole_data[dn]['shuffle_mse_list'] = np.array(lole_data[dn]['shuffle_mse_list'])

hkl.dump(lole_data, os.path.join(DATAROOT, 'lole_data.hkl'))

lole_data = hkl.load(os.path.join(DATAROOT, 'lole_data.hkl'))

# fit a line on the speed-mse data
dn = dataset_name[0]
x, y = lole_data[dn]['speed_list'][:, 1:].flatten() * 100, lole_data[dn]['mse_list'][:, 1:].flatten() * 10000 # 2: means start from speed = 10 cm
print(y.shape, x.shape)
model = fit_a_line(x, y)
slope, intercept, slope_conf_int, r2, p_value_intercept, p_value_slope = output_params(model)
print(f'R-Squared: {model.r_squared:.2f}')
print(f'p-value: {model.p_values[0]:.5f}, {model.p_values[1]:.5f}')

x, shuffle_y = lole_data[dn]['speed_list'][:, 1:].flatten() * 100, lole_data[dn]['shuffle_mse_list'][:, 1:].flatten() * 10000
shuffle_model = fit_a_line(x, shuffle_y)
slope, intercept, slope_conf_int, r2, p_value_intercept, p_value_slope = output_params(shuffle_model)
print(f'R-Squared shuffle: {shuffle_model.r_squared:.2f}')
print(f'p-value shuffle: {shuffle_model.p_values[0]:.5f}, {shuffle_model.p_values[1]:.5f}')

fig, ax = plt.subplots(figsize=(3, 3))
draw_line(ax, x, y, model, color='tab:blue', line_label=None, data_label='Data')
draw_line(ax, x, shuffle_y, shuffle_model, color='tab:grey', line_label=None, data_label='Shuffled Data')
ax.set_xlabel('Speed (cm/s)')
ax.set_ylabel('Decoding MSE of LOLE ' + r'($cm^2$)')
ax.legend()
fig.savefig(os.path.join(FIGROOT, 'lole_speed_mse.svg'))
plt.show()
