import numpy as np
from grid_cell.persistent_homology import Is_Torus
from grid_cell.manifold_fitter import GP_Fitter, Avg_Fitter
import matplotlib.pyplot as plt
import os
from sklearn import datasets
from global_setting import *

#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'
pca_str = '6' # none or number as a string

preprocessed_file_name = 'preprocessed_data_{}_{}_{}_{}_pca{}.npz'.format(mouse_name, module, day, session, pca_str)
preprocessed_dir = os.path.join(DATAROOT, 'preprocessed/')
if not os.path.exists(preprocessed_dir): os.makedirs(preprocessed_dir)
preprocessed_path = os.path.join(preprocessed_dir, preprocessed_file_name)

n_bins, x_bound, y_bound = 50, [-0.75, 0.75], [-0.75, 0.75]

maxdim = 1 # maximum homology dimension computed
coeff = 47 # prime coefficient for constructing the coefficient field
replace_inf = 20
plot_prcnt = 80
n_shuffle_barcode = 1 # number of shuffles for barcode
quantile = -1

n_boot = 2 # number of bootstrap to compute the probability of sucessing find a torus
n_time_point_list = np.arange(100, 300, 100)

it_timepoint_file_name = 'it_timepoint_{}_{}_{}_{}_pca{}.npz'.format(mouse_name, module, day, session, pca_str)
it_timepoint_path = os.path.join(DATAROOT, it_timepoint_file_name)

#################### Main ####################
# load data
data = np.load(preprocessed_path, allow_pickle=True)
x, y, dt = data['x'], data['y'], data['dt']
label = np.array([x, y]).T
feamap = data['fire_rate']

# set up Is_Torus and fitter
itoruser = Is_Torus(n_bins=n_bins, x_bound=x_bound, y_bound=y_bound, coeff=coeff, n_shuffle=n_shuffle_barcode, quantile=quantile)

def get_is_torus_timepoint_list(fitter, n_time_point):
    itoruser.set_fitter(fitter)
    it_list_temp = []
    for _ in range(n_boot):
        selected_rows = np.random.choice(feamap.shape[0], n_time_point, replace=False)
        feamap_boot, label_boot = feamap[selected_rows, :], label[selected_rows, :] # randomly sample time points
        it_list_temp.append(itoruser.is_torus(feamap_boot, label_boot))
    return it_list_temp

it_list_gpr, it_list_avg, it_list_none = [], [], []
for n_time_point in n_time_point_list:
    it_list_gpr_temp = get_is_torus_timepoint_list(GP_Fitter(), n_time_point)
    it_list_gpr.append(it_list_gpr_temp)

    it_list_avg_temp = get_is_torus_timepoint_list(Avg_Fitter(), n_time_point)
    it_list_avg.append(it_list_avg_temp)

    it_list_none_temp = get_is_torus_timepoint_list(None, n_time_point)
    it_list_none.append(it_list_none_temp)

np.savez(it_timepoint_path, it_list_gpr=it_list_gpr, it_list_avg=it_list_avg, it_list_none=it_list_none, n_time_point_list=n_time_point_list)
