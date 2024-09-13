import numpy as np
import hickle as hkl
from grid_cell.ploter import error_bar_plot
import scipy.ndimage as ndi
from grid_cell.tuning import ppdata_to_avg_ratemap
from scipy import stats
import matplotlib.pyplot as plt
import os
from global_setting import *

#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'
pca_str = '6' # none or number as a string

it_gp_timepoint_file_name = 'it_timepoint_gp_{}_{}_{}_{}_pca{}.hkl'.format(mouse_name, module, day, session, pca_str)
it_gp_timepoint_path = os.path.join(DATAROOT, it_gp_timepoint_file_name)
it_avg_timepoint_file_name = 'it_timepoint_avg_{}_{}_{}_{}_pca{}.hkl'.format(mouse_name, module, day, session, pca_str)
it_avg_timepoint_path = os.path.join(DATAROOT, it_avg_timepoint_file_name)
preprocessed_file_name = 'preprocessed_data_{}_{}_{}_{}_pca{}.npz'.format(mouse_name, module, day, session, pca_str)
preprocessed_dir = os.path.join(DATAROOT, 'preprocessed/')
preprocessed_path = os.path.join(preprocessed_dir, preprocessed_file_name)

corr_file_name = 'corr_data_{}_{}_{}_{}_pca{}.npz'.format(mouse_name, module, day, session, pca_str)
corr_path = os.path.join(DATAROOT, corr_file_name)

n_boots = 20
#################### Main ####################
### Get the approximate grouped truth
data = np.load(preprocessed_path, allow_pickle=True)
x, y, dt = data['x'], data['y'], data['dt']
label = np.array([x, y]).T
feamap = data['fire_rate']
n_pc = feamap.shape[1]

true_ratemap = ppdata_to_avg_ratemap(None, feamap, label)
true_ratemap = true_ratemap.flatten()
true_ratemap = np.nan_to_num(true_ratemap, nan=0)

### Get the GP data
data_gp = hkl.load(it_gp_timepoint_path)

### compute the correlation between gp and truth
correlations = {}
for n, mls in zip(data_gp['n_time_point_list'], data_gp['manifold_list_gp']):
    correlations[n] = []
    for ml in mls:
        correlations[n].append(stats.pearsonr(true_ratemap, ml.flat)[0])

x = np.array(list(correlations.keys()))
y = np.array(list(correlations.values()))

corr_gp = {'n_time_point_list_gp': x, 'correlations_gp': y}

### get the avg data and compute correlation
data_avg = hkl.load(it_avg_timepoint_path)
nt_avg = data_avg['n_time_point_list']

x = []
y = []
for nt in nt_avg:
    print('working on number of time points = {}'.format(nt))
    y.append([])
    x.append(nt)
    for _ in range(n_boots):
        boot_ratemap = ppdata_to_avg_ratemap(nt, feamap, label)
        boot_ratemap = np.nan_to_num(boot_ratemap, nan=0)
        boot_ratemap_flat = boot_ratemap.flatten()
        y[-1].append( stats.pearsonr(boot_ratemap_flat, true_ratemap)[0] )

corr_avg = {'n_time_point_list_avg': x, 'correlations_avg': y}

corr_data = {**corr_gp, **corr_avg}
np.savez(corr_path, **corr_data)
