# compute the speed modulation of each cell, by fitting a line between firing rate and speed
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from grid_cell.grid_cell_processor import Grid_Cell_Processor, Speed_Processor, Data_Transformer
from grid_cell.lasso_evaluation import lasso_evaluate_multiple_label
import grid_cell.util as util
import hickle as hkl
import os
from global_setting import *

#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'

gridness_thre = 0.3
subset_size = 25000 # m number of samples to subset
sub_sample_size = 2500 # m number of samples for each bootrap
n_bootstrap = 10 # Number of repeated R2 score evaluations
fit_label_id = [0, 1, 3] # remove time label
n_component = 15
data_preprocess_para_data = [{'transform_method': 'data', 'n_component': None}]
data_preprocess_para_pca = [{'transform_method': 'pca', 'n_component': i} for i in range(1, n_component)]
data_preprocess_para_pls = [{'transform_method': 'pls', 'n_component': i} for i in range(1, n_component)]
data_preprocess_para = data_preprocess_para_data + data_preprocess_para_pca + data_preprocess_para_pls

preprocessed_data_with_speed_path = os.path.join(DATAROOT, 'preprocessed_data_with_speed.hkl')
# speed_temporal_file_name = 'speed_temporal_{}_{}_{}_{}.npz'.format(mouse_name, module, day, session)
# speed_temporal_path = os.path.join(DATAROOT, speed_temporal_file_name)

#################### Main ####################
# # generate preprocessed data with speed label
# gcp = Grid_Cell_Processor()
# gcp.load_data(mouse_name, day, module, session)
# fire_rate, x, y, t, speed = gcp.preprocess(gridness_thre=gridness_thre, pca_components=None, use_zscore=True, return_speed=True)
# label = np.array([x, y, t, speed]).T
# data = {'fire_rate': fire_rate, 'label': label}
# hkl.dump(data, preprocessed_data_with_speed_path)
# exit()

data = hkl.load(preprocessed_data_with_speed_path)
fire_rate, label = data['fire_rate'], data['label']
label = label[:, [0, 1, 3]] # remove time

sdt = Data_Transformer() # transform by none, pca, pls
sdt.load_data(fire_rate, label)

r2_table = {'transform_method': [], 'n_component': []}
for i in range(len(fit_label_id)): r2_table['label' + str(i)] = []

### compute R2 score for each label and each method
for para in data_preprocess_para:
    fire_rate_trans, label_trans = sdt.subset_transform(fit_subset_size=subset_size, test_subset_size=subset_size, **para)
    r2_list = lasso_evaluate_multiple_label(fire_rate_trans, label_trans, n_bootstrap=n_bootstrap)
    r2_table['transform_method'].append(para['transform_method'])
    r2_table['n_component'].append(para['n_component'])
    [r2_table[key].append(r2_list[key].copy()) for key in r2_list.keys()]

r2_table = pd.DataFrame(r2_table)

# reshape the r2 score into plotable format
for i in range(len(fit_label_id)):
    r2_table['label' + str(i) + 'mean'] = r2_table['label' + str(i)].apply(np.mean)
    r2_table['label' + str(i) + 'std'] = r2_table['label' + str(i)].apply(np.std)

grouped = r2_table.groupby('transform_method')
n_components = grouped['n_component'].apply(list).to_dict()
label_mean, label_std = {}, {}
for i in range(len(fit_label_id)):
    label_mean[i] = grouped['label' + str(i) + 'mean'].apply(list).to_dict()
    label_std[i] = grouped['label' + str(i) + 'std'].apply(list).to_dict()

# plot
title_name = ['x position', 'y position', 'speed']
fig, ax = plt.subplots(1, len(fit_label_id), figsize=(8, 3))
for i in range(len(fit_label_id)):
    ax[i].errorbar(n_components['pca'], label_mean[i]['pca'], yerr=label_std[i]['pca'], fmt='o', linestyle='-', label='pca', color='tab:blue', ecolor='tab:blue')
    ax[i].errorbar(n_components['pls'], label_mean[i]['pls'], yerr=label_std[i]['pls'], fmt='o', linestyle='-', label='pls', color='tab:red', ecolor='tab:red')

    x_range = [0, n_component - 1]
    ax[i] = util.plot_horizontal_line_with_error_band(x_range, label_mean[i]['data'][0], label_std[i]['data'][0], ax[i], label='data')

    ax[i].set_xlabel('n_component')
    ax[i].set_title(title_name[i])

ax[0].set_ylabel('LASSO Regression \n R2 score')
ax[0].legend()

fig.tight_layout()
plt.show()
