import os
from grid_cell.gkr_fitting_helper import load_and_process_data, fit_gkr_models
import hickle as hkl
import numpy as np
from global_setting import *

dataset_name = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
preprocessed_file_name = 'preprocessed_data_OF.hkl'
n_sample_data = 10000
n_bootstrap = 50
gkr_n_epoch = 30
speed_min = 0.05
speed_max = 0.45
gpr_params = {'n_inducing': 200, 'standardize': True}

# Use the refactored functions
processed_data = load_and_process_data(
    dataset_name=dataset_name,
    preprocessed_file_name=preprocessed_file_name,
    n_sample_data=n_sample_data,
    speed_min = speed_min,
    speed_max = speed_max,
    pca_components=None
)

# fit gkr
gkr_models = fit_gkr_models(
    processed_data=processed_data,
    gpr_params=gpr_params,
    n_bootstrap=n_bootstrap,
    gkr_n_epochs=gkr_n_epoch, # no fitting covariance
)
for dn in dataset_name:
    hkl.dump(gkr_models[dn], os.path.join(DATAROOT, f'gkr_models_{dn}_pcaNone.hkl'))

# fit shuffle gkr
for dn in dataset_name:
    processed_data[dn]['label'] = np.random.permutation(processed_data[dn]['label'])

gkr_models_shuffle = fit_gkr_models(
    processed_data=processed_data,
    gpr_params=gpr_params,
    n_bootstrap=n_bootstrap,
    gkr_n_epochs=gkr_n_epoch, # no fitting covariance
)
for dn in dataset_name:
    hkl.dump(gkr_models_shuffle[dn], os.path.join(DATAROOT, f'shuffle_gkr_models_{dn}_pcaNone.hkl'))

##################################################
########## Repeat the above code, but using pca = 6
##################################################
n_pca = 6

processed_data = load_and_process_data(
    dataset_name=dataset_name,
    preprocessed_file_name=preprocessed_file_name,
    n_sample_data=n_sample_data,
    speed_min = 0.05,
    speed_max = 0.45,
    pca_components=n_pca
)

# fit gkr
gkr_models = fit_gkr_models(
    processed_data=processed_data,
    gpr_params=gpr_params,
    n_bootstrap=n_bootstrap,
    gkr_n_epochs=gkr_n_epoch, # no fitting covariance
)
for dn in dataset_name:
    hkl.dump(gkr_models[dn], os.path.join(DATAROOT, f'gkr_models_{dn}_pca{n_pca}.hkl'))

# fit shuffle gkr
for dn in dataset_name:
    processed_data[dn]['label'] = np.random.permutation(processed_data[dn]['label'])

gkr_models_shuffle = fit_gkr_models(
    processed_data=processed_data,
    gpr_params=gpr_params,
    n_bootstrap=n_bootstrap,
    gkr_n_epochs=gkr_n_epoch, # no fitting covariance
)
for dn in dataset_name:
    hkl.dump(gkr_models_shuffle[dn], os.path.join(DATAROOT, f'shuffle_gkr_models_{dn}_pca{n_pca}.hkl'))
