import os
from grid_cell.gkr_fitting_helper import load_and_process_data, fit_gkr_models
import hickle as hkl
import numpy as np
from global_setting import *

# # dataset_name = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
dataset_name = ['r1m1']
speed_bound = [0.05, 0.45]
speed_bins = np.arange(speed_bound[0], speed_bound[1] + 0.001, 0.05) # include the upper bound
preprocessed_file_name = 'preprocessed_data_OF.hkl'
n_sample_data = 20000
gpr_params = {'n_inducing': 200, 'standardize': True}

# Use the refactored functions
processed_data = load_and_process_data(
    dataset_name=dataset_name,
    preprocessed_file_name=preprocessed_file_name,
    n_sample_data=n_sample_data,
    pca_components=6,
    speed_min=speed_bound[0], 
    speed_max=speed_bound[1]
)

gkr_models = fit_gkr_models(
    processed_data=processed_data,
    gpr_params=gpr_params,
    n_bootstrap=10,
    gkr_n_epochs=20,
    speed_bins=speed_bins
)

model_data = {'model_list': gkr_models, 'speed_bins': speed_bins}
hkl.dump(model_data, os.path.join(DATAROOT, f'gkr_speed_box.hkl'))
