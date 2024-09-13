# import cProfile, pstats
# pr = cProfile.Profile()
# pr.enable()

# test the performance of methods depending on the number of datapoints
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
import os
from grid_cell.sphere_manifold import Circle_Manifold, get_circle_mesh
from grid_cell.ori_tuning_manifold import Ori_Tuning_Manifold, Place_Cell_Manifold
from grid_cell.ggpr import GGPR_Fitter
from grid_cell.manifold_fitter import Avg_Fitter, label_mesh
import grid_cell.toy_manifold_metric_estimator as tmsm
from global_setting import *

########################################### hyperparameters
manifold_kwargs = {
    'circle': {
        'sigma': 0.15,
        'center': [100, 100],
        'x_scale': 2
    },
    'ori_tuning': {
        'n_cell': 15,
        'random_gain': True,
        'std_frac': 0.8
    },
    'place_cell': {
        'n_cell': 64,
        'sigma': 0.3,
        'std_frac': 2,
        'random_sigma': True
    }
}

n_points = 500 # number of training points
n_boot = 2
n_dim_list = np.arange(10, 50, 30)

toy_manifold_name_list = ['ori_tuning']
model_name_list = ['gpr', 'kernel']

# Initialize empty data
data = tmsm.init_empty_data(n_dim_list.shape[0], n_boot, toy_manifold_name_list, model_name_list)
data['n_dim_list'] = n_dim_list

########################################### Main
for tm_name in toy_manifold_name_list:
    for ip, n_dim in enumerate(n_dim_list):
        manifold_kwargs[tm_name]['n_cell'] = n_dim
        tm_wrapper = tmsm.Toy_Manifold_Wrapper(tm_name, **manifold_kwargs[tm_name]) # initialize toy manifold
        theta_noiseless, r_noiseless, cov_noiseless, r_metric_noiseless = tm_wrapper.get_ground_truth() # ground truth data

        for model_name in model_name_list:
            if model_name == 'gpr': mpd = tmsm.Metric_Predictor(GGPR_Fitter(circular_period=tm_wrapper.circular_period), model_name=model_name)
            else: mpd = tmsm.Metric_Predictor(Avg_Fitter(circular_period=tm_wrapper.circular_period), model_name=model_name) # initialize model

            for ib in range(n_boot):
                r, theta = tm_wrapper.get_train_data(n_points)
                r_pred, cov_pred, r_metric = mpd.fit_predict(r, theta, theta_noiseless)
                data[f'{tm_name}_{model_name}_sim_score_arr'][ip, ib] = tmsm.matrix_similarity(r_noiseless, r_pred)
                data[f'{tm_name}_{model_name}_cov_sim_score_arr'][ip, ib] = tmsm.matrix_similarity(cov_noiseless, cov_pred)
                data[f'{tm_name}_{model_name}_metric_sim_score_arr'][ip, ib] = tmsm.matrix_similarity(r_metric_noiseless, r_metric)

# pr.disable()
# sortby = 'cumulative'
# ps = pstats.Stats(pr).sort_stats(sortby)
# ps.print_stats(10)

hkl.dump(data, os.path.join(DATAROOT, f"manifold_sim_score_n_dim.hkl"))
