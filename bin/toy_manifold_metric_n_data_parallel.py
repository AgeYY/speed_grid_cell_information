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
from grid_cell.gkr import GKR_Fitter
from grid_cell.gwpr import GWPR_Fitter
from grid_cell.manifold_fitter import Avg_Fitter, label_mesh
import grid_cell.toy_manifold_metric_estimator as tmsm
from collections import defaultdict
from global_setting import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # GPU is not needed for this small dataset # TEMP

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

np.random.seed(RANDOM_STATE + rank)

########################################### hyperparameters
manifold_kwargs = {
    'circle': {
        'sigma': 0.15,
        'center': [100, 100],
        'x_scale': 2
    },
    'ori_tuning': {
        'n_cell': 10,
        'random_gain': True,
        'std_frac': 0.2,
        'sigma': 0.3
    },
    'place_cell': {
        'n_cell': 10,
        'sigma': 0.3,
        'std_frac': 0.5,
        'random_sigma': True
    }
}

blanket_spacing_each_dim = 0.2
jacob_h = blanket_spacing_each_dim # the delta_h for computing jacobian matrix. Setting jacob_h larger than blanket_spacing is only required by 'avg' and 'lw' methods
avg_bin_size = blanket_spacing_each_dim
ground_true_points = 100
n_boot = 10
n_points_list = np.arange(100, 2000, 300)

toy_manifold_name_list = ['place_cell', 'ori_tuning']
model_name_list = ['lw', 'avg', 'gkr'] # available kwarg are: 'lw', 'avg', 'gwpr', 'gkr', 'kernel', 'gpr'

# Initialize empty data
data = defaultdict(lambda: np.full((n_points_list.shape[0], n_boot), np.inf))
data['n_points_list'] = n_points_list
tm_name = toy_manifold_name_list[0]
precision_data = defaultdict(lambda: np.full((n_points_list.shape[0], ground_true_points, manifold_kwargs[tm_name]['n_cell'], manifold_kwargs[tm_name]['n_cell']), np.inf))
cov_data = defaultdict(lambda: np.full((n_points_list.shape[0], ground_true_points, manifold_kwargs[tm_name]['n_cell'], manifold_kwargs[tm_name]['n_cell']), np.inf))

########################################### Main
for tm_name in toy_manifold_name_list:
    tm_wrapper = tmsm.Toy_Manifold_Wrapper(tm_name, **manifold_kwargs[tm_name]) # initialize toy manifold. This is only used to get some hyperparameters

    for model_name in model_name_list:
        if model_name == 'gpr': mpd = tmsm.Metric_Predictor(GGPR_Fitter(circular_period=tm_wrapper.circular_period), model_name=model_name)
        elif model_name=='kernel': mpd = tmsm.Metric_Predictor(Avg_Fitter(circular_period=tm_wrapper.circular_period), model_name=model_name) # initialize model
        elif model_name=='avg': mpd = tmsm.Metric_Predictor(Avg_Fitter(circular_period=tm_wrapper.circular_period, avg_method='bin', bin_search=False, bin_size_mean=avg_bin_size, bin_size_cov=avg_bin_size), model_name=model_name)
        elif model_name=='gwpr':
            mpd = tmsm.Metric_Predictor(GWPR_Fitter(circular_period=tm_wrapper.circular_period), model_name=model_name)
        elif model_name=='gkr':
            mpd = tmsm.Metric_Predictor(GKR_Fitter(circular_period=tm_wrapper.circular_period, n_input=tm_wrapper.n_label, n_output=tm_wrapper.n_cell), model_name=model_name)
        elif model_name=='lw': mpd = tmsm.Metric_Predictor(Avg_Fitter(circular_period=tm_wrapper.circular_period, avg_method='bin', bin_search=False, bin_size_mean=avg_bin_size, bin_size_cov=avg_bin_size, use_bin_lw=True), model_name=model_name)


        for ip, n_points in enumerate(n_points_list):
            for ib in range(rank, n_boot, size):
                print(f'thread {rank} is working on {tm_name}; {model_name}; {n_points};')

                tm_wrapper = tmsm.Toy_Manifold_Wrapper(tm_name, **manifold_kwargs[tm_name]) # initialize toy manifold. This is used for generating data.
                theta_noiseless, r_noiseless, cov_noiseless, r_metric_noiseless, precision_noiseless, fisher_noiseless = tm_wrapper.get_ground_truth(n_points=ground_true_points) # ground truth data
                r, theta = tm_wrapper.get_train_data(n_points, blanket_spacing_each_dim=blanket_spacing_each_dim)
                r_pred, cov_pred, r_metric, precision_pred, fisher = mpd.fit_predict(r, theta, theta_noiseless, jacob_h=jacob_h)

                data[f'{tm_name}_blanket_n_points'] = theta.shape[0] - n_points
                data[f'{tm_name}_{model_name}_manifold_sim_score_arr'][ip, ib] = tmsm.relative_error(r_noiseless, r_pred)
                data[f'{tm_name}_{model_name}_cov_sim_score_arr'][ip, ib] = tmsm.relative_error(cov_noiseless, cov_pred)
                data[f'{tm_name}_{model_name}_metric_sim_score_arr'][ip, ib] = tmsm.relative_error(r_metric_noiseless, r_metric)
                data[f'{tm_name}_{model_name}_precision_sim_score_arr'][ip, ib] = tmsm.relative_error(precision_noiseless, precision_pred)
                data[f'{tm_name}_{model_name}_fisher_sim_score_arr'][ip, ib] = tmsm.relative_error(fisher_noiseless, fisher)

            # only save the last boot
            precision_data[f'{tm_name}_precision_noiseless'][ip] = precision_noiseless
            precision_data[f'{tm_name}_precision_{model_name}_pred'][ip] = precision_pred
            cov_data[f'{tm_name}_cov_noiseless'][ip] = cov_noiseless
            cov_data[f'{tm_name}cov_{model_name}_pred'][ip] = cov_pred

data = dict(data)
precision_data = dict(precision_data)
cov_data = dict(cov_data)
gathered_data = comm.gather(data, root=0)
gathered_precision_data = comm.gather(precision_data, root=0)
gathered_cov_data = comm.gather(cov_data, root=0)
if rank == 0:
    for data_thread in gathered_data: # merge this thread's data into the main data
        for key in data_thread.keys(): # merge one key
            if 'n_points' in key: continue
            computed_data_thread_mask = (data_thread[key] != np.inf)
            data[key][computed_data_thread_mask] = data_thread[key][computed_data_thread_mask]
    for precision_thread in gathered_precision_data:
        for key in precision_thread.keys():
            computed_precision_thread_mask = (precision_thread[key] != np.inf)
            precision_data[key][computed_precision_thread_mask] = precision_thread[key][computed_precision_thread_mask]
    for cov_thread in gathered_cov_data:
        for key in cov_thread.keys():
            computed_cov_thread_mask = (cov_thread[key] != np.inf)
            cov_data[key][computed_cov_thread_mask] = cov_thread[key][computed_cov_thread_mask]

# pr.disable()
# sortby = 'cumulative'
# ps = pstats.Stats(pr).sort_stats(sortby)
# ps.print_stats(10)

if rank == 0:
    hkl.dump(data, os.path.join(DATAROOT, f"manifold_sim_score_n_data.hkl"))
    hkl.dump(precision_data, os.path.join(DATAROOT, f"manifold_precision_n_data.hkl"))
    hkl.dump(cov_data, os.path.join(DATAROOT, f"manifold_cov_n_data.hkl"))
