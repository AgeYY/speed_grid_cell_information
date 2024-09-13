# compute the speed modulation of each cell, by fitting a line between firing rate and speed
import numpy as np
import matplotlib.pyplot as plt
from grid_cell.grid_cell_processor import Grid_Cell_Processor, Speed_Processor
import os
from global_setting import *

#################### Hyperparameters ####################
mouse_name = 'S'
day = 'day1'
module = '1'
session = 'open_field_1'

gridness_thre = 0.3
cell_id = [0]
ds = 0.025 # speed bin size
speed_bin_range = [0.025, 0.5]
ci = 0.95
n_shuffle = 1000

speed_temporal_file_name = 'speed_temporal_{}_{}_{}_{}.npz'.format(mouse_name, module, day, session)
speed_temporal_path = os.path.join(DATAROOT, speed_temporal_file_name)

#################### Main ####################
gcp = Grid_Cell_Processor()
gcp.load_data(mouse_name, day, module, session)
fire_rate, x, y, t, speed = gcp.preprocess(gridness_thre=gridness_thre, pca_components=None, use_zscore=False, return_speed=True)
label = np.array([x, y, t, speed]).T
rate_map = gcp.rate_map[:, :, gcp.gridness > gridness_thre]

sp = Speed_Processor()
sp.load_data(fire_rate, label)
speed_bins, mean_firing_rate, low_ci, up_ci = sp.compute_mean_ci_temporal(speed_bin_range, ds, ci_method='se') # for drawing mean firing rate and error band
slopes, intercepts, r_values, p_values, std_errs, intercept_stderrs = sp.compute_linear_regression_temporal() # for drawing regression line
slopes_shuf, intercepts_shuf, r_values_shuf, p_values_shuf = sp.compute_linear_regression_temporal_shuffle(n_shuffle=n_shuffle) # for computing the statistical significance of the regression line

np.savez(speed_temporal_path, speed_bins=speed_bins, mean_firing_rate=mean_firing_rate, low_ci=low_ci, up_ci=up_ci, slopes=slopes, intercepts=intercepts, r_values=r_values, p_values=p_values, std_errs=std_errs, intercept_stderrs=intercept_stderrs, slopes_shuf=slopes_shuf, intercepts_shuf=intercepts_shuf, r_values_shuf=r_values_shuf, p_values_shuf=p_values_shuf, rate_map=rate_map)
