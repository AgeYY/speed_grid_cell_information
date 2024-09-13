# fit probablistic representation manifold (PRM) of the grid cell population in different speed bins
import numpy as np
import os
from grid_cell.data_preprocessing import shuffle_fire_rate
from global_setting import *

#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'

# neural data within a combined bin (x, y, speed) will be shuffled
speed_delta = 0.025
speed_win = [[i, i + speed_delta] for i in np.arange(speed_delta, 0.5, speed_delta)]
speed_bin = [sw[0] for sw in speed_win] + [speed_win[-1][-1]]

x_bound, y_bound = [-0.75, 0.75], [-0.75, 0.75]
EPS = 1e-8
spatial_n_bins = 50
x_bins = np.linspace(x_bound[0] - EPS, x_bound[1] + EPS, spatial_n_bins+1)
y_bins = np.linspace(y_bound[0] - EPS, y_bound[1] + EPS, spatial_n_bins+1)

preprocessed_dir = os.path.join(DATAROOT, 'preprocessed/')
preprocessed_file_name = 'preprocessed_data_{}_{}_{}_{}_pca6.npz'.format(mouse_name, module, day, session)
preprocessed_path = os.path.join(preprocessed_dir, preprocessed_file_name)

preprocessed_shuffle_dir = os.path.join(DATAROOT, 'preprocessed_shuffle/')
if not os.path.exists(preprocessed_shuffle_dir): os.makedirs(preprocessed_shuffle_dir)
preprocessed_shuffle_file_name = 'preprocessed_shuffle_data_{}_{}_{}_{}_pca6.npz'.format(mouse_name, module, day, session)
preprocessed_shuffle_path = os.path.join(preprocessed_shuffle_dir, preprocessed_shuffle_file_name)

#################### Main ####################
data = np.load(preprocessed_path, allow_pickle=True)
fire_rate, x, y, t, speed = data['fire_rate'], data['x'], data['y'], data['t'], data['speed']

shuffle_fire_rate = shuffle_fire_rate(fire_rate, x, y, speed, x_bins, y_bins, speed_bin)

np.savez(preprocessed_shuffle_path, fire_rate=shuffle_fire_rate, x=x, y=y, t=t, speed=speed)
