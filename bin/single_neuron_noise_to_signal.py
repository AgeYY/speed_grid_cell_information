# compute the single neuron noise to signal ratio
import numpy as np
import matplotlib.pyplot as plt
import os
from grid_cell.grid_cell_processor import Grid_Cell_Processor
from grid_cell.speed_partition_processor import Speed_Partition_Processor
from scipy.stats import binned_statistic_2d
from grid_cell.ploter import scatter_torus_ploter, error_bar_plot
from grid_cell.util import remove_outliers
from grid_cell.manifold_fitter import GP_Fitter, Avg_Fitter, label_mesh
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from global_setting import *

def compute_single_neuron_noise2signal(feamap, label, EPS=1e-8, n_bins=50, x_bound=[-0.75, 0.75], y_bound=[-0.75, 0.75], bootsize=10000, n_boot=10, count_thre=5):
    n2s_list = []
    for _ in range(n_boot):
        idx = np.random.choice(feamap.shape[0], bootsize, replace=True)
        feamap_boot = feamap[idx]; label_boot = label[idx]
        n2s = compute_single_neuron_noise2signal_one_boot(feamap_boot, label_boot, EPS, n_bins, x_bound, y_bound, count_thre)
        n2s_list.append(np.nanmean(n2s))
    return n2s_list

def compute_single_neuron_noise2signal_one_boot(feamap, label, EPS=1e-8, n_bins=50, x_bound=[-0.75, 0.75], y_bound=[-0.75, 0.75], count_thre=5):
    # convert label to binned spatial position
    EPS = 1e-8
    x_bins = np.linspace(x_bound[0] - EPS, x_bound[1] + EPS, n_bins+1)
    y_bins = np.linspace(y_bound[0] - EPS, y_bound[1] + EPS, n_bins+1)
    # compute the mean and variance for each bin and each neuron. And count in each bin
    fire_rate_binned, _, _, _, = binned_statistic_2d(label[:, 0], label[:, 1], feamap.T, statistic='mean', bins=[x_bins, y_bins])
    fire_rate_binned = fire_rate_binned - np.nanmin(fire_rate_binned, axis=(1, 2), keepdims=True)
    std_binned, _, _, _, = binned_statistic_2d(label[:, 0], label[:, 1], feamap.T, statistic='std', bins=[x_bins, y_bins])
    count, _, _, _, = binned_statistic_2d(label[:, 0], label[:, 1], feamap.T, statistic='count', bins=[x_bins, y_bins])
    count_mask = count > count_thre # only include bins where there are more than 5 samples
    # compute the noise to signal ratio for each bin and each neuron
    n_neuron = fire_rate_binned.shape[0]
    noise_signal_ratio = std_binned / fire_rate_binned
    noise_signal_ratio = np.where(count_mask, noise_signal_ratio, np.nan) # remove bins with less than 5 samples
    noise_signal_ratio = noise_signal_ratio.reshape(n_neuron, -1)
    noise_signal_ratio = np.where(np.isinf(noise_signal_ratio), np.nan, noise_signal_ratio) # remove inf (unzscored firing rate = 0)

    n2s = np.nanmean(noise_signal_ratio, axis=1) # n2s for each neuron averaged over all bins
    return n2s

#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'
# session = 'wagon_wheel_1'
pca_str = '6' # none or number as a string

delta_speed = 0.025
speed_win = [[i, i + delta_speed] for i in np.arange(delta_speed, 0.5, delta_speed)]

preprocessed_file_name = 'preprocessed_data_{}_{}_{}_{}_pca{}.npz'.format(mouse_name, module, day, session, pca_str)
preprocessed_dir = os.path.join(DATAROOT, 'preprocessed/')
if not os.path.exists(preprocessed_dir): os.makedirs(preprocessed_dir)
preprocessed_path = os.path.join(preprocessed_dir, preprocessed_file_name)
#################### Main ####################
data = np.load(preprocessed_path, allow_pickle=True)

x, y, speed, dt = data['x'], data['y'], data['speed'], data['dt']
label = np.array([x, y]).T
feamap = data['fire_rate']

spp = Speed_Partition_Processor(feamap, label, speed)
spp.load_speed_win(speed_win)
n2s = spp.apply_on_speed_win(compute_single_neuron_noise2signal)
n2s = np.array(n2s)
# n2s = [remove_outliers(n2s[i]) for i in range(n2s.shape[0])]

speed_arr = np.array(speed_win)[:, 0]
fig, ax = error_bar_plot(speed_arr, n2s)
ax.set_xlabel('Speed (m/s)')
ax.set_ylabel('Noise to signal ratio')
plt.show()
