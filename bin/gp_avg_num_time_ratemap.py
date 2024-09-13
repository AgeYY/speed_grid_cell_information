# please run test_num_time_point_gp first to get the gp data
import numpy as np
import hickle as hkl
from grid_cell.manifold_fitter import Avg_Fitter
from grid_cell.ploter import error_bar_plot
from grid_cell.tuning import ppdata_to_avg_ratemap
import matplotlib.pyplot as plt
import os
from global_setting import *

#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'
pca_str = '6' # none or number as a string

target_nt = 900
n_bins, x_bound, y_bound = 50, [-0.75, 0.75], [-0.75, 0.75] # for avg data

it_gp_timepoint_file_name = 'it_timepoint_gp_{}_{}_{}_{}_pca{}.hkl'.format(mouse_name, module, day, session, pca_str)
it_gp_timepoint_path = os.path.join(DATAROOT, it_gp_timepoint_file_name)
preprocessed_file_name = 'preprocessed_data_{}_{}_{}_{}_pca{}.npz'.format(mouse_name, module, day, session, pca_str)
preprocessed_dir = os.path.join(DATAROOT, 'preprocessed/')
preprocessed_path = os.path.join(preprocessed_dir, preprocessed_file_name)

#################### Obtain data ####################
### GP
data_gp = hkl.load(it_gp_timepoint_path)

def get_ratemap(data, target_nt=1900, key='gp'):
    '''
    key: 'gp' or 'avg'
    '''
    nt, manifold_list = data_gp['n_time_point_list'], data_gp['manifold_list_{}'.format(key)]
    idx = np.where(nt == target_nt)[0][0]
    ratemap = manifold_list[idx][0]
    n_pca = ratemap.shape[1]
    ratemap = ratemap.reshape(50, 50, n_pca)
    return ratemap

ratemap_gp = get_ratemap(data_gp, target_nt=target_nt, key='gp')

### Avg
data = np.load(preprocessed_path, allow_pickle=True)
x, y, dt = data['x'], data['y'], data['dt']
label = np.array([x, y]).T
feamap = data['fire_rate']

ratemap_target = ppdata_to_avg_ratemap(target_nt, feamap, label, n_bins=n_bins, , x_bound=x_bound, y_bound=y_bound)
ratemap_torus = ppdata_to_avg_ratemap(60000, feamap, label, n_bins=n_bins, , x_bound=x_bound, y_bound=y_bound) # when the number of time points is sufficient to get a torus
ratemap_full = ppdata_to_avg_ratemap(feamap.shape[0], feamap, label, n_bins=n_bins, x_bound=x_bound, y_bound=y_bound) # when the number of time points is sufficient to get a torus

#################### Plot ####################
img1, img2, img3, img4 = ratemap_target[:, :, 0], ratemap_gp[:, :, 0], ratemap_torus[:, :, 0], ratemap_full[:, :, 0]

fig, axs = plt.subplots(1, 4, figsize=(7, 2))
images = [np.ma.masked_invalid(img) for img in [img1, img2, img3, img4]]

for ax, img in zip(axs, images):
    pcm = ax.imshow(img)
    ax.axis('off')

cbar_ax = fig.add_axes([0.95, 0.20, 0.02, 0.6])

fig.colorbar(pcm, cax=cbar_ax)
# fig.tight_layout()
plt.show()
