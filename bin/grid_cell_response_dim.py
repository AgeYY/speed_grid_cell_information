import numpy as np
import matplotlib.pyplot as plt
import os
from grid_cell.grid_cell_processor import Grid_Cell_Processor
from grid_cell.util import pca_accumulated_variance_explained_ratio
from sklearn.decomposition import PCA
from global_setting import *

#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'
# session = 'wagon_wheel_1'
pca_str = 'None' # none or number as a string

preprocessed_file_name = 'preprocessed_data_{}_{}_{}_{}_pca{}.npz'.format(mouse_name, module, day, session, pca_str)
preprocessed_dir = os.path.join(DATAROOT, 'preprocessed/')
if not os.path.exists(preprocessed_dir): os.makedirs(preprocessed_dir)
preprocessed_path = os.path.join(preprocessed_dir, preprocessed_file_name)

downsample_rate_plot = 250 # downsample rate for plotting data
#################### Main ####################
### load and downsample data
data = np.load(preprocessed_path, allow_pickle=True)
x, y, dt = data['x'], data['y'], data['dt']
label = np.array([x, y]).T
feamap = data['fire_rate']
feamap = feamap[::downsample_rate_plot]
label = label[::downsample_rate_plot]

ver = pca_accumulated_variance_explained_ratio(feamap)

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.plot(ver, marker='o')
ax.set_xlabel('Number of PCs')
ax.set_ylabel('Accumulated Variance Explained Ratio')
fig.tight_layout()
plt.show()
