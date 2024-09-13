import numpy as np
import hickle as hkl
import os
from grid_cell.grid_cell_processor import Grid_Cell_Processor
from global_setting import *

#################### Hyperparameters ####################
dataset_name = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
preprocessed_file_name = 'preprocessed_data_OF.hkl'

data = {}
for dn in dataset_name:
    mouse_name, day, module = dn[0], 'day' + dn[1], dn[-1]
    gcp = Grid_Cell_Processor()
    gcp.load_data(mouse_name, day, module, 'open_field_1', fr_smooth_sigma=2, adaptive_fr_sigma=False)

    fire_rate, x, y, t, speed = gcp.preprocess(downsample_rate=None, pca_components=None, return_speed=True, gridness_thre=0.1, use_zscore=False, speed_thre=0.0, speed_max=0.57, spatial_digitize_bin_size=None)
    feamap = fire_rate
    label = np.array([x, y, t, speed]).T

    data[dn] = {'feamap': feamap, 'label': label}

hkl.dump(data, os.path.join(DATAROOT, preprocessed_file_name))

data = hkl.load(os.path.join(DATAROOT, preprocessed_file_name))
# count the number of cells
for dn in dataset_name:
    n_grid = data[dn]['feamap'].shape[1]
    print(f'number of valide grid cells in {dn}: {n_grid}')
