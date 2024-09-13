import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import hickle as hkl
from grid_cell.speed_partition_processor import Speed_Partition_Processor
from grid_cell.manifold_fitter import GP_Fitter
import grid_cell.ploter as ploter
import grid_cell.torus_geometry as tg
from global_setting import *

#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'

speed_win = [[i, i + 0.025] for i in np.arange(0.025, 0.5, 0.025)]

preprocessed_dir = os.path.join(DATAROOT, 'preprocessed/')
preprocessed_file_name = 'preprocessed_data_{}_{}_{}_{}_pca6.npz'.format(mouse_name, module, day, session)
preprocessed_path = os.path.join(preprocessed_dir, preprocessed_file_name)

gp_model_path = os.path.join(DATAROOT, 'gp_torus_speed_{}_{}_{}_{}.hkl'.format(mouse_name, module, day, session))

#################### Main ####################
data = np.load(preprocessed_path, allow_pickle=True)

fire_rate, x, y, t, speed = data['fire_rate'], data['x'], data['y'], data['t'], data['speed']

feamap = fire_rate
label = np.array([x, y, t, speed]).T
speed_label = speed

def fit_gp_func(feamap, label, sample_size=2500, n_boot=10):
    gp_collect = []
    for _ in range(n_boot):
        idx = np.random.choice(feamap.shape[0], sample_size, replace=False)
        feamap_train, label_train  = feamap[idx], label[idx]

        # generate a gp fitter, get the mesh
        gp_fitter = GP_Fitter()
        gp_fitter.fit(feamap_train, label_train[:, [0, 1]])
        gp_collect.append(copy.deepcopy(gp_fitter))

    return gp_collect


spp = Speed_Partition_Processor(feamap, label, speed_label)
spp.load_speed_win(speed_win)
result = spp.apply_on_speed_win(fit_gp_func)
result_dict = {'speed_win': speed_win, 'result': result}

hkl.dump(result_dict, gp_model_path)
