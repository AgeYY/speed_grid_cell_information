# fit probablistic representation manifold (PRM) of the grid cell population in different speed bins
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import hickle as hkl
from grid_cell.speed_partition_processor import Speed_Partition_Processor
from grid_cell.manifold_fitter import GP_Fitter
from grid_cell.gpr_covnet import GPR_CovNet, save_gpr_covnet_all, load_gpr_covnet_all
import grid_cell.ploter as ploter
import grid_cell.torus_geometry as tg
from global_setting import *

#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'

speed_delta = 0.025
speed_win = [[i, i + speed_delta] for i in np.arange(speed_delta, 0.5, speed_delta)]

## original dataset
# preprocessed_dir = os.path.join(DATAROOT, 'preprocessed/')
# preprocessed_file_name = 'preprocessed_data_{}_{}_{}_{}_pca6.npz'.format(mouse_name, module, day, session)
# preprocessed_path = os.path.join(preprocessed_dir, preprocessed_file_name)
# gpr_covnet_dir = os.path.join(DATAROOT, 'gpr_covnet_torus_speed_{}_{}_{}_{}/'.format(mouse_name, module, day, session))

## shuffled dataset
preprocessed_dir = os.path.join(DATAROOT, 'preprocessed_shuffle/')
preprocessed_file_name = 'preprocessed_shuffle_data_{}_{}_{}_{}_pca6.npz'.format(mouse_name, module, day, session)
preprocessed_path = os.path.join(preprocessed_dir, preprocessed_file_name)

# gpr_covnet_dir = os.path.join(DATAROOT, 'gpr_shuffle_covnet_torus_speed_{}_{}_{}_{}/'.format(mouse_name, module, day, session))

### test, two speed win only
speed_win = [[0.025, 0.05], [0.450, 0.475]]
gpr_covnet_dir = os.path.join(DATAROOT, 'short_long_gpr_shuffle_covnet_torus_speed_{}_{}_{}_{}/'.format(mouse_name, module, day, session))

# ### test, two speed win only
# speed_win = [[0.025, 0.05], [0.450, 0.475]]
# gpr_covnet_dir = os.path.join(DATAROOT, 'short_long_gpr_covnet_torus_speed_{}_{}_{}_{}/'.format(mouse_name, module, day, session))

#################### Main ####################
data = np.load(preprocessed_path, allow_pickle=True)

fire_rate, x, y, t, speed = data['fire_rate'], data['x'], data['y'], data['t'], data['speed']

feamap = fire_rate
label = np.array([x, y, t, speed]).T
speed_label = speed

def fit_gpr_covnet_func(feamap, label, sample_size=2500, n_boot=5, epochs=300, with_cov=True):
    model_collect = []
    for _ in range(n_boot):
        idx = np.random.choice(feamap.shape[0], sample_size, replace=False)
        feamap_train, label_train  = feamap[idx], label[idx]

        # generate a gp fitter, get the mesh
        model = GPR_CovNet(epochs=epochs, with_cov=with_cov, batch_size=2048, scale=0.15)
        model.fit(feamap_train, label_train[:, [0, 1]])
        model_collect.append(model.copy())

    return model_collect

spp = Speed_Partition_Processor(feamap, label, speed_label)
spp.load_speed_win(speed_win)
result = spp.apply_on_speed_win(fit_gpr_covnet_func)

save_gpr_covnet_all(speed_win, result, gpr_covnet_dir)
# data = load_gpr_covnet_all(gpr_covnet_dir)

# result = data['result']
# print(result[1][4].predict(label[:5, [0, 1]]))

