# please run fit_gp_torus_speed.py first to obtain the gp model
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
import os
from grid_cell.speed_partition_processor import Speed_Partition_Processor
from grid_cell.manifold_fitter import GP_Fitter
import grid_cell.ploter as ploter
import grid_cell.torus_geometry as tg
from global_setting import *

SET_MPL_FONT_SIZE(10)

#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'

gp_model_path = os.path.join(DATAROOT, 'gp_torus_speed_{}_{}_{}_{}.hkl'.format(mouse_name, module, day, session))
torus_geometry_path = os.path.join(DATAROOT, 'torus_geometry_{}_{}_{}_{}.hkl'.format(mouse_name, module, day, session))

#################### Main ####################
### generate data
data = hkl.load(gp_model_path)
speed_win, gp_model = np.array(data['speed_win'], dtype=float), np.array(data['result'])

tga = tg.Torus_GP_Analyzer()
tga.load_gp_model(gp_model, speed_win)

def make_pred(gp=None):
    '''
    input:
    '''
    pred, pred_label = gp.predict()
    pred, pred_label = np.array(pred), np.array(pred_label)

    n_lattice = np.sqrt(pred.shape[0]).astype(int)
    pred_grid, pred_label_grid = pred.reshape(n_lattice, n_lattice, -1), pred_label.reshape(n_lattice, n_lattice, -1)
    radius = tg.average_distance_to_center(pred_grid)

    return pred, radius

tga.apply_on_gp_model(make_pred)
radius = tga.select_result(1)
geo_data = {'radius': radius, 'speed_win': speed_win}
hkl.dump(geo_data, torus_geometry_path)
exit()

data = hkl.load(torus_geometry_path)
radius, speed_win = np.array(np.array(data['radius']), np.array(data['speed_win'])

x = [speed_wini[0] for speed_wini in speed_win]
fig, ax = plt.subplots(2, 1, figsize=(2.5, 4))
fig, ax[0] = ploter.error_bar_plot(x, radius, color='tab:blue', fig=fig, ax=ax[0], markersize=5)
ax[0].set_ylabel('Averaged Torus Radius \n (a.u.)')
fig.tight_layout()

# fig, ax[1] = ploter.error_bar_plot(x, area, color='tab:blue', fig=fig, ax=ax[1], markersize=5)
# ax[1].set_xlabel('Speed (m/s)')
# ax[1].set_ylabel('Averaged Lattice Area \n (a.u.)')
# fig.tight_layout()
plt.show()

