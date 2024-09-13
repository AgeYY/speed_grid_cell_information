# please run fit_gp_torus.py first to obtain the gp model
import hickle as hkl
import os
import numpy as np
import grid_cell.ploter as ploter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from global_setting import *

#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'

gp_model_path = os.path.join(DATAROOT, 'gp_torus_speed_{}_{}_{}_{}.hkl'.format(mouse_name, module, day, session))

speed_bin_id = [0, -2]

#################### Main ####################
data = hkl.load(gp_model_path)
speed_win, gp_model = np.array(data['speed_win'], dtype=float), np.array(data['result'])

n_speed = len(speed_bin_id)
fig, ax = plt.subplots(1, n_speed, figsize=(n_speed * 3, 3), subplot_kw={'projection': '3d'})
if n_speed == 1: ax = [ax,]

for i, gpm in enumerate(gp_model[speed_bin_id]):
    pred, pred_label = gpm[0].predict()
    _, _, torus3d = ploter.scatter_torus_ploter(pred, output_data=True)
    fig, ax[i] = ploter.gp_torus_surface(torus3d, fig=fig, ax=ax[i])

    if i == 0:
        ax[i].set_title('Speed: {:.3f} m/s'.format(speed_win[speed_bin_id[i]][0]))
    else:
        ax[i].set_title('{:.3f}'.format(speed_win[speed_bin_id[i]][0]))

    ax[i].set_axis_off()
fig.tight_layout()
plt.show()
