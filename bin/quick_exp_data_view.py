'''
Simple statistics on the expermental data
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from grid_cell.util import select_arr_by_t
from global_setting import *

########## Hyperparameters ##########
# file_name = 'rat_q_grid_modules_1_2.npz'
file_name = 'rat_r_day1_grid_modules_1_2_3.npz'
qgrid_path = os.path.join(EXP_DATAROOT, file_name)
cell_module = 'spikes_mod2'
session = 'open_field_1'

########## Main ##########
qgrid_data = np.load(qgrid_path, allow_pickle=True)
print('File name: {}'.format(file_name))
print('Files: {}'.format(qgrid_data.files))

print('Statistics of the position and time: ')
space_t_data = {key: qgrid_data[key] for key in ['x', 'y', 'z', 't', 'azimuth']}
space_t_data_pd = pd.DataFrame(space_t_data)
print(space_t_data_pd.describe())

print('\n\n')
print('Looking at cell module {}'.format(cell_module))

spikes_mod1 = qgrid_data[cell_module].item()
print('number of cells: {}'.format(len(spikes_mod1)))
print('number of spikes of cell 0: {}'.format(spikes_mod1[0].shape[0]))
print('A few example spiking time of cell 0: {}'.format(spikes_mod1[0][:10]))

print('\n\n')
print('Selecting session {}...'.format(session))
spikes_mod1 = {key: select_arr_by_t(value, value, session=session, file_name=file_name) for key, value in spikes_mod1.items()}
space_t_data = {key: select_arr_by_t(qgrid_data[key], qgrid_data['t'], session=session, file_name=file_name) for key in ['x', 'y', 'z', 'azimuth', 't']}
space_t_data_pd = pd.DataFrame(space_t_data)
print(space_t_data_pd.describe())
