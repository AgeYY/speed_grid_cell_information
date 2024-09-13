'''
we showed a few example gridcell with nice speed modulation in the paper. Their ids are [38, 7, 157, 12, 20, 145], in 'R, day1, 2, open_field_1', gridness > -inf. However, in the paper I should use only cells with gridness > 0.3, it is unclear, after thresholding, what are the new ids.
'''
import numpy as np
from grid_cell.grid_cell_processor import Grid_Cell_Processor, Speed_Processor
import os
from global_setting import *

#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'
cell_id_full = [38, 7, 157, 12, 20, 145] # These are the correct ids for cell shown in the figure, when gridness = -inf. Please go to generate_speed_neuron_info.py to get the correct ids
gridness_thre = 0.3

gcp = Grid_Cell_Processor()
gcp.load_data(mouse_name, day, module, session)
gridness = gcp.compute_gridness()
invalid_cell = gridness <= gridness_thre
cum_invalid_cell = np.cumsum(invalid_cell)
new_cell_id = np.array(cell_id_full) - cum_invalid_cell[cell_id_full]
print(new_cell_id)
