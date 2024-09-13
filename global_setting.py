import os
import matplotlib as mpl
import numpy as np
import tensorflow as tf
import gpflow

def SET_MPL_FONT_SIZE(font_size):
    mpl.rcParams['axes.labelsize'] = font_size
    mpl.rcParams['ytick.labelsize'] = font_size
    mpl.rcParams['xtick.labelsize'] = font_size
    mpl.rcParams['xtick.labelsize'] = font_size
    mpl.rcParams['ytick.labelsize'] = font_size
    return
SET_MPL_FONT_SIZE(13)
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['legend.frameon'] = False
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

PYTHON_COMPUTING_DEVICE = os.environ['PYTHON_COMPUTING_DEVICE']
if PYTHON_COMPUTING_DEVICE == 'local_computer':
    DATAROOT = './data/'
elif PYTHON_COMPUTING_DEVICE == 'high_performance_computer':
    DATAROOT = '/storage1/fs1/ralfwessel/Active/grid_cell/'
elif PYTHON_COMPUTING_DEVICE == 'high_performance_computer_physics':
    DATAROOT = '/data/zeyuan/grid_cell'

if not os.path.exists(DATAROOT): os.makedirs(DATAROOT)

EXP_DATAROOT = os.path.join(DATAROOT, 'Toroidal_topology_grid_cell_data/')

FIG_DATAROOT = os.path.join(DATAROOT, 'fig_data/')
if not os.path.exists(FIG_DATAROOT): os.makedirs(FIG_DATAROOT)

FIGROOT = os.path.join(DATAROOT, 'fig/')
if not os.path.exists(FIGROOT): os.makedirs(FIGROOT)

EXP_DATA_FILENAMES = [
    'rat_q_grid_modules_1_2.npz',
    'rat_r_day1_grid_modules_1_2_3.npz',
    'rat_r_day2_grid_modules_1_2_3.npz',
    'rat_s_grid_modules_1.npz'
]

FLOAT_TYPE = "float64"
tf.keras.backend.set_floatx(FLOAT_TYPE)
gpflow.config.set_default_float(FLOAT_TYPE)
