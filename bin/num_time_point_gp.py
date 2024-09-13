from mpi4py import MPI
import numpy as np
import hickle as hkl
from grid_cell.persistent_homology import Is_Torus
from grid_cell.manifold_fitter import GP_Fitter, Avg_Fitter
from grid_cell.ploter import error_bar_plot
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn import datasets
from global_setting import *

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

gpus = tf.config.list_physical_devices('GPU')
def worker(i): # set the GPU memory for each worker (thread)
    tf.config.experimental.set_visible_devices(gpus[i % len(gpus)], 'GPU')

#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'
pca_str = '6' # none or number as a string

preprocessed_file_name = 'preprocessed_data_{}_{}_{}_{}_pca{}.npz'.format(mouse_name, module, day, session, pca_str)
preprocessed_dir = os.path.join(DATAROOT, 'preprocessed/')
if not os.path.exists(preprocessed_dir): os.makedirs(preprocessed_dir)
preprocessed_path = os.path.join(preprocessed_dir, preprocessed_file_name)

n_bins, x_bound, y_bound = 50, [-0.75, 0.75], [-0.75, 0.75]

maxdim = 1 # maximum homology dimension computed
coeff = 47 # prime coefficient for constructing the coefficient field
replace_inf = 20
plot_prcnt = 80

n_boot = 20 # number of bootstrap to compute the probability of sucessing find a torus
n_time_point_list_short = np.arange(100, 2000, 100)
n_time_point_list = n_time_point_list_short
n_time_point_list = np.arange(2000, 3000, 100)

it_timepoint_file_name = 'it_timepoint_gp_extra_{}_{}_{}_{}_pca{}.hkl'.format(mouse_name, module, day, session, pca_str)
it_timepoint_path = os.path.join(DATAROOT, it_timepoint_file_name)

#################### Main ####################
# load data
data = np.load(preprocessed_path, allow_pickle=True)
x, y, dt = data['x'], data['y'], data['dt']
label = np.array([x, y]).T
feamap = data['fire_rate']

# set up Is_Torus and fitter
itoruser = Is_Torus(n_bins=n_bins, x_bound=x_bound, y_bound=y_bound, coeff=coeff)

def get_is_torus_timepoint_list(fitter, n_time_point):
    itoruser.set_fitter(fitter)
    it_list_temp = []
    manifold_list_temp, manifold_label_list_temp = [], []
    for _ in range(rank, n_boot, size):
        worker(rank) # allocate GPU memory
        selected_rows = np.random.choice(feamap.shape[0], n_time_point, replace=True)
        feamap_boot, label_boot = feamap[selected_rows, :], label[selected_rows, :] # randomly sample time points
        it, mf, mf_label = itoruser.is_torus(feamap_boot, label_boot, output_manifold=True)
        it_list_temp.append(it)
        manifold_list_temp.append(mf)
        manifold_label_list_temp.append(mf_label)
    comm.Barrier()
    it_list_temp = comm.gather(it_list_temp, root=0)
    manifold_list_temp = comm.gather(manifold_list_temp, root=0)
    manifold_label_list_temp = comm.gather(manifold_label_list_temp, root=0)
    if rank == 0:
        it_list_temp = [val for sublist in it_list_temp for val in sublist]
        manifold_list_temp = [val for sublist in manifold_list_temp for val in sublist]
        manifold_label_list_temp = [val for sublist in manifold_label_list_temp for val in sublist]
    return it_list_temp, manifold_list_temp, manifold_label_list_temp

it_list_gp = []
manifold_list_gp, manifold_label_list_gp = [], []
for n_time_point in n_time_point_list:
    it, mf, mfl = get_is_torus_timepoint_list(GP_Fitter(), n_time_point)
    it_list_gp.append(it)
    manifold_list_gp.append(mf)
    manifold_label_list_gp.append(mfl)

if rank == 0:
    data = {'it_list_gp': it_list_gp, 'manifold_list_gp': manifold_list_gp, 'manifold_label_list_gp': manifold_label_list_gp, 'n_time_point_list': n_time_point_list}
    hkl.dump(data, it_timepoint_path)
