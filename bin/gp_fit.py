import numpy as np
import matplotlib.pyplot as plt
import os
from grid_cell.grid_cell_processor import Grid_Cell_Processor
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from disentangle.model.nw_helper import label_mesh
from grid_cell.ploter import scatter_torus_ploter
from global_setting import *
import gpflow

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

gp_file_name = 'gp_{}_{}_{}_{}_pca{}.npz'.format(mouse_name, module, day, session, pca_str)
gp_dir = os.path.join(DATAROOT, 'gp/')
if not os.path.exists(gp_dir): os.makedirs(gp_dir)
gp_path = os.path.join(gp_dir, gp_file_name) # path to save gp data

fr_smooth_sigma = 10
downsample_rate_gen = 1 # downsample rate for generating data
downsample_rate_plot = 250 # downsample rate for plotting data

query_mesh = label_mesh([[-0.75, 0.75], [-0.75, 0.75]], mesh_size=50, grid=True)
# umap parameters for dimension reduction. This is identical to the paper
visualize_method = 'umap'
umap_components = 3
umap_n_neighbors = 50
umap_min_dist = 0.8
umap_metric = 'cosine'
umap_init = 'spectral'

#################### Main ####################
### Load data
data = np.load(preprocessed_path, allow_pickle=True)
x, y = data['x'], data['y']
label = np.array([x, y]).T
feamap = data['fire_rate']
# downsample
feamap = feamap[::downsample_rate_plot]
label = label[::downsample_rate_plot]

print('Downsampled feamap size {}'.format(feamap.shape))

# fit by gaussian process
model = gpflow.models.GPR(
    (label.astype(np.float64), feamap.astype(np.float64)),
    kernel=gpflow.kernels.SquaredExponential() + gpflow.kernels.White(),
)
opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss, model.trainable_variables)

feamap, _ = model.predict_f(query_mesh)

np.savez(gp_path, feamap=feamap, query_mesh=query_mesh) # save fitted data

print(feamap.shape)
fig, ax = scatter_torus_ploter(feamap, visualize_method=visualize_method, umap_components=umap_components, umap_n_neighbors=umap_n_neighbors, umap_min_dist=umap_min_dist, umap_metric=umap_metric, umap_init=umap_init) # visualization

plt.show()
