import numpy as np
import matplotlib.pyplot as plt
import os
from grid_cell.grid_cell_processor import Grid_Cell_Processor
from grid_cell.ploter import scatter_torus_ploter
from grid_cell.manifold_fitter import GP_Fitter, Avg_Fitter, label_mesh
from grid_cell.gkr import GKR_Fitter
from sklearn.decomposition import PCA
from umap import UMAP
from global_setting import *

#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'
pca_str = 'None' # none or number as a string
speed_value = 0.2

preprocessed_file_name = 'preprocessed_data_{}_{}_{}_{}_pca{}.npz'.format(mouse_name, module, day, session, pca_str)
preprocessed_dir = os.path.join(DATAROOT, 'preprocessed/')
if not os.path.exists(preprocessed_dir): os.makedirs(preprocessed_dir)
preprocessed_path = os.path.join(preprocessed_dir, preprocessed_file_name)

manifold_file_name = 'manifold_{}_{}_{}_{}_pca{}_speed{}.npz'.format(mouse_name, module, day, session, pca_str, str(speed_value))
manifold_dir = os.path.join(DATAROOT, 'manifold/')
if not os.path.exists(manifold_dir): os.makedirs(manifold_dir)
manifold_path = os.path.join(manifold_dir, manifold_file_name) # path to save gp data

downsample_rate_plot = 50 # downsample data

# umap parameters for dimension reduction. This is identical to the paper
visualize_method = 'umap'
umap_components = 3
umap_n_neighbors = 100
umap_min_dist = 0.8
umap_metric = 'cosine'
umap_init = 'spectral'
n_bins, x_bound, y_bound = 50, [-0.75, 0.75], [-0.75, 0.75]
#################### Main ####################
### load and downsample data
data = np.load(preprocessed_path, allow_pickle=True)
x, y, dt, speed = data['x'], data['y'], data['dt'], data['speed']
label = np.array([x, y, speed]).T
feamap = data['fire_rate']
feamap = feamap[::downsample_rate_plot]
label = label[::downsample_rate_plot]

pca = PCA(n_components=6)
feamap = pca.fit_transform(feamap)

# ### Fit by GP. Only use the spatial label
model = GKR_Fitter(n_input=label.shape[1], n_output=feamap.shape[1], n_epochs=0)
model.fit(feamap, label)
query_mesh = label_mesh([x_bound, y_bound], mesh_size=n_bins, grid=True)
query_mesh = np.concatenate([query_mesh, np.ones((query_mesh.shape[0], 1)) * speed_value], axis=1) # add speed value
feamap_gp, _ = model.predict(query_mesh)
label_gp = query_mesh

### Save data
np.savez(manifold_path, feamap_gp=feamap_gp, label_gp=label_gp) # save fitted data

# Plot data
fig, ax = scatter_torus_ploter(feamap_gp, visualize_method=visualize_method, umap_components=umap_components, umap_n_neighbors=umap_n_neighbors, umap_min_dist=umap_min_dist, umap_metric=umap_metric, umap_init=umap_init) # visualization

plt.show()
