# run fit_gkr_models.py first
import hickle as hkl
from umap import UMAP
from grid_cell.ploter import gp_torus_surface
from grid_cell.util import pca_accumulated_variance_explained_ratio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from grid_cell.gkr import GKR_Fitter
from grid_cell.manifold_fitter import label_mesh
from global_setting import *

#################### Hyperparameters ####################
dn = 'r1m2'
model_path = os.path.join(DATAROOT, f'gkr_models_{dn}_pcaNone.hkl')

n_bins, x_bound, y_bound, speed_bound = 30, [-0.7, 0.7], [-0.7, 0.7], [0.05, 0.45]
speed_slice = 0.2
x_slice = 0.0

# Load the model
gkr = hkl.load(model_path)[0]

# visualize speed slice
query_mesh_speed = label_mesh([x_bound, y_bound], mesh_size=n_bins, grid=True)
query_mesh_speed = np.concatenate([query_mesh_speed, np.ones((query_mesh_speed.shape[0], 1)) * speed_slice], axis=1) # add speed value
feamap_speed, _ = gkr.predict(query_mesh_speed)
label_speed = query_mesh_speed.copy()

# compute the pca
ver = pca_accumulated_variance_explained_ratio(feamap_speed)

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.plot(ver, marker='o', color='tab:green')
ax.set_xlabel('Number of PCs')
ax.set_ylabel('Accumulated Variance Explained Ratio')
fig.tight_layout()
fig.savefig(os.path.join(FIGROOT, 'pca_speed_slice.svg'))

feamap_speed = PCA(n_components=6).fit_transform(feamap_speed)
fitter = UMAP(n_components=3, n_neighbors=100, min_dist=0.8, metric='cosine', init='spectral')
feamap_speed = fitter.fit_transform(feamap_speed)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = gp_torus_surface(feamap_speed, fig=fig, ax=ax, color_arr=feamap_speed[:, 2].reshape(n_bins, n_bins), cbar_label='UMAP3 dim', transparency_factor=0.2)
fig.savefig(os.path.join(FIGROOT, 'speed_slice.svg'))

# visualize x slice
query_mesh_x = label_mesh([y_bound, speed_bound], mesh_size=n_bins, grid=True)
query_mesh_x = np.concatenate([query_mesh_x, np.ones((query_mesh_x.shape[0], 1)) * x_slice], axis=1) # add speed value
query_mesh_x = query_mesh_x[:, [2, 0, 1]]
feamap_x, _ = gkr.predict(query_mesh_x)
label_x = query_mesh_x.copy()

# compute the pca
ver = pca_accumulated_variance_explained_ratio(feamap_x)

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.plot(ver, marker='o', color='tab:orange')
ax.set_xlabel('Number of PCs')
ax.set_ylabel('Accumulated Variance Explained Ratio')
fig.tight_layout()
fig.savefig(os.path.join(FIGROOT, 'pca_x_slice.svg'))

pca = PCA(n_components=3)
feamap_x = pca.fit_transform(feamap_x)
label_x = label_x.reshape(n_bins, n_bins, 3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = gp_torus_surface(feamap_x, fig=fig, ax=ax, color_arr=label_x[:, :, 2] * 100, cbar_label='Speed (cm/s)', cmap='Oranges')
fig.savefig(os.path.join(FIGROOT, 'x_slice.svg'))

plt.show()
