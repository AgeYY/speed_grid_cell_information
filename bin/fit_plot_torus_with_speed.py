import numpy as np
import matplotlib.pyplot as plt
from grid_cell.grid_cell_processor import Speed_Processor
from ripser import Rips
from grid_cell.persistent_homology import find_quantile_lifetime, cloud_2_sparse_mat
import hickle as hkl
import os
from grid_cell.grid_cell_processor import Grid_Cell_Processor
from grid_cell.ploter import scatter_torus_ploter, plot_barcodes
from grid_cell.manifold_fitter import GP_Fitter, Avg_Fitter, label_mesh
from grid_cell.gkr import GKR_Fitter
from sklearn.decomposition import PCA
from umap import UMAP
from global_setting import *

### Compute grey bar by computing zscore
def compute_thresh_bar(dgm, quantile=80, sig_factor=5):
    life_time = dgm[:, 1] - dgm[:, 0]
    # remove the infinite life time
    life_time = life_time[np.isfinite(life_time)]
    cutoff = np.percentile(life_time, quantile)
    life_time = life_time[life_time >= cutoff]
    sig = np.std(life_time)
    mu = np.mean(life_time)
    thresh_bar = mu + sig * sig_factor
    return thresh_bar

def compute_barcode(feamap, maxdim, coeff, replace_inf, plot_prcnt):
    dsparse = cloud_2_sparse_mat(feamap, eps=0.5) # distance matrix of the sparse point cloud
    rips = Rips(coeff=coeff, maxdim=maxdim)
    dgms = rips.fit_transform(dsparse, distance_matrix=True)

    ## set up the threshold bar
    tbar1= compute_thresh_bar(dgms[1])
    # tbar2= compute_thresh_bar(dgms[2])
    # quantile_life_shuffle = [None, tbar1, tbar2]
    quantile_life_shuffle = [None, tbar1, None]
    return dgms, quantile_life_shuffle
#################### Hyperparameters ####################
# dataset_name = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
dataset_name = ['q1m1']
dn = 'q1m1'
preprocessed_file_name = 'preprocessed_data_OF.hkl'
n_sample_data = 1000

# umap parameters for dimension reduction. This is identical to the paper
visualize_method = 'umap'
umap_components = 3
umap_n_neighbors = 100
umap_min_dist = 0.8
umap_metric = 'cosine'
umap_init = 'spectral'
n_bins, x_bound, y_bound = 30, [-0.75, 0.75], [-0.75, 0.75]
#################### Main ####################
data = hkl.load(os.path.join(DATAROOT, preprocessed_file_name))

feamap = data[dn]['feamap']
label = data[dn]['label']
feamap = feamap[label[:, -1] > 0.05] # remove low speed data
label = label[label[:, -1] > 0.05] # remove low speed data
label = label[:, [0, 1, 3]] # remove time
shuffle_label = label.copy()
shuffle_label = shuffle_label[np.random.permutation(shuffle_label.shape[0])] # shuffle label

sp = Speed_Processor()
sp.load_data(feamap, label)
feamap, label = sp.sample_data(n_sample_data=n_sample_data, speed_min=0.05)

# ### Fit GP
model = GKR_Fitter(n_input=label.shape[1], n_output=feamap.shape[1], n_epochs=0, circular_period=None, gpr_params={'n_inducing': 200, 'standardize': True, 'seperate_kernel': False})
model.fit(feamap, label)
query_mesh = label_mesh([x_bound, y_bound], mesh_size=n_bins, grid=True, random=False)
query_mesh = np.concatenate([query_mesh, np.ones((query_mesh.shape[0], 1)) * 0.2], axis=1) # add speed value
feamap_gp, _ = model.predict(query_mesh)
label_gp = query_mesh

#  plot ratemap
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
axes = axes.flatten()
cell_id = np.linspace(0, feamap_gp.shape[1], 25).astype(int)
feamap_gp_grid = feamap_gp.reshape(n_bins, n_bins, -1)
for i in range(25):
    axes[i].imshow(feamap_gp_grid[:, :, i], cmap='jet')
    axes[i].set_title('Cell ID: %d' % cell_id[i])

# Plot data see whether the result is torus
pca = PCA(n_components=6)
feamap_gp_pca = pca.fit_transform(feamap_gp)

fig, ax = scatter_torus_ploter(feamap_gp_pca, visualize_method=visualize_method, umap_components=umap_components, umap_n_neighbors=umap_n_neighbors, umap_min_dist=umap_min_dist, umap_metric=umap_metric, umap_init=umap_init) # visualization

maxdim = 1 # maximum homology dimension computed
coeff = 47 # prime coefficient for constructing the coefficient field
replace_inf = 50
plot_prcnt = 90

# compute barcode for the slice
dgms, quantile_life_shuffle = compute_barcode(feamap_gp_pca, maxdim, coeff, replace_inf, plot_prcnt)
fig_speed, _ = plot_barcodes(dgms, quantile_life_shuffle=quantile_life_shuffle, replace_inf=replace_inf, plot_prcnt=plot_prcnt)

plt.show()
