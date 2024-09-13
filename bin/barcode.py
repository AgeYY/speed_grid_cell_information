# run fit_gkr_models.py first
import numpy as np
from ripser import Rips, ripser
# import ripserplusplus as rpp_py
from gph import ripser_parallel
from persim import plot_diagrams
import hickle as hkl
from grid_cell.ploter import scatter_torus_ploter, plot_barcodes
from grid_cell.persistent_homology import find_quantile_lifetime, cloud_2_sparse_mat
from grid_cell.util import Shuffled_Matrix
import matplotlib.pyplot as plt
from grid_cell.manifold_fitter import label_mesh
import os
from sklearn import datasets
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.cluster import KMeans
import skfuzzy as fuzz
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

def compute_thresh_bar_shuffle(feamap, n_shuffle=30, max_dim=1, n_threads=-1, eps=0.1):
    sm = Shuffled_Matrix(feamap)
    max_lifetime = np.zeros(max_dim+1)
    for i_shuffle in range(n_shuffle):
        feamap_shuffle = sm[i_shuffle]
        dsparse_shuffle = cloud_2_sparse_mat(feamap_shuffle, eps=eps) # distance matrix of the sparse point cloud
        dgms_shuffle = ripser_parallel(dsparse_shuffle, metric='precomputed', maxdim=maxdim, n_threads=n_threads)
        dgms_shuffle = dgms_shuffle['dgms']
        for dim in range(max_dim+1):
            max_lifetime[dim] = max(max_lifetime[dim], np.max(dgms_shuffle[dim][:, 1] - dgms_shuffle[dim][:, 0]))
    return max_lifetime

def compute_barcode(feamap, maxdim, replace_inf, plot_prcnt, n_threads=-1, thresh_method='shuffle', eps=0.1):
    dsparse = cloud_2_sparse_mat(feamap, eps=eps) # distance matrix of the sparse point cloud
    dgms = ripser_parallel(dsparse, metric='precomputed', maxdim=maxdim, n_threads=n_threads)
    dgms = dgms['dgms']

    if thresh_method == 'quantile':
        ## set up the threshold bar
        tbar1= compute_thresh_bar(dgms[1])
        if maxdim == 1:
            tbar2 = tbar1
        else:
            tbar2= compute_thresh_bar(dgms[2])
        quantile_life_shuffle = [None, tbar1, tbar2]
    elif thresh_method == 'shuffle':
        quantile_life_shuffle = compute_thresh_bar_shuffle(feamap, n_shuffle=20, max_dim=maxdim, n_threads=n_threads, eps=eps)
    return dgms, quantile_life_shuffle

def compute_barcode_pipline(feamap, n_cluster, maxdim, replace_inf, plot_prcnt, eps=0.1, n_pca=6, use_kmeans=True):
    if n_pca is not None:
        feamap = PCA(n_components=n_pca).fit_transform(feamap)
    if use_kmeans:
        kmeans = KMeans(n_clusters=n_cluster).fit(feamap)
        feamap = kmeans.cluster_centers_

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(feamap[:, 0], feamap[:, 1], feamap[:, 2], c='r', marker='o')
    # plt.show()

    dgms, quantile_life_shuffle = compute_barcode(feamap, maxdim, replace_inf, plot_prcnt, eps=eps)
    fig, ax = plot_barcodes(dgms, quantile_life_shuffle=quantile_life_shuffle, replace_inf=replace_inf, plot_prcnt=plot_prcnt)
    return fig, ax

# generate prediction
x_bound, y_bound, speed_bound = [-0.75, 0.75], [-0.75, 0.75], [0.05, 0.45]
speed_slice = 0.2
n_cluster = 1200
n_bins = 50
full_manifold_point = 6400
maxdim = 2
replace_inf = 50
eps = 0.1
plot_prcnt = None
n_pca_list = [6, None]

# Load the model
dataset_names = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
for dn in dataset_names:
    gkr = hkl.load(os.path.join(DATAROOT, f'gkr_models_{dn}_pcaNone.hkl'))[0]

    for n_pca in n_pca_list:
        # speed slice
        print(f'Processing {dn} speed slice...')
        query_mesh_speed = label_mesh([x_bound, y_bound], mesh_size=n_bins, grid=True)
        query_mesh_speed = np.concatenate([query_mesh_speed, np.ones((query_mesh_speed.shape[0], 1)) * speed_slice], axis=1) # add speed value
        feamap_speed, _ = gkr.predict(query_mesh_speed)
        fig, ax = compute_barcode_pipline(feamap_speed, n_cluster, maxdim, replace_inf, plot_prcnt, eps=eps, n_pca=n_pca, use_kmeans=False)
        fig.savefig(os.path.join(FIGROOT, f'barcode_speed_slice_{dn}_pca{n_pca}.svg'))

        # full manifold
        print(f'Processing {dn} full manifold, n_pca = {n_pca}...')
        query_mesh = label_mesh([x_bound, y_bound, speed_bound], mesh_size=full_manifold_point, grid=False, random=True)
        feamap, _ = gkr.predict(query_mesh)
        fig, ax = compute_barcode_pipline(feamap, n_cluster, maxdim, replace_inf, plot_prcnt, n_pca=n_pca, use_kmeans=True)
        fig.savefig(os.path.join(FIGROOT, f'barcode_full_{dn}_pca{n_pca}.svg'))
plt.show()
