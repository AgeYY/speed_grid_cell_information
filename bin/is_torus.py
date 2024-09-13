# run fit_gkr_models.py first
import hickle as hkl
from umap import UMAP
from grid_cell.ploter import gp_torus_surface
from grid_cell.persistent_homology import betti1_equal_2, betti_torus_check
from grid_cell.util import pca_accumulated_variance_explained_ratio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from grid_cell.gkr import GKR_Fitter
from grid_cell.manifold_fitter import label_mesh
from global_setting import *

def make_fix_speed_query(x_bound, y_bound, n_bins, speed_val):
    query_mesh = label_mesh([x_bound, y_bound], mesh_size=n_bins, grid=True)
    query_mesh = np.concatenate([query_mesh, np.ones((query_mesh.shape[0], 1)) * speed_val], axis=1) # add speed value
    return query_mesh
#################### Hyperparameters ####################
# dataset_name = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
dataset_name = ['r1m1', 'r1m2']
pca_component =6
speed_bin_size = 0.05
n_bins, x_bound, y_bound, speed_bound = 60, [-0.75, 0.75], [-0.75, 0.75], [0.05, 0.45]

speed_list = np.arange(speed_bound[0], speed_bound[1], speed_bin_size)
is_torus_dict = {}
dgms_dict = {}

umap_fitter = UMAP(n_components=3, n_neighbors=100, min_dist=0.8, metric='cosine', init='spectral')
#################### Compute is torus ####################
for dn in dataset_name:
    is_torus_dict[dn] = []
    dgms_dict[dn] = []
    model_path = os.path.join(DATAROOT, f'gkr_models_{dn}_pcaNone.hkl')
    models = hkl.load(model_path)
    for gkr in models:
        is_torus_dict[dn].append([])
        dgms_dict[dn].append([])
        for speed_val in speed_list:
            query_mesh = make_fix_speed_query(x_bound, y_bound, n_bins, speed_val)
            feamap, _ = gkr.predict(query_mesh)
            feamap = PCA(n_components=pca_component).fit_transform(feamap)
            feamap = umap_fitter.fit_transform(feamap)

            # is_torus, dgms = betti1_equal_2(feamap, eps=0.2)
            is_torus, dgms  = betti_torus_check(feamap, eps=0.2)
            print('dn {}, Speed: {}, is_torus: {}'.format(dn, speed_val, is_torus))
            is_torus_dict[dn][-1].append(is_torus)
            dgms_dict[dn][-1].append(dgms)
    is_torus_dict[dn] = np.array(is_torus_dict[dn])

data = {'speed_list': speed_list, 'is_torus_dict': is_torus_dict, 'dgms': dgms_dict}
hkl.dump(data, os.path.join(DATAROOT, 'is_torus_speed.hkl'))
##################################################

data = hkl.load(os.path.join(DATAROOT, 'is_torus_speed.hkl'))
is_torus_list = data['r1m1'][0]
speed_list = data['speed_list']

fig, ax = plt.subplots(figsize=(3, 3))
ax.plot(speed_list, is_torus_list, c='tab:green')
ax.scatter(speed_list, is_torus_list, c='tab:green')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
for axis in ['bottom', 'left']:
  ax.spines[axis].set_linewidth(2)
ax.set_xlabel('Speed (m/s)')
ax.set_ylabel('Probability of finding a torus')
plt.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGROOT, 'is_torus_speed.svg'))
plt.show()
