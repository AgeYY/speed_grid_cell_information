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

def plot_square_vertices(vertex1, vertex2, vertex3, vertex4, speed, ax, with_colorbar=True, cmap='GnBu'):
    """
    Plot the points corresponding to the four vertices of a square parameterized by t,
    and visualize them in a 2D space with the speed as the color.

    Parameters:
    vertex1, vertex2, vertex3, vertex4: np.ndarray
        Arrays of shape (n, 2) representing the coordinates of the vertices.
    speed: np.ndarray
        Array of shape (n,) representing the speed values.
    """
    # Plot the vertices with scatter points and color them by speed
    sc1 = ax.scatter(vertex1[:, 0], vertex1[:, 1], c=speed, cmap=cmap, label='Vertex 1')
    sc2 = ax.scatter(vertex2[:, 0], vertex2[:, 1], c=speed, cmap=cmap, label='Vertex 2')
    sc3 = ax.scatter(vertex3[:, 0], vertex3[:, 1], c=speed, cmap=cmap, label='Vertex 3')
    sc4 = ax.scatter(vertex4[:, 0], vertex4[:, 1], c=speed, cmap=cmap, label='Vertex 4')

    # Connect the same vertices with black lines
    for i in range(len(vertex1) - 1):
        ax.plot([vertex1[i, 0], vertex1[i + 1, 0]], [vertex1[i, 1], vertex1[i + 1, 1]], 'k-', linewidth=0.5)
        ax.plot([vertex2[i, 0], vertex2[i + 1, 0]], [vertex2[i, 1], vertex2[i + 1, 1]], 'k-', linewidth=0.5)
        ax.plot([vertex3[i, 0], vertex3[i + 1, 0]], [vertex3[i, 1], vertex3[i + 1, 1]], 'k-', linewidth=0.5)
        ax.plot([vertex4[i, 0], vertex4[i + 1, 0]], [vertex4[i, 1], vertex4[i + 1, 1]], 'k-', linewidth=0.5)

    # Connect the vertices of the same speed
    for i in range(len(vertex1)):
        ax.plot([vertex1[i, 0], vertex2[i, 0]], [vertex1[i, 1], vertex2[i, 1]], 'k-', linewidth=0.5)
        ax.plot([vertex2[i, 0], vertex3[i, 0]], [vertex2[i, 1], vertex3[i, 1]], 'k-', linewidth=0.5)
        ax.plot([vertex3[i, 0], vertex4[i, 0]], [vertex3[i, 1], vertex4[i, 1]], 'k-', linewidth=0.5)
        ax.plot([vertex4[i, 0], vertex1[i, 0]], [vertex4[i, 1], vertex1[i, 1]], 'k-', linewidth=0.5)

    if with_colorbar:
        # Add a color bar
        cbar = plt.colorbar(sc1)
        cbar.set_label('Speed (cm/s)')

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return ax

def stack_position_and_speed(position_arr, speed_arr):
    # Check the length of the speed array to determine n
    n = len(speed_arr)
    m = len(position_arr)
    
    # Repeat each array in position_arr n times
    repeated_position_arr = np.repeat(position_arr, n, axis=0)
    
    # Repeat the speed array 4 times
    repeated_speed_arr = np.tile(speed_arr, m)
    
    # Stack the repeated speed array to the last column of the repeated position arrays
    final_output = np.column_stack((repeated_position_arr, repeated_speed_arr))
    
    return final_output

def compute_feamap_one_edge(gkr, position_arr, speed_arr):
    stacked_arr = stack_position_and_speed(position_arr, speed_arr)
    feamap, _ = gkr.predict(stacked_arr)
    return feamap
#################### Hyperparameters ####################
dn = 'r1m2'
model_path = os.path.join(DATAROOT, f'gkr_models_{dn}_pcaNone.hkl')
# cmap = 'GnBu' # inferno as red, GnBu as blue
cmap = 'inferno' # inferno as red, GnBu as blue
pca_color = 'tab:red' if cmap == 'inferno' else 'tab:blue'

n_bins, x_bound, y_bound, speed_bound = 15, [-0.75, 0.75], [-0.75, 0.75], [0.05, 0.45]
## Lattice expansion
lattice_edge = 0.04 # lattice, 0.04 for small lattice, 0.3 for large lattice
# lattice_edge = 0.2 # four points
center_point = 0.0
position_arr_mm = [center_point - lattice_edge/2, center_point - lattice_edge/2]
position_arr_mp = [center_point - lattice_edge/2, center_point + lattice_edge/2]
position_arr_pm = [center_point + lattice_edge/2, center_point - lattice_edge/2]
position_arr_pp = [center_point + lattice_edge/2, center_point + lattice_edge/2]

speed_arr = np.linspace(speed_bound[0], speed_bound[1], n_bins)

# Load the model
gkr = hkl.load(model_path)[1]

feamap_mm = compute_feamap_one_edge(gkr, [position_arr_mm], speed_arr)
feamap_mp = compute_feamap_one_edge(gkr, [position_arr_mp], speed_arr)
feamap_pm = compute_feamap_one_edge(gkr, [position_arr_pm], speed_arr)
feamap_pp = compute_feamap_one_edge(gkr, [position_arr_pp], speed_arr)

def visualize_one_edge(pca, feamap, speed_arr, fig, ax, with_colorbar=True):
    feamap_pca = pca.transform(feamap)
    feamap_pca = feamap_pca.reshape(2, len(speed_arr), 3)

    ax = gp_torus_surface(feamap_pca, fig=fig, ax=ax, color_arr=np.tile(speed_arr * 100, (2, 1)), cbar_label='Speed', reshape_feamap=False, with_colorbar=with_colorbar, cmap=cmap, lw=1, transparency_factor=0.3) # * 100, converting to cm/s

pca = PCA(n_components=3)
stack_feamap = np.vstack((feamap_mm, feamap_mp, feamap_pm, feamap_pp))
pca.fit(stack_feamap)

# each point was counted twice, but this should not affect the result
ver = pca_accumulated_variance_explained_ratio(stack_feamap)

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.plot(ver, marker='o', color=pca_color)
ax.set_xlabel('Number of PCs')
ax.set_ylabel('Accumulated Variance Explained Ratio')
fig.tight_layout()
fig.savefig(os.path.join(FIGROOT, f'pca_lattice_slice_{lattice_edge}.svg'))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

feamap_l = np.vstack([feamap_mm, feamap_mp])
visualize_one_edge(pca, feamap_l, speed_arr, fig, ax)
feamap_t = np.vstack([feamap_mp, feamap_pp])
visualize_one_edge(pca, feamap_t, speed_arr, fig, ax, with_colorbar=False)
feamap_r = np.vstack([feamap_pp, feamap_pm])
visualize_one_edge(pca, feamap_r, speed_arr, fig, ax, with_colorbar=False)
feamap_b = np.vstack([feamap_pm, feamap_mm])
visualize_one_edge(pca, feamap_b, speed_arr, fig, ax, with_colorbar=False)
fig.savefig(os.path.join(FIGROOT, f'3d_lattice_slice_edgel{lattice_edge}.svg'))

feamap_mm_pca = pca.transform(feamap_mm)[:, :2] # First two components
feamap_mp_pca = pca.transform(feamap_mp)[:, :2]
feamap_pm_pca = pca.transform(feamap_pm)[:, :2]
feamap_pp_pca = pca.transform(feamap_pp)[:, :2]
fig, ax = plt.subplots(figsize=(4, 3))
ax = plot_square_vertices(feamap_mm_pca, feamap_mp_pca, feamap_pp_pca, feamap_pm_pca, speed_arr * 100, ax=ax, cmap=cmap)
fig.savefig(os.path.join(FIGROOT, f'lattice_slice_edgel{lattice_edge}.svg'))
plt.show()
