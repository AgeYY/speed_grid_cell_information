# generate rate map data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from matplotlib import cm
import grid_cell.ploter as plter
import os
from grid_cell.grid_cell_processor import Grid_Cell_Processor
from global_setting import *

#################### Hyperparameters ####################
dn = 'q1m2'
session = 'open_field_1'

mouse_name, day, module = dn[0], 'day' + dn[1], dn[-1]
rate_map_file_name = 'rate_map_data_{}_{}_{}_{}.npz'.format(mouse_name, module, day, session)
rate_map_path = os.path.join(FIG_DATAROOT, rate_map_file_name)

fig_path = os.path.join(FIGROOT, 'rate_map.pdf')
gridness_fig_path = os.path.join(FIGROOT, 'gridness.svg')

gridness_thre = 0.1

#################### Main ####################
### Generate data
gcp = Grid_Cell_Processor()
gcp.load_data(mouse_name, day, module, session)

gridness = gcp.compute_gridness()
rate_map = gcp.rate_map

print(rate_map.shape)
print(gridness.shape)

plt.figure()
plt.imshow(rate_map[:, :, 10])
plt.show()

np.savez(rate_map_path, rate_map=rate_map, gridness=gridness)

### Plot data
def plot_images(sorted_indices, images, scores, score_thre, spine_width=2):
    n_images = len(sorted_indices)
    nrows = int(np.sqrt(n_images)) + 1
    fig, axs = plt.subplots(nrows=nrows, ncols=nrows, figsize=(13, 13),
                            subplot_kw={'xticks': [], 'yticks': []})
    axs = axs.ravel()
    
    for i in range(n_images):
        index = sorted_indices[i]
        img = images[:, :, index]
        score = scores[index]
        edgecolor = 'red' if score < score_thre else 'black'
        axs[i].imshow(img)

        axs[i].spines['bottom'].set_linewidth(spine_width)
        axs[i].spines['top'].set_linewidth(spine_width)
        axs[i].spines['right'].set_linewidth(spine_width)
        axs[i].spines['left'].set_linewidth(spine_width)
        for spine in axs[i].spines.values():
            spine.set_edgecolor(edgecolor)

    # for ax in axs: ax.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig, axs

# draw the color bar
cbar_fig, cbar_ax = plt.subplots(figsize=(6, 1))
cbar_fig.subplots_adjust(bottom=0.5)
cmap = cm.viridis
norm = mcolors.Normalize(vmin=0, vmax=1)
cbar = mcolorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='horizontal')
cbar.set_label('Normalized Firing Rate')
cbar.ax.tick_params(labelsize=0, length=0)
cbar_fig.savefig(os.path.join(FIGROOT, 'rate_map_colorbar.svg'))

# Sort indices based on scores 
data = np.load(rate_map_path)
rate_map, gridness = data['rate_map'], data['gridness']
gridness = np.array(gridness)
sorted_indices = np.argsort(gridness)[::-1]
fig, axs = plot_images(sorted_indices, rate_map, gridness, gridness_thre)
fig.savefig(fig_path, format='pdf', dpi=500)

fig, ax = plt.subplots()
ax.hist(gridness, bins=20)
ax = plter.better_spine(ax)
ax.set_xlabel('Gridness')
ax.set_ylabel('Count')
fig.savefig(gridness_fig_path, format='svg')
fig.tight_layout()

plt.show()
