import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from global_setting import *

# draw the color bar
cbar_fig, cbar_ax = plt.subplots(figsize=(6, 1))
# cbar_fig.subplots_adjust(bottom=0.5)
cmap = cm.hsv
norm = mcolors.Normalize(vmin=0, vmax=2 * np.pi)
cbar = mcolorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='horizontal')
cbar.set_label(r'$\theta$')
# cbar.ax.tick_params(labelsize=0, length=0)
cbar_fig.savefig(os.path.join(FIGROOT, 'tuning_colorbar.svg'))
plt.show()
