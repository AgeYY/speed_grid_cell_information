import os
import pprint
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import ztest
from grid_cell.linear_regression import fit_a_line, output_params
from grid_cell.slope_ploter import SlopeAnalysis
from global_setting import DATAROOT, FIGROOT

# Example usage:
dataset_names = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
meta_data = {}
for dn in dataset_names:
    lca_data = hkl.load(os.path.join(DATAROOT, f'accuracy_iid_{dn}.hkl'))
    meta_data[dn]['speed_list'] = lca_data[dn]['speed_bins'][:-1] * 100
    meta_data[dn]['accuracy'] = lca_data[dn]['accuracy']
    meta_data[dn]['accuracy_shuffle'] = lca_data[dn]['accuracy_shuffle']

analysis = SlopeAnalysis(lole_data, dataset_names, x_key='speed_list', y_key='accuracy_list', shuffle_y_key='accuracy_shuffle')
fig, ax = analysis.analyze(speed_offset=1, y_text_offset=0.01)
ax.set_ylabel('SCA-Speed Slope (a.u.)', fontsize=16)
ax.hlines(0, -1, len(dataset_names), color='k', linestyle='--')
fig.savefig(os.path.join(FIGROOT, 'SCA_speed_slope.svg'))
plt.show()
