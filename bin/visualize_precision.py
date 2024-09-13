# please run Toy_manifold_metric_n_data_parallel first
import hickle as hkl
import os
import matplotlib.pyplot as plt
from global_setting import *

def draw_precision_cov(precision_data, n_data_i=0, sample_size=3, tm_name='ori_tuning'):
    filtered_data = {k: v for k, v in precision_data.items() if tm_name in k}

    key0 = list(filtered_data.keys())[0]
    n_method = len(filtered_data.keys())
    fig, axes = plt.subplots(n_method, sample_size, figsize=(3, 3))
    for ik, key in enumerate(filtered_data.keys()): # this will iterate over methods
        datak = filtered_data[key] # the shape is (n_num_data_points, n_noiseless_sample, n_dim, n_dim)
        for i_sample in range(sample_size):
            axes[ik, i_sample].imshow(datak[n_data_i, i_sample])
            axes[ik, i_sample].set_title(key)
    return fig, axes

# tm_name = 'place_cell'
tm_name = 'ori_tuning'
data = hkl.load(os.path.join(DATAROOT, "manifold_sim_score_n_data.hkl"))
n_points_list = data['n_points_list']
precision_data = hkl.load(os.path.join(DATAROOT, "manifold_precision_n_data.hkl"))
cov_data = hkl.load(os.path.join(DATAROOT, "manifold_cov_n_data.hkl"))
draw_precision_cov(precision_data, tm_name=tm_name)
draw_precision_cov(cov_data, tm_name=tm_name)
plt.show()
