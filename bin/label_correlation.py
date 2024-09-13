from grid_cell.util import compute_correlation_and_p_values
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import hickle as hkl
import os
from global_setting import *

def compute_corr(label, axes, fmt='.2f'):
    correlation_matrix, p_value_matrix = compute_correlation_and_p_values(label)
    
    ax = axes[dataset_name.index(dn) // 3, dataset_name.index(dn) % 3]
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)
    sns.heatmap(correlation_matrix, ax=ax, cmap=cmap, annot=True, center=0, fmt=fmt, square=True, mask=mask, cbar=False, vmin=-0.35, vmax=0.43)
    feature_names = ['x', 'y', 't', 'speed']
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_yticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels(feature_names, rotation=0)
    # Set title at the bottom
    ax.text(0.5, -0.1, dn.upper(), transform=ax.transAxes, ha='center', va='top', fontsize=16)
    return correlation_matrix, p_value_matrix, ax
#################### Hyperparameters ####################
preprocessed_file_name = 'preprocessed_data_OF.hkl'
#################### Main ####################
dataset_name = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
n_sample_data = 2500
data = hkl.load(os.path.join(DATAROOT, preprocessed_file_name))

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
shuffle_fig, shuffle_axes = plt.subplots(3, 3, figsize=(10, 10))
for dn in dataset_name:
    label = data[dn]['label']
    label = label[label[:, -1] > 0.05]
    _, p_value_matrix, _ = compute_corr(label, axes)
    print(f'{dn} p-value matrix: {p_value_matrix}')
    # shuffle each column of label respectively
    shuffle_label = np.copy(label)
    for i in range(label.shape[1]):
        shuffle_label[:, i] = np.random.permutation(label[:, i])
    _, p_value_matrix, _ = compute_corr(shuffle_label, shuffle_axes, fmt='.1e')
    print(f'{dn} shuffle p-value matrix: {p_value_matrix}')

fig.tight_layout()
shuffle_fig.tight_layout()
fig.savefig(os.path.join(FIGROOT, 'label_correlation.svg'))
shuffle_fig.savefig(os.path.join(FIGROOT, 'shuffle_label_correlation.svg'))
plt.show()
