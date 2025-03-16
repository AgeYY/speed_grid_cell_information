import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
import os

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from grid_cell.slope_ploter import SlopeAnalysis, draw_two_line_data_boot
from grid_cell.util import permute_columns
import matplotlib.ticker as ticker

from grid_cell.locf_grid_cell import compute_locf_accuracy
from global_setting import *

##################################################
#################### Main Process ################

##################################################

# List of dn values to process
# dn_values = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
dn_values = ['r1m2']
dl = 0.05
n_sample_data = 10000

x_bound, y_bound, speed_bound = [-0.75, 0.75], [-0.75, 0.75], [0.05, 0.45]
min_data = 50
box_size = [0.05, 0.05, 0.05] # We apologize for this bad code design. The spatial box size has width (or height) as 2 * box_size[0] or 2 * box_size[1]. However, the speed box size is bounded by speed_win_size below. Please make sure that box_size[2] is larger than speed_win_size. This will fix the box's speed size equal to the speed_win_size.
speed_win_size = 0.05
n_boot = 5
preprocessed_file_name = 'preprocessed_data_OF.hkl'
# model_name = 'Logistic Regression'
# model = LogisticRegression(C=1, solver='liblinear')
# model_name = 'Perceptron'
# model = Perceptron()
model_name = 'SVC'
model = SVC()

data = hkl.load(os.path.join(DATAROOT, preprocessed_file_name))

for dn in dn_values:
    print(f'Processing DN: {dn}')
    
    # Get features and labels from data
    feamap = data[dn]['feamap']
    label = data[dn]['label']
    
    # Dictionary to store accuracy results for different classifiers
    result_data = {}
    
    # For consistency, we compute speed_bins with the first classifier run.
    # Loop over classifiers to compute LOCF accuracy
    print(f'Computing LOCF accuracy using {model_name}')

    speed_bins, acc = compute_locf_accuracy(
        feamap, label, n_boot, n_sample_data, 
        x_bound, y_bound, speed_bound, dl, min_data, 
        box_size, speed_win_size, iid_mode=False, model=model
    )


    label_shuffled = permute_columns(label)
    speed_bins, accuracy_shuffle = compute_locf_accuracy(feamap, label_shuffled, n_boot, n_sample_data, x_bound, y_bound, speed_bound, dl, min_data, box_size, speed_win_size, iid_mode=False, model=model)

    result_data['speed_bins'] = speed_bins
    result_data['accuracy'] = acc
    result_data['accuracy_shuffle'] = accuracy_shuffle

    # Save results for this dn value into one file
    output_path = os.path.join(DATAROOT, f'accuracy_{dn}_{model_name}_classifier.hkl')
    hkl.dump(result_data, output_path)
    
    print(f'Finished processing DN: {dn}\n')

##################################################
## Plot results (multiple classifiers)
##################################################
# We plot the results for a selected DN (here, 'r1m2')
dn = 'r1m2'
data = hkl.load(os.path.join(DATAROOT, f'accuracy_{dn}_{model_name}_classifier.hkl'))
speed_bins = data['speed_bins']
# Adjust speed bins if necessary (originally multiplied by 100 for cm/s and dropping last bin edge)
speed_bins = speed_bins[:-1] * 100  
speed = np.tile(speed_bins, (n_boot, 1))
n_boot = data['accuracy'].shape[0]


fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax = draw_two_line_data_boot(speed, data['accuracy'], None, ax, line_label=None, data_label=None, color_data='tab:blue', color_shuffle='tab:grey', mode='BBLR', draw_scatter_data=True)

ax.set_xlabel('Speed (cm/s)')
ax.set_ylabel('Spatial Coding Accuracy')
ax.legend(fontsize=10)
ax.set_title(f'{dn}: {model_name}')
fig.savefig(os.path.join(FIGROOT, f'lca_classifier_{dn}_multi_{model_name}.svg'))

#################### All datasets ####################
dataset_names = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
meta_data = {}
# combine all datasets
for dn in dataset_names:
    data = hkl.load(os.path.join(DATAROOT, f'accuracy_{dn}_{model_name}_classifier.hkl'))
    n_boot = data['accuracy'].shape[0]
    data['speed_bins'] = np.tile(data['speed_bins'][:-1], (n_boot, 1)) * 100
    meta_data[dn] = data

sa = SlopeAnalysis(meta_data, dataset_names, x_key='speed_bins', y_key='accuracy', shuffle_y_key=f'accuracy_shuffle')
fig, ax = sa.analyze(y_text_offset=0, data_color='tab:blue', shuffle_data_color='tab:grey', x_shift=0., add_connecting_line=False, mode='BBLR')
ax.set_ylabel('SCA-Speed Slope (a.u.)', fontsize=16)
ax.hlines(0., -0.2, len(dataset_names), color='k', linestyle='--')
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

fig.savefig(os.path.join(FIGROOT, f'accuracy_shuffle_all_dataset_{model_name}.svg'))

plt.show()
exit()