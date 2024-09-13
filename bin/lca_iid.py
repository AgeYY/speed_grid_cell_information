import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
from sklearn.decomposition import PCA
import os
from sklearn.linear_model import LogisticRegression
from grid_cell.grid_cell_processor import Speed_Processor
from grid_cell.locf_grid_cell import compute_locf # locf is also called as lca
from grid_cell.toy_locf import calculate_locf_accuracy_upper_bound
from grid_cell.slope_ploter import draw_two_line_data, add_small_noise
from grid_cell.util import permute_columns
import scipy.stats as stats
from global_setting import *

##################################################
#################### Functions ####################
##################################################
def compute_locf_accuracy(feamap, label, n_boot, n_sample_data, x_bound, y_bound, speed_bound, dl, min_data, box_size, speed_win_size, query_mesh_size=500, iid_mode=False):
    sp = Speed_Processor()
    sp.load_data(feamap, label)

    accuracy = []

    for i_boot in range(n_boot):
        print(f'Boot: {i_boot}')

        # Sample data
        feamap, label = sp.sample_data(
            n_sample_data=n_sample_data, 
            speed_min=speed_bound[0], 
            speed_max=speed_bound[1], 
            replace=False, 
            n_random_projection=None
        )
        label = label[:, [0, 1, 3]]  # Drop time label

        # Compute LOCF
        model_locf = LogisticRegression(solver='liblinear')
        speed_bins, accuracy_temp = compute_locf(
            label, 
            feamap, 
            x_bound, 
            y_bound, 
            speed_min=speed_bound[0], 
            speed_max=speed_bound[1], 
            dl=dl, 
            min_data=min_data, 
            box_size=box_size, 
            model=model_locf, 
            speed_win_size=speed_win_size, 
            query_mesh_size=300,
            iid_mode=iid_mode
        )
        accuracy.append(accuracy_temp)

    return speed_bins, np.array(accuracy)

##################################################
#################### Main Process ####################
##################################################

# List of dn values to process
dn_values = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
n_pca = None
iid_mode = False
dl = 0.05
n_sample_data = 10000
x_bound, y_bound, speed_bound = [-0.75, 0.75], [-0.75, 0.75], [0.05, 0.45]
min_data = 50
box_size = [0.05, 0.05, 0.05]
speed_win_size = 0.05
n_boot = 50
preprocessed_file_name = 'preprocessed_data_OF.hkl'
data = hkl.load(os.path.join(DATAROOT, preprocessed_file_name))

for dn in dn_values:
    print(f'Processing DN: {dn}')
    
    # Compute LOCF accuracy
    print('Computing LOCF accuracy')
    feamap = data[dn]['feamap']
    label = data[dn]['label']
    if n_pca is not None: feamap = PCA(n_pca=n_pca).fit_transform(feamap)
    speed_bins, accuracy = compute_locf_accuracy(feamap, label, n_boot, n_sample_data, x_bound, y_bound, speed_bound, dl, min_data, box_size, speed_win_size, iid_mode=False)

    # Compute LOCF accuracy iid
    print('Computing LOCF accuracy iid')
    feamap = data[dn]['feamap']
    label = data[dn]['label']
    if n_pca is not None: feamap = PCA(n_pca=n_pca).fit_transform(feamap)
    speed_bins, accuracy_iid = compute_locf_accuracy(feamap, label, n_boot, n_sample_data, x_bound, y_bound, speed_bound, dl, min_data, box_size, speed_win_size, iid_mode=True)

    # Compute LOCF accuracy, shuffled data
    print('Computing LOCF accuracy shuffle')
    feamap = data[dn]['feamap']
    label = data[dn]['label']
    if n_pca is not None: feamap = PCA(n_pca=n_pca).fit_transform(feamap)
    # label_shuffled = np.random.permutation(label)
    label_shuffled = permute_columns(label)
    speed_bins, accuracy_shuffle = compute_locf_accuracy(feamap, label_shuffled, n_boot, n_sample_data, x_bound, y_bound, speed_bound, dl, min_data, box_size, speed_win_size, iid_mode=True)
    
    # Save results
    result_data = {'speed_bins': speed_bins, 'accuracy': accuracy, 'accuracy_iid': accuracy_iid, 'accuracy_shuffle': accuracy_shuffle}
    output_path = os.path.join(DATAROOT, f'accuracy_iid_{dn}_pca{n_pca}.hkl')
    hkl.dump(result_data, output_path)

    print(f'Finished processing DN: {dn}\n')

##################################################
#################### Code below serves as a quick check ####################
##################################################

#################### Plot results ####################
data = hkl.load(os.path.join(DATAROOT, f'accuracy_iid_r1m1_pca{n_pca}.hkl'))
speed_bins = data['speed_bins']; accuracy = data['accuracy']; accuracy_iid = data['accuracy_iid']

speed_bins = speed_bins[:-1]
speed_flat = np.tile(speed_bins, (n_boot, 1)).flatten()
accuracy_iid_flat = accuracy_iid.flatten()
accuracy_flat = accuracy.flatten()

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax = draw_two_line_data(speed_flat, accuracy_iid_flat, accuracy_flat, ax, data_label='IFGC', shuffle_label='Original', color_data='tab:red', color_shuffle='tab:blue')

ax.set_xlabel('Speed (cm/s)')
ax.set_ylabel('Local Classification Accuracy')
ax.legend()
fig.savefig(os.path.join(FIGROOT, f'lca_iid_pca{n_pca}.svg'))
plt.show()
