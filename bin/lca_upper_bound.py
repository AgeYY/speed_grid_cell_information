# # measure the localling decoding performance on different data. Run generate_processed_exp_data.py first. Requires run torus_size.py first
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
import os
from sklearn.linear_model import LogisticRegression
from grid_cell.grid_cell_processor import Speed_Processor
from grid_cell.locf_grid_cell import compute_locf
from grid_cell.toy_locf import calculate_locf_accuracy_upper_bound
from grid_cell.slope_ploter import draw_two_line_data, add_small_noise
from sklearn.decomposition import PCA
import scipy.stats as stats
from global_setting import *

##################################################
#################### Functions ####################
##################################################
def compute_upper_bound(fisher, dl):
    upper_bound = np.zeros((fisher.shape[0], fisher.shape[1]))
    for i_boot, fisher_each_boot in enumerate(fisher):
        for i_speed, fisher_each_speed in enumerate(fisher_each_boot):
            # the shape of fisher_each_speed is (n_sample_on_torus, n_label, n_label)
            upper_bound_temp = calculate_locf_accuracy_upper_bound(
                n_input=2, 
                dl=dl * 100, 
                fisher=fisher_each_speed
            )  # Convert dl to cm as Fisher is based on cm
            upper_bound[i_boot, i_speed] = upper_bound_temp

    return upper_bound

def compute_locf_accuracy(feamap, label, n_boot, n_sample_data, x_bound, y_bound, speed_bound, dl, min_data, box_size, speed_win_size, query_mesh_size=500):
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
            query_mesh_size=500
        )
        accuracy.append(accuracy_temp)

    return speed_bins, np.array(accuracy)

def save_results(speed_bins, accuracy, upper_bound, dn, n_pca=None):
    data = {'speed_bins': speed_bins, 'accuracy': accuracy, 'upper_bound': upper_bound}
    output_path = os.path.join(DATAROOT, f'locf_fisher_speed_box_{dn}_pca{n_pca}.hkl')
    hkl.dump(data, output_path)

##################################################
#################### Main Process ####################
##################################################

# List of dn values to process
dn_values = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
n_pca = 6
iid_mode = False
dl = 0.05
n_sample_data = 10000
x_bound, y_bound, speed_bound = [-0.75, 0.75], [-0.75, 0.75], [0.05, 0.45]
min_data = 50
box_size = [0.05, 0.05, 0.05]
speed_win_size = 0.05
preprocessed_file_name = 'preprocessed_data_OF.hkl'
data = hkl.load(os.path.join(DATAROOT, preprocessed_file_name))

for dn in dn_values:
    print(f'Processing DN: {dn}')
    
    # Compute upper bound from Fisher
    torus_size_path = os.path.join(DATAROOT, f'torus_shape_speed_{dn}_pca{n_pca}_iid{iid_mode}.hkl')
    size_data = hkl.load(torus_size_path)
    speed_flat = size_data['speed'].flatten()
    fisher = size_data['fisher'] # (n_boot, n_speed, n_sample_on_torus, n_label, n_label)
    fisher = fisher[:, :, :, :2, :2] # keep the spatial labels

    upper_bound = compute_upper_bound(fisher, dl)

    # Compute LOCF accuracy
    n_boot = upper_bound.shape[0]
    feamap = data[dn]['feamap']
    label = data[dn]['label']
    if n_pca is not None: feamap = PCA(n_components=n_pca).fit_transform(feamap)
    speed_bins, accuracy = compute_locf_accuracy(feamap, label, n_boot, n_sample_data, x_bound, y_bound, speed_bound, dl, min_data, box_size, speed_win_size)

    # Save results
    save_results(speed_bins, accuracy, upper_bound, dn, n_pca=n_pca)

    print(f'Finished processing DN: {dn}\n')

##################################################
#################### Code below serves as a quick check ####################
##################################################

#################### Plot results ####################
data = hkl.load(os.path.join(DATAROOT, f'locf_fisher_speed_box_r1m1_pca{n_pca}.hkl'))
speed_bins = data['speed_bins']; accuracy = data['accuracy']; upper_bound = data['upper_bound']
n_boot = upper_bound.shape[0]

speed_bins = speed_bins[:-1]
speed_flat = np.tile(speed_bins, (n_boot, 1)).flatten()
up_flat = upper_bound.flatten()
accuracy_flat = accuracy.flatten()

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax = draw_two_line_data(speed_flat, up_flat, accuracy_flat, ax, data_label='Upper bound \n (computed from Fisher info)', shuffle_label='SCA', color_data='tab:red', color_shuffle='tab:blue')

ax.set_xlabel('Speed (cm/s)')
ax.set_ylabel('Spatial Coding Accuracy')
ax.legend()
fig.savefig(os.path.join(FIGROOT, 'locf_fisher_speed_box.svg'))
plt.show()

# compute correlation, p value of correlation and confidence interval

correlation, p_value = stats.pearsonr(up_flat, accuracy_flat)
print('correlation: ', correlation)
print('p_value: ', p_value)
n = len(up_flat)
z = np.arctanh(correlation)
z_se = 1 / np.sqrt(n - 3)
z_crit = stats.norm.ppf(1 - 0.05 / 2)
ci = np.tanh([z - z_crit * z_se, z + z_crit * z_se])
print('confidence interval: ', ci)
