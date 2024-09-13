# # measure the localling decoding performance on different data. Run generate_processed_exp_data.py first
import numpy as np
import matplotlib.pyplot as plt
import grid_cell.torus_geometry as tg
from scipy.stats import norm
from grid_cell.gkr import GKR_Fitter
from grid_cell.linear_regression import fit_a_line, output_params, draw_line
import hickle as hkl
import os
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from grid_cell.manifold_fitter import label_mesh
from sklearn.model_selection import train_test_split
from sklearn.linear_model import MultiTaskLasso, LinearRegression, LogisticRegression, Perceptron, RidgeClassifier, LogisticRegressionCV
from grid_cell.grid_cell_processor import Speed_Processor
from sklearn.preprocessing import StandardScaler
from grid_cell.linear_regression import BayesianLinearRegression
from sklearn.pipeline import Pipeline
from grid_cell.lole import LOLE, LOCF, get_random_dl_vector
from grid_cell.locf_grid_cell import compute_locf
from grid_cell.toy_locf import calculate_locf_accuracy_upper_bound
from global_setting import *

def calculate_box_indices(label, ranges, box_size):
    """Calculate the box indices for each sample based on label ranges and box sizes."""
    n_samples, n_labels = label.shape
    box_indices = []
    
    for i in range(n_labels):
        n_boxes_i = int(np.ceil((ranges[i][1] - ranges[i][0]) / box_size[i]))
        indices_i = np.floor((label[:, i] - ranges[i][0]) / box_size[i]).astype(int)
        indices_i = np.clip(indices_i, 0, n_boxes_i - 1)
        box_indices.append(indices_i)
    
    return np.array(box_indices).T

def group_samples_by_box(feamap, box_indices):
    """Group samples by their box indices."""
    boxes = {}
    for i in range(feamap.shape[0]):
        idx = tuple(box_indices[i])
        if idx not in boxes:
            boxes[idx] = []
        boxes[idx].append(feamap[i])
    
    return boxes

def shuffle_features_within_boxes(boxes, shuffle=True):
    """Shuffle features within each box."""
    shuffled_boxes = {}
    for key, box_data in boxes.items():
        box_data = np.array(box_data)
        for j in range(box_data.shape[1]):
            if shuffle:
                np.random.shuffle(box_data[:, j])
        shuffled_boxes[key] = box_data
    
    return shuffled_boxes

def calculate_mean_label_for_box(key, ranges, box_size):
    """Calculate the mean label for a given box."""
    return np.array([ranges[j][0] + (k + 0.5) * box_size[j] for j, k in enumerate(key)])

def combine_shuffled_boxes(shuffled_boxes, ranges, box_size):
    """Combine shuffled boxes into a single feamap and label array."""
    shuffled_feamap = []
    shuffled_label = []
    
    for key, box_data in shuffled_boxes.items():
        box_mean_label = calculate_mean_label_for_box(key, ranges, box_size)
        labels_for_box = np.tile(box_mean_label, (box_data.shape[0], 1))
        
        shuffled_feamap.append(box_data)
        shuffled_label.append(labels_for_box)
    
    shuffled_feamap = np.vstack(shuffled_feamap)
    shuffled_label = np.vstack(shuffled_label)
    
    return shuffled_feamap, shuffled_label

def split_data_into_boxes_and_shuffle(feamap, label, ranges, box_size, shuffle=True):
    """Main function to split data into boxes, shuffle features within each box, and combine them back."""
    box_indices = calculate_box_indices(label, ranges, box_size)
    boxes = group_samples_by_box(feamap, box_indices)
    shuffled_boxes = shuffle_features_within_boxes(boxes, shuffle=shuffle)
    return combine_shuffled_boxes(shuffled_boxes, ranges, box_size)

def create_fitter_wrapper(model):
    def fitter_wrapper(query_mesh):
        return model.predict(query_mesh, return_cov=False)[0]
    return fitter_wrapper
##################################################
#################### Get data within a speed box ####################
##################################################

#################### Hyperparameters ####################
x_bound, y_bound, speed_bound = [-0.75, 0.75], [-0.75, 0.75], [0.05, 0.45]
speed_win_size = 0.05
speed_bins = np.arange(speed_bound[0], speed_bound[1] + 0.01, speed_win_size) # include the upper bound
preprocessed_file_name = 'preprocessed_data_OF.hkl'
dn = 'r1m2'
n_sample_data = 20000
n_pca = None
# comment here, not to generate data
data = hkl.load(os.path.join(DATAROOT, preprocessed_file_name))
feamap = data[dn]['feamap']
label = data[dn]['label']
sp = Speed_Processor()
sp.load_data(feamap, label)
feamap, label = sp.sample_data(n_sample_data=n_sample_data, speed_min=speed_bound[0], speed_max=speed_bound[1], replace=False, n_random_projection=None)

speed_box_data = {'feamap': feamap, 'label': label}
hkl.dump(speed_box_data, os.path.join(DATAROOT, 'speed_box_data.hkl'))

####################
########## Load data and preprocessing ##########
####################
# n_random_projection = None
speed_box_data = hkl.load(os.path.join(DATAROOT, 'speed_box_data.hkl'))
feamap, label = speed_box_data['feamap'], speed_box_data['label']
if n_pca is not None:
    pca = PCA(n_components=n_pca)
    feamap = pca.fit_transform(feamap)
label = label[:, [0, 1, 3]] # drop time label

####################
########## get gkr model ##########
####################
model_data = hkl.load(os.path.join(DATAROOT, f'gkr_models_{dn}_pca{n_pca}.hkl'))
model = model_data[0]

# compute fisher information for each speed bin
dl = 0.05
up_list = []
diag_up_list = []
for i in range(speed_bins.size - 1):

    label_idx = (label[:, -1] > speed_bins[i]) * (label[:, -1] < speed_bins[i + 1])
    label_sp = label[label_idx]
    feamap_sp = feamap[label_idx]

    # compute fisher information
    qm_idx = np.random.choice(np.arange(label_sp.shape[0]), 500)
    query_mesh = label_sp[qm_idx]

    _, cov = model.predict(query_mesh)

    prec = np.linalg.inv(cov)
    model_wrap = create_fitter_wrapper(model)
    J = tg.compute_jacobian_central(model_wrap, query_mesh, h=0.01)
    fisher = tg.compute_fisher_info(J, prec)

    # mask the diagonal
    n_sample, n_feature, _ = cov.shape
    print('model, n_feature: ', n_feature)
    mask = np.eye(n_feature)
    diag_cov = cov * mask[None, :, :]

    diag_prec = np.linalg.inv(diag_cov)
    model_wrap = create_fitter_wrapper(model)
    diag_fisher = tg.compute_fisher_info(J, diag_prec)

    ########## Compute upper bound ##########
    space_fisher = fisher[:, :2, :2]
    upper_bound = calculate_locf_accuracy_upper_bound(n_input=2, dl=dl, fisher=space_fisher)
    up_list.append(upper_bound)

    space_diag_fisher = diag_fisher[:, :2, :2]
    diag_upper_bound = calculate_locf_accuracy_upper_bound(n_input=2, dl=dl, fisher=space_diag_fisher)
    diag_up_list.append(diag_upper_bound)
print('upper_bound: ', up_list)

####################
########## Compute LOCF ##########
####################
min_data = 50
box_size = [0.05, 0.05, 0.05]

model_locf = LogisticRegression(solver='liblinear')
speed_bins, accuracy = compute_locf(label, feamap, x_bound, y_bound, speed_min=speed_bound[0], speed_max=speed_bound[1], dl=dl, min_data=min_data, box_size=box_size, model=model_locf, speed_win_size=speed_win_size, query_mesh_size=500)
print('accuracy: ', accuracy)

#################### Save results ####################
data = {'speed_bins': speed_bins, 'accuracy': accuracy, 'up_list': up_list}
hkl.dump(data, os.path.join(DATAROOT, 'locf_fisher_speed_box.hkl'))

#################### Plot results ####################
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(speed_bins[:-1], accuracy, label='LOCF', marker='o')
ax.plot(speed_bins[:-1], up_list, label='Upper bound', marker='o')
ax.plot(speed_bins[:-1], diag_up_list, label='Diagonal Upper bound', marker='o')
ax.set_xlabel('Speed')
ax.set_ylabel('Accuracy')
ax.legend()
fig.savefig(os.path.join(FIGROOT, 'locf_fisher_speed_box.svg'))
plt.show()
