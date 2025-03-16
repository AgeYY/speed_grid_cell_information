import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg
import hickle as hkl
import os

from scipy import stats

from sklearn.linear_model import LogisticRegression
from grid_cell.lole import LOLE, LOCF
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from grid_cell.grid_cell_processor import Speed_Processor
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
# dn_values = ['r1m2']
dn = 'r1m2'
dl = 0.05
n_sample_data = 10000

x_bound, y_bound, speed_bound = [-0.75, 0.75], [-0.75, 0.75], [0.05, 0.45]
min_data = 50
max_data = 5000
box_size = [0.05, 0.05, 0.025] # the box is a cube with edge length 2 * box_size[0] X 2 * box_size[1] X 2 * box_size[2]
speed_win_size = 0.05
n_random_label_anchor = 10
preprocessed_file_name = 'preprocessed_data_OF.hkl'

data = hkl.load(os.path.join(DATAROOT, preprocessed_file_name))

# Get features and labels from data
feamap = data[dn]['feamap']
label = data[dn]['label']

sp = Speed_Processor()
sp.load_data(feamap, label)
# Sample data
feamap, label = sp.sample_data(
    n_sample_data=n_sample_data, 
    speed_min=speed_bound[0], 
    speed_max=speed_bound[1], 
    replace=False, 
    n_random_projection=None
)
label = label[:, [0, 1, 3]]

locf = LOCF(model=None, box_size=box_size) # We only use this class to sample data from boxes, i.e., get_data_within_box
label_anchor = label[np.random.choice(label.shape[0], n_random_label_anchor)]

label_box, feamap_box = locf.get_data_within_box(label, feamap, label_anchor[0])


# feamap_box = feamap_box[feamap_box[:, 10] > 0.01]
# result = pg.multivariate_normality(np.log10(feamap_box), alpha=0.05)
# print(result)


# Plot the distribution of the first feature
# feamap_single = feamap_box[feamap_box[:, 10] < 0.2]
plt.figure(figsize=(8, 6))
plt.hist(feamap_box[:, 0], bins=500, density=False)
plt.xlabel('Feature Value')
plt.ylabel('Count')
plt.title('Distribution of First Feature in Box')
plt.show()


exit()

# show example neuron, by matching to the inferred gaussian curve
p_value_list = []
for la in label_anchor:
    label_box, feamap_box = LOLE.get_data_within_box(label, feamap, la)
    if feamap_box.shape[0] < min_data:
        pass
    for col in range(feamap_box.shape[1]):
        # Perform Shapiro-Wilk test
        try:
            _, p_value = stats.shapiro(np.log10(feamap_box[:, col]))
            p_value_list.append(p_value)
        except UserWarning as e:
            print(e)
            print("feamap_box values:")
            print(feamap_box[:, col])
            print("feamap_box shape:", feamap_box[:, col].shape)
            pass

# After collecting all p-values, plot their distribution
plt.figure(figsize=(8, 6))
# plt.hist(np.log10(p_value_list), bins=50, density=True)
plt.hist(p_value_list, bins=50, density=True)
# plt.axvline(x=np.log10(0.05), color='red', linestyle='--', label='p=0.05')
plt.xlabel('log10(p-value)')
plt.ylabel('Density')
plt.title('Distribution of Shapiro-Wilk Test p-values')
plt.legend()
plt.show()
