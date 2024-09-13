# run generate_processed_exp_data first
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from grid_cell.util import pca_accumulated_variance_explained_ratio
from sklearn.metrics import mean_squared_error, r2_score
import hickle as hkl
import os
from grid_cell.gkr import GKR_Fitter
from sklearn.decomposition import PCA
from global_setting import *

# Generate 2D Fourier features
def fourier_features_2d(x, y, n_terms, max_period=1.5):
    x = x * 2 * np.pi / max_period
    y = y * 2 * np.pi / max_period

    features = [np.ones_like(x)]
    for i in range(1, n_terms + 1):
        for j in range(1, n_terms + 1):
            features.append(np.sin(i * x) * np.sin(j * y))
            features.append(np.cos(i * x) * np.cos(j * y))
            features.append(np.sin(i * x) * np.cos(j * y))
            features.append(np.cos(i * x) * np.sin(j * y))

    # zero j feature
    for i in range(1, n_terms + 1):
        features.append(np.cos(i * x))
        features.append(np.sin(i * x))
    # zero i feature
    for j in range(1, n_terms + 1):
        features.append(np.cos(j * y))
        features.append(np.sin(j * y))
    return np.hstack(features)
#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'
pca_str = 'None' # none or number as a string

preprocessed_file_name = 'preprocessed_data_{}_{}_{}_{}_pca{}.npz'.format(mouse_name, module, day, session, pca_str)
preprocessed_dir = os.path.join(DATAROOT, 'preprocessed/')
if not os.path.exists(preprocessed_dir): os.makedirs(preprocessed_dir)
preprocessed_path = os.path.join(preprocessed_dir, preprocessed_file_name)

#################### Main ####################
### load data
data = np.load(preprocessed_path, allow_pickle=True)
x, y, dt, speed = data['x'], data['y'], data['dt'], data['speed']
feamap = data['fire_rate']

speed_bound = [0.05, 0.62]
speed_idx = (speed > speed_bound[0]) * (speed < speed_bound[1])
x, y, speed = x[speed_idx], y[speed_idx], speed[speed_idx]
feamap = feamap[speed_idx]

# Create Fourier features
spatial_label = fourier_features_2d(x.reshape(-1, 1), y.reshape(-1, 1), 3)
print(spatial_label.shape)

max_pc = 30
pca = PCA(n_components=max_pc)
feamap_pca = pca.fit_transform(feamap)

# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(x, y, c=feamap_pca[:, 5], cmap='viridis', s=50, alpha=0.7)
# plt.colorbar(scatter)
# plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feamap_pca, spatial_label, test_size=0.3, random_state=42)

pc_id_list = np.arange(0, max_pc)
r2_list = []

for pcid in pc_id_list:
    model = LinearRegression()
    model.fit(y_train, X_train[:, pcid])
    x_pred = model.predict(y_test)

    r2 = r2_score(X_test[:, pcid], x_pred)
    r2_list.append(r2)

fig, ax = plt.subplots(figsize=(4, 3))

ax.plot(pc_id_list, r2_list, marker='o')
ax.set_xlabel('PC ID')
ax.set_ylabel('R2 Score')
ax.vlines(5, 0.0, 0.7, colors='r', linestyles='dashed')
# Adding the x-tick label under the red dashed line
fig.tight_layout()
fig.savefig(os.path.join(FIGROOT, 'pca_linear_decoding_r2.svg'))

ver = pca_accumulated_variance_explained_ratio(feamap, cumsum=False)
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.plot(ver[:max_pc], marker='o')
ax.set_xlabel('PC ID')
ax.set_ylabel('Variance Explained Ratio')
ax.vlines(5, 0.0, 0.1, colors='r', linestyles='dashed')
# Adding the x-tick label under the red dashed line
fig.tight_layout()
fig.savefig(os.path.join(FIGROOT, 'pca_processed_data.svg'))
plt.show()
