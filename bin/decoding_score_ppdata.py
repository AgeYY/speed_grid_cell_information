import numpy as np
import matplotlib.pyplot as plt
import os
import hickle as hkl
from grid_cell.speed_partition_processor import Speed_Partition_Processor
from sklearn.linear_model import MultiTaskLassoCV, RidgeCV
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import grid_cell.ploter as ploter
from global_setting import *

#################### Hyperparameters ####################
mouse_name = 'R'
day = 'day1'
module = '2'
session = 'open_field_1'

speed_win = [[i, i + 0.025] for i in np.arange(0.025, 0.5, 0.025)]

preprocessed_dir = os.path.join(DATAROOT, 'preprocessed/')
preprocessed_file_name = 'preprocessed_data_{}_{}_{}_{}_pca6.npz'.format(mouse_name, module, day, session)
preprocessed_path = os.path.join(preprocessed_dir, preprocessed_file_name)

decoding_score_path = os.path.join(DATAROOT, 'knn_decoding_score_data_{}_{}_{}_{}.hkl'.format(mouse_name, module, day, session))

# #################### Main ####################
# data = np.load(preprocessed_path, allow_pickle=True)

# fire_rate, x, y, t, speed = data['fire_rate'], data['x'], data['y'], data['t'], data['speed']

# feamap = fire_rate
# label = np.array([x, y]).T

# def compute_knn_score(feamap=None, label=None, fix_set_size=10000, n_boot=10, n_neighbors=3, test_size=0.3):
#     r2 = []
#     index = np.random.choice(feamap.shape[0], fix_set_size, replace=False)
#     feamap, label = feamap[index], label[index]

#     for boot_i in range(n_boot):
#         feamap_train, feamap_test, label_train, label_test = train_test_split(feamap, label, test_size=test_size, random_state=RANDOM_STATE+boot_i)

#         model = KNeighborsRegressor()
#         param = {'n_neighbors': list(range(1, 100, 1))}
#         rand_search = RandomizedSearchCV(model, param, n_jobs=-1)  # 5-fold cross-validation
#         rand_search.fit(feamap_train, label_train)
#         pred = rand_search.predict(feamap_test)
#         r2_boot = r2_score(label_test, pred)

#         r2.append(r2_boot)
#     return r2

# spp = Speed_Partition_Processor(feamap, label, speed)
# spp.load_speed_win(speed_win)
# result = spp.apply_on_speed_win(compute_knn_score)
# result_dict = {'speed_win': speed_win, 'result': result}
# hkl.dump(result_dict, decoding_score_path)
# exit()

# plot the result
data = hkl.load(decoding_score_path)
speed_win, result = np.array(data['speed_win']), np.array(data['result'])

x = [speed_wini[0] for speed_wini in speed_win]
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
fig, ax = ploter.error_bar_plot(x, result, fig=fig, ax=ax)
ax.set_xlabel('Speed (m/s)')
ax.set_ylabel('x, y position decoding \n r2 score')
fig.tight_layout()
plt.show()


