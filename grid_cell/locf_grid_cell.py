from grid_cell.lole import LOLE, LOCF
from grid_cell.manifold_fitter import label_mesh
from sklearn.linear_model import MultiTaskLasso, LinearRegression, LogisticRegression
from global_setting import *

def compute_locf(label, feamap, x_bound, y_bound, speed_min, speed_max, speed_win_size=0.05, dl=0.2, min_data=50, box_size=[0.1, 0.1, 0.05], model=LogisticRegression(C=1, solver='liblinear'), query_mesh_size=100, iid_mode=False):
    '''
    label (n_sample, 3): x, y, speed labels
    '''
    locf = LOCF(model, dl=dl, min_data=min_data, box_size=box_size)
    accuracy_list = []

    speed_bins = np.arange(speed_min, speed_max + speed_win_size / 10.0, speed_win_size)
    for i in range(speed_bins.size - 1):
        label_sp_idx = (label[:, 2] > speed_bins[i]) * (label[:, 2] < speed_bins[i+1])
        label_sp = label[label_sp_idx]
        feamap_sp = feamap[label_sp_idx]
        accuracy, frac_valid = locf.calculate_locf_accuracy(label_sp, feamap_sp, query_mesh_size=query_mesh_size, active_dim=[0, 1], iid_mode=iid_mode)
        accuracy_list.append(accuracy)
        print(f'speed win: {speed_bins[i]:.2f} to {speed_bins[i + 1]:.2f}, accuracy: {accuracy:.2f}, frac_valid: {frac_valid:.2f}')
    return speed_bins, accuracy_list
