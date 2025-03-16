from grid_cell.lole import LOLE, LOCF
from grid_cell.manifold_fitter import label_mesh
from sklearn.linear_model import MultiTaskLasso, LinearRegression, LogisticRegression
from grid_cell.grid_cell_processor import Speed_Processor
from global_setting import *


def compute_locf(label, feamap, x_bound, y_bound, speed_min, speed_max, speed_win_size=0.05, dl=0.2, min_data=50, box_size=[0.05, 0.05, 0.05], model=LogisticRegression(C=1, solver='liblinear'), query_mesh_size=100, iid_mode=False):
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

def compute_locf_accuracy(feamap, label, n_boot, n_sample_data, x_bound, y_bound, speed_bound, dl, min_data, box_size, speed_win_size, query_mesh_size=500, iid_mode=False, model=LogisticRegression(C=1, solver='liblinear')):
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
            model=model, 
            speed_win_size=speed_win_size, 
            query_mesh_size=300,
            iid_mode=iid_mode

        )
        accuracy.append(accuracy_temp)

    return speed_bins, np.array(accuracy)