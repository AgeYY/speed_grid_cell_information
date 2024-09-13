import numpy as np
import os
import hickle as hkl
from sklearn.decomposition import PCA
from grid_cell.grid_cell_processor import Speed_Processor
from grid_cell.gkr import GKR_Fitter
from global_setting import DATAROOT

# Combined data loading and processing function
def load_and_process_data(dataset_name, preprocessed_file_name, n_sample_data, pca_components=6, speed_min=0.05, speed_max=None, n_random_projection=None, replace=False, speed_bin_width=0.05):
    data = hkl.load(os.path.join(DATAROOT, preprocessed_file_name))
    processed_data = {}
    
    for dn in dataset_name:
        feamap = data[dn]['feamap']
        label = data[dn]['label']
        
        sp = Speed_Processor()
        sp.load_data(feamap, label)
        feamap, label = sp.sample_data(bin_width=speed_bin_width, n_sample_data=n_sample_data, speed_min=speed_min, speed_max=speed_max, replace=replace, n_random_projection=n_random_projection)
        
        if pca_components is not None:
            pca = PCA(n_components=pca_components)
            feamap = pca.fit_transform(feamap)
            
        label = label[:, [0, 1, 3]]  # remove time
        
        processed_data[dn] = {'feamap': feamap, 'label': label}
    
    return processed_data

# Generalized model fitting function
def fit_gkr_models(processed_data, gpr_params, n_bootstrap=10, speed_bins=None, gkr_n_epochs=20):
    models = {}
    
    for dn, data in processed_data.items():
        feamap = data['feamap']
        label = data['label']
        
        model_list = []
        
        if speed_bins is not None:
            # Compute fisher information for each speed bin
            for i in range(speed_bins.size - 1):
                bin_models = []
                for _ in range(n_bootstrap):
                    label_idx = (label[:, -1] > speed_bins[i]) & (label[:, -1] < speed_bins[i + 1])
                    label_sp = label[label_idx]
                    feamap_sp = feamap[label_idx]

                    model = GKR_Fitter(n_input=label_sp.shape[1], n_output=feamap_sp.shape[1], gpr_params=gpr_params, n_epochs=gkr_n_epochs)
                    model.fit(feamap_sp, label_sp)
                    bin_models.append(model)
                model_list.append(bin_models)
            models[dn] = model_list
        else:
            for _ in range(n_bootstrap):
                model_temp = GKR_Fitter(n_input=label.shape[1], n_output=feamap.shape[1], gpr_params=gpr_params, n_epochs=gkr_n_epochs)
                model_temp.fit(feamap, label)
                model_list.append(model_temp)
            models[dn] = model_list

    return models
