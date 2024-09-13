# Description: Implementation of the Local Optimal Linear Estimator (LOLE) algorithm.
import numpy as np
from sklearn.linear_model import MultiTaskLasso, LinearRegression, LogisticRegression
from grid_cell.manifold_fitter import label_mesh
from sklearn.metrics import mean_squared_error
from grid_cell.util import permute_columns
from sklearn.model_selection import train_test_split

class LOLE:
    def __init__(self, model=None, box_size=[0.15, 0.15, 0.05], min_data=30):
        self.model = model if model is not None else MultiTaskLasso(alpha=0.1)
        self.box_size = np.array(box_size)
        self.min_data = min_data

    def get_data_within_box(self, label, feamap, label_anchor):
        """
        Select data in a hypercube centered around label_anchor with the given width.
        
        Parameters:
        - label (ndarray): An array of shape (n_samples, n_dimensions) representing the data labels.
        - feamap (ndarray): An array of shape (n_samples, ...) representing the feature map data.
        - label_anchor (ndarray): An array of shape (n_dimensions,) representing the center of the box.
        
        Returns:
        - label_box (ndarray): The labels within the specified box.
        - feamap_box (ndarray): The feature map data corresponding to the labels within the specified box.
        """
        label_valide_idx = np.ones(label.shape[0], dtype=bool)
        for dim in range(label.shape[1]):
            label_valide_idx &= (label[:, dim] > label_anchor[dim] - self.box_size[dim])
            label_valide_idx &= (label[:, dim] < label_anchor[dim] + self.box_size[dim])
        label_box = label[label_valide_idx]
        feamap_box = feamap[label_valide_idx]
        return label_box, feamap_box

    def lole_mse_box(self, feamap_train, label_train, feamap_test, label_test):
        self.model.fit(feamap_train, label_train)
        label_test_pred = self.model.predict(feamap_test)
        mse = mean_squared_error(label_test, label_test_pred)
        return mse

    def calculate_lole_mse(self, label, feamap, query_mesh=None, query_mesh_size=30):
        """
        Calculate the average mean squared error (MSE) for data within boxes centered at query_mesh points.
        
        Parameters:
        - query_mesh (ndarray): An array of shape (n_queries, n_dimensions) representing the mesh points to query. If not given, a mesh with size equals to query_mesh_size will be generated based on the label.
        - label (ndarray): An array of shape (n_samples, n_dimensions) representing the data labels.
        - feamap (ndarray): An array of shape (n_samples, ...) representing the feature map data.
        
        Returns:
        - float: The average MSE across all valid boxes.
        - float: The fraction of valid qm (query mesh points) that had enough data.
        """
        cum_mse = []
        valid_count = 0
        if query_mesh is None:
            qm_id = np.random.choice(label.shape[0], query_mesh_size)
            query_mesh = label[qm_id]

        for qm in query_mesh:
            label_box, feamap_box = self.get_data_within_box(label, feamap, qm)
            label_box = label_box[:, :2]  # Assuming we're only interested in the first two dimensions
            if label_box.shape[0] < self.min_data:
                continue  # Skip if not enough data in the box
            valid_count += 1
            # Split data into training and testing sets
            feamap_box_train, feamap_box_test, label_box_train, label_box_test = train_test_split(
                feamap_box, label_box, test_size=0.33, random_state=42)
            # Calculate MSE
            mse = self.lole_mse_box(feamap_box_train, label_box_train, feamap_box_test, label_box_test)
            cum_mse.append(mse)
        average_mse = np.mean(cum_mse) if cum_mse else float('nan')
        fraction_valid_qm = valid_count / len(query_mesh) if len(query_mesh) > 0 else float('nan')
        return average_mse, fraction_valid_qm


class LOCF:
    def __init__(self, model=None, box_size=[0.15, 0.15, 0.05], min_data=50, dl=1):
        '''
        ds (float): the length between two boxes (to be classfied) is 2 * dl
        '''
        self.model = model if model is not None else LogisticRegression(C=1)
        self.box_size = np.array(box_size)
        self.min_data = min_data
        self.dl = dl

    def get_data_within_box(self, label, feamap, label_anchor):
        """
        Select data in a hypercube centered around label_anchor with the given width.
        
        Parameters:
        - label (ndarray): An array of shape (n_samples, n_dimensions) representing the data labels.
        - feamap (ndarray): An array of shape (n_samples, ...) representing the feature map data.
        - label_anchor (ndarray): An array of shape (n_dimensions,) representing the center of the box.
        
        Returns:
        - label_box (ndarray): The labels within the specified box.
        - feamap_box (ndarray): The feature map data corresponding to the labels within the specified box.
        """
        label_valide_idx = np.ones(label.shape[0], dtype=bool)
        for dim in range(label.shape[1]):
            label_valide_idx &= (label[:, dim] > label_anchor[dim] - self.box_size[dim])
            label_valide_idx &= (label[:, dim] < label_anchor[dim] + self.box_size[dim])
        label_box = label[label_valide_idx]
        feamap_box = feamap[label_valide_idx]
        return label_box, feamap_box

    def locf_accuracy_box(self, feamap_train, label_train, feamap_test, label_test):
        self.model.fit(feamap_train, label_train)
        accuracy = self.model.score(feamap_test, label_test)
        return accuracy

    def calculate_locf_accuracy(self, label, feamap, query_mesh=None, query_mesh_size=30, active_dim=None, iid_mode=False):
        """
        Calculate the average mean squared error (MSE) for data within boxes centered at query_mesh points.
        
        Parameters:
        - query_mesh (ndarray): An array of shape (n_queries, n_dimensions) representing the mesh points to query. If not given, a mesh with size equals to query_mesh_size will be generated based on the label.
        - label (ndarray): An array of shape (n_samples, n_dimensions) representing the data labels.
        - feamap (ndarray): An array of shape (n_samples, ...) representing the feature map data.
        - active_dim (list of int): for each qm (query_mesh[i]), a random vector with shape equals to the length of active_dim will be generated (with length equals to self.dl). This random vector will then add extra dimensions to have the same dimension as qm, the values of extra dimensions are zero. If None, all dimensions are active dimension
        - iid_mode (bool): if True, the columns of feamap_box will be permuted before fitting the model. This approximately shuffles the noise correlation
        
        Returns:
        - float: The average MSE across all valid boxes.
        - float: The fraction of valid qm (query mesh points) that had enough data.
        """
        cum_accu = []
        valid_count = 0
        if query_mesh is None:
            qm_id = np.random.choice(label.shape[0], query_mesh_size)
            query_mesh = label[qm_id]

        if active_dim is None:
            n_active_dim = query_mesh.shape[1]
        else:
            n_active_dim = len(active_dim)

        for qm in query_mesh:
            random_dl = get_random_dl_vector(n_active_dim, dl=self.dl)[0] # only use one vector
            if active_dim is not None:
                random_dl_full = np.zeros(query_mesh.shape[1])
                random_dl_full[active_dim] = random_dl
                random_dl = random_dl_full

            qm_plus, qm_minus = qm + random_dl, qm - random_dl
            _, feamap_plus_box = self.get_data_within_box(label, feamap, qm_plus)
            _, feamap_minus_box = self.get_data_within_box(label, feamap, qm_minus)
            if iid_mode:
                feamap_plus_box = permute_columns(feamap_plus_box)
                feamap_minus_box = permute_columns(feamap_minus_box)

            feamap_box, label_box = combine_two_feature_maps(feamap_plus_box, feamap_minus_box, equal_size=True)
            if label_box.shape[0] < self.min_data:
                continue  # Skip if not enough data in the box
            valid_count += 1
            # Split data into training and testing sets
            feamap_box_train, feamap_box_test, label_box_train, label_box_test = train_test_split(
                feamap_box, label_box, test_size=0.33, random_state=42)
            # Calculate MSE
            accu = self.locf_accuracy_box(feamap_box_train, label_box_train, feamap_box_test, label_box_test)
            cum_accu.append(accu)
        average_accu = np.mean(cum_accu) if cum_accu else float('nan')
        fraction_valid_qm = valid_count / len(query_mesh) if len(query_mesh) > 0 else float('nan')
        return average_accu, fraction_valid_qm

def combine_two_feature_maps(feamap_plus, feamap_minus, equal_size=True):
    """
    Combines two feature maps into a single feature map with corresponding labels and shuffles them.
    
    Args:
    - feamap_plus (np.ndarray): Feature map with label 1.
    - feamap_minus (np.ndarray): Feature map with label 0.
    
    Returns:
    - shuffled_feamap (np.ndarray): Combined and shuffled feature map.
    - shuffled_labels (np.ndarray): Labels corresponding to the shuffled feature map.

    Example usage:
    feamap_plus = np.random.rand(100, 10)  # Replace with actual data
    feamap_minus = np.random.rand(80, 10)  # Replace with actual data
    shuffled_feamap, shuffled_labels = combine_and_shuffle_feature_maps(feamap_plus, feamap_minus)
    """
    if equal_size:
        # Ensure both feature maps have the same number of samples
        min_size = min(feamap_plus.shape[0], feamap_minus.shape[0])
        feamap_plus = feamap_plus[np.random.choice(feamap_plus.shape[0], min_size, replace=False)]
        feamap_minus = feamap_minus[np.random.choice(feamap_minus.shape[0], min_size, replace=False)]

    # Create labels
    labels_plus = np.ones(feamap_plus.shape[0])
    labels_minus = np.zeros(feamap_minus.shape[0])

    # Combine feature maps and labels
    combined_feamap = np.concatenate((feamap_plus, feamap_minus), axis=0)
    combined_labels = np.concatenate((labels_plus, labels_minus), axis=0)

    # Shuffle the combined data
    indices = np.arange(combined_feamap.shape[0])
    np.random.shuffle(indices)

    shuffled_feamap = combined_feamap[indices]
    shuffled_labels = combined_labels[indices]

    return shuffled_feamap, shuffled_labels

def get_random_dl_vector(n_dimensions, dl=1, n_samples=1):
    """
    Generates a random vector of length dl in n_dimensions.
    inputs:
    - n_dimensions (int): The number of dimensions of the vector.
    - dl (float): The length of the vector.
    - n_samples (int): The number of samples to generate.
    outputs:
    - vec (np.ndarray): A random vector of length dl in n_dimensions. The shape is (n_samples, n_dimensions).
    """
    vec = np.random.randn(n_samples, n_dimensions) # Shape: (n_samples, n_dimensions)
    vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True) * dl
    return vec
