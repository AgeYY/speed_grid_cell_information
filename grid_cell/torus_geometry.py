import numpy as np
import hickle as hkl
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsRegressor
from grid_cell.manifold_fitter import label_mesh
from sklearn.metrics import r2_score
import copy
from global_setting import *

def average_lattice_area(feamap):
    '''
    input:
    feamap: (para0, para1, n_feature). Para0, para1 can be x and y
    '''
    diffs_i = np.linalg.norm(feamap[1:, :-1] - feamap[:-1, :-1], axis=2)
    diffs_j = np.linalg.norm(feamap[:-1, 1:] - feamap[:-1, :-1], axis=2)
    lattice_areas = diffs_i * diffs_j
    avg_area = np.mean(lattice_areas)
    return avg_area

def average_distance_to_center(feamap):
    '''
    input:
    feamap: (n_lattice, n_lattice, n_feature)
    '''
    center = np.mean(feamap, axis=(0, 1))
    distances = np.linalg.norm(feamap - center, axis=2)
    avg_distance = np.mean(distances)
    return avg_distance

def torus_avg_radius(feamap):
    '''
    same as average_distance_to_center, but the input has shape (n_sample, n_feature). The n_samples are expected to be uniformly distributed on the torus
    '''
    center = np.mean(feamap, axis=0)
    distances = np.linalg.norm(feamap - center, axis=1)
    avg_distance = np.mean(distances)
    return avg_distance

def knn_decoding_score(gp_feamap, gp_label, feamap_test, label_test, test_subset_size=2500, n_neighbors=500, n_boot=10, return_boot=False):
    '''
    input:
        gp_feamap: (n_lattice * n_lattice, n_feature)
        gp_label: (n_lattice * n_lattice, n_label) strictly speaking, n_label should be 2 corresponding to x and y
        feamap_test: (n_sample, n_feature)
        label_test: (n_sample, n_label)
        test_subset_size: the size of the subset of the test set
        n_neighbors: the number of neighbors used in knn regressor
        n_boot: the number of bootstrap
        return_boot: whether to return the bootstrap result
    output:
        r2: the r2 score of the knn regressor
    '''
    knn_regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_regressor.fit(gp_feamap, gp_label)

    r2 = []
    for _ in range(n_boot):
        index = np.random.choice(np.arange(feamap_test.shape[0]), test_subset_size, replace=False)
        feamap_test_boot, label_test_boot = feamap_test[index], label_test[index]
        pred = knn_regressor.predict(feamap_test_boot)
        r2_boot = r2_score(label_test_boot, pred)
        r2.append(r2_boot)
    r2_mean = np.mean(r2)

    if return_boot:
        return r2_mean, r2
    else:
        return r2_mean

class Torus_GP_Analyzer():
    def load_gp_model(self, gp_model, speed_win):
        '''
        load gp models
        input:
            gp_model: a neatsed list of gp models. The first dimension is the speed, the second dimension is the gp model of different bootstrap but same speed
            speed_win: a list of speed win.
        '''
        self.gp_model = copy.deepcopy(gp_model)
        self.speed_win = speed_win.copy()

    def apply_on_gp_model(self, func, *arg, **kwarg):
        '''
        apply a function on the gp model
        input:
            func: a function that takes gp as input.
            arg: the arguments of the function
            kwarg: the keyword arguments of the function
        output:
            result: a nested list of the same shape as gp_model. The shape is [speed, n_gp_model_boot, *func_output_shape]
        '''
        self.result = []
        for gp_speed in self.gp_model:
            self.result.append([])
            for gp in gp_speed:
                result_gp = func(*arg, **kwarg, gp=gp)
                self.result[-1].append(result_gp)
        return copy.deepcopy(self.result)

    def select_result(self, slice_idx):
        '''
        select the result of a specific idx along the third dimension. Note the shape of result is [speed, n_gp_model_boot, *func_output_shape]
        input:
            slice_idx: the dimension to slice
        output:
            sliced_result: the shape is [speed, n_gp_model_boot, *func_output[idx].shape]
        '''
        sliced_result = slice_third_axis_list(self.result, slice_idx)
        return sliced_result

def slice_third_axis_list(nested_list, idx):
    '''
    input:
        nested_list: a nested list, with shape (n1, n2, n3)
        idx: the index of the third dimension
    output:
        sliced_list: a list with shape (n1, n2)
    '''
    # Initialize an empty nested list to store the sliced elements
    sliced_list = []

    # Loop over the expected range of the outermost list (assuming it's regular and non-empty)
    for matrix in nested_list:
        # Initialize an empty list for the current slice
        current_slice = []

        # Loop through each row
        for row in matrix:
            current_slice.append(row[idx])

        # Append the current slice to the sliced_list
        sliced_list.append(current_slice)
    return sliced_list

def compute_jacobian_central(f, X, h=1e-5, *args, **kwargs):
    """
    Estimate the Jacobian matrix J of a vector-valued function f at points X using central differences.

    :param f: The function should accept an array of shape (N, D) and return an array of shape (N, P).
    :param X: 2D NumPy array of shape (N, D), where N is the number of points and D is the dimension of the input space.
    :param h: Small step size for estimating the derivative. Defaults to 1e-5.
    :param args: Additional positional arguments for the function f.
    :param kwargs: Additional keyword arguments for the function f.
    
    :return: A 3D NumPy array J of shape (N, P, D), where J[n, :, i] represents the estimated partial derivatives of the P outputs with respect to the i-th input variable, evaluated at the n-th point.
    """
    N, D = X.shape  # Number of points (N) and dimensions of input space (D)
    
    # Precompute the output shape by evaluating the function at the first data point
    y = f(X[0].reshape(1, -1), *args, **kwargs)
    P = y.shape[1]  # Number of dimensions in the output space
    
    # Initialize the Jacobian matrix with zeros
    J = np.zeros((N, P, D))
    
    for i in range(D):  # Iterate over each dimension
        # Shift the nth data point in the positive and negative direction along the ith dimension
        X_plus_h = np.copy(X)
        X_minus_h = np.copy(X)
        X_plus_h[:, i] += h / 2.0
        X_minus_h[:, i] -= h / 2.0
            
        # Evaluate the function at the shifted data points
        y_plus = f(X_plus_h, *args, **kwargs)
        y_minus = f(X_minus_h, *args, **kwargs)

        # Estimate the partial derivative using central difference
        J[:, :, i] = (y_plus - y_minus) / h
    
    return J

def compute_riemannian_metric_from_jacobian(jacobian):
    '''
    jacobian shape is (n_point, n_output, n_input)
    '''
    return np.einsum('nij,nik->njk', jacobian, jacobian)

def compute_riemannian_metric(f, X, h=1e-5, *args, **kwargs):
    '''
    compute the riemannian metric using the jacobian
    '''
    jacobian = compute_jacobian_central(f, X, h, *args, **kwargs)
    return compute_riemannian_metric_from_jacobian(jacobian)

def project_noise_to_plane(covmat, tan_vec0, tan_vec1, EPS=1e-10):
    """
    Project the covariance matrices into the planes formed by normalized tangent vectors.

    :param covmat: A 3D NumPy array of covariance matrices of shape (N, P, P).
    :param tan_vec0: A 2D NumPy array of first tangent vectors of shape (N, P).
    :param tan_vec1: A 2D NumPy array of second tangent vectors of shape (N, P).
    :return: A 3D NumPy array of projected noise covariance matrices of shape (N, 2, 2).
    """
    N, P = tan_vec0.shape
    projected_covmat = np.zeros((N, 2, 2))

    for n in range(N):
        # Normalize the first tangent vector
        norm_vec0 = (tan_vec0[n] + EPS) / np.linalg.norm(tan_vec0[n] + EPS)
        # Normalize the second tangent vector
        norm_vec1 = (tan_vec1[n] + EPS) / np.linalg.norm(tan_vec1[n] + EPS)

        # Stack the normalized vectors to form the projection matrix
        projection = np.vstack((norm_vec0, norm_vec1)).T
        # Project the covariance matrix onto the plane defined by the normalized tangent vectors
        projected_covmat[n] = projection.T @ covmat[n] @ projection
    
    return projected_covmat

def compute_fisher_info(jacobian, precision_matrix):
    '''
    compute the fisher information matrix
    inputs:
        jacobian: the jacobian matrix with shape (n_sample, n_feature, n_label)
        precision_matrix: the precision matrix with shape (n_sample, n_feature, n_feature)
    '''
    return np.einsum('nij,nik,nkl->njl', jacobian, precision_matrix, jacobian)

def decorator_for_nested_results(func):
    def wrapper(nested_results, *args, **kwargs):
        return [
            [func(model, *args, **kwargs) for model in sublist]
            for sublist in nested_results
        ]
    return wrapper

def compute_lattice_area(jacobian):
    assert jacobian.shape[2] == 2, "n_label must be 2"
    # Compute the area formed by the two tangent vectors
    # which are the two columns of the Jacobian matrix for each feature point
    v1 = jacobian[:, :, 0]; v2 = jacobian[:, :, 1]
    areas = np.sqrt(np.sum(v1 * v1, axis=1) * np.sum(v2 * v2, axis=1) - np.sum(v1 * v2, axis=1) ** 2)
    return areas

def project_variance(Sigma, d):
    """
    Project the variance for each sample along each direction vector.

    :param Sigma: 3-D numpy array with shape (n_sample, P, P), containing covariance matrices for each sample
    :param d: 2-D numpy array with shape (n_sample, P), containing direction vectors for each sample
    :return: 1-D numpy array containing the variance of the noise projected onto each direction vector
    """
    # Normalize each direction vector
    d_norm = d / np.linalg.norm(d, axis=1)[:, np.newaxis]
    
    # Compute the projected variance for each sample
    variance_projection = np.einsum('ij,ijk,ik->i', d_norm, Sigma, d_norm)
    
    return variance_projection

def compute_fisher_one_vec(vec, J, cov):
    """
    compute fisher information matrix for one direction vector
    """
    s = project_variance(cov, vec)
    inv_s = 1.0 / s

    derivative = np.einsum('ij,ijk->ik', vec, J)
    fisher_matrices = np.einsum('ij,ik,i->ijk', derivative, derivative, inv_s)
    return fisher_matrices

# to add a new metric:
# 1. define a new function that takes feamap, cov, J as input
# 2. add the function to the metric_func dictionary in the __init__ function
# 3. add the function name to the requires_jacobian or requires_cov set in the __init__ function
# 4. add metric_transformations in save_results.

def feamap_cov_J_lattice_area(feamap=None, cov=None, J=None):
    return compute_lattice_area(J[:, :, :-1]).mean()

def feamap_cov_J_radius(feamap=None, cov=None, J=None):
    return torus_avg_radius(feamap)

def feamap_cov_J_center_dist(feamap=None, cov=None, J=None):
    return np.linalg.norm(np.mean(feamap, axis=0))

def feamap_cov_J_total_noise(feamap=None, cov=None, J=None):
    return np.trace(cov, axis1=1, axis2=2).mean()

def feamap_cov_J_noise_proj(feamap=None, cov=None, J=None):
    return np.trace(project_noise_to_plane(cov, J[:, :, 0], J[:, :, 1]), axis1=1, axis2=2).mean()

def feamap_cov_J_noise_ratio(feamap=None, cov=None, J=None):
    return (np.trace(project_noise_to_plane(cov, J[:, :, 0], J[:, :, 1]), axis1=1, axis2=2) / np.trace(cov, axis1=1, axis2=2)).mean()

def feamap_cov_J_fisher(feamap=None, cov=None, J=None):
    return compute_fisher_info(J, np.linalg.inv(cov))

def feamap_cov_J_total_fisher(feamap=None, cov=None, J=None):
    return np.trace(compute_fisher_info(J, np.linalg.inv(cov)), axis1=1, axis2=2).mean()

class TorusShapeAnalyzer:
    def __init__(self, mesh_size, x_bound, y_bound, speed_bound, speed_win_size=0.05, dataroot=DATAROOT):
        self.mesh_size = mesh_size
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.speed_bound = speed_bound
        self.speed_win_size = speed_win_size
        self.dataroot = dataroot

        self.metric_functions = {
            'lattice_area': feamap_cov_J_lattice_area,
            'radius': feamap_cov_J_radius,
            'center_dist': feamap_cov_J_center_dist,
            'total_noise': feamap_cov_J_total_noise,
            'noise_proj': feamap_cov_J_noise_proj,
            'noise_ratio': feamap_cov_J_noise_ratio,
            'fisher': feamap_cov_J_fisher,
            'total_fisher': feamap_cov_J_total_fisher
        }

        # Metrics that require Jacobian (J) and covariance (cov)
        self.requires_jacobian = {'lattice_area', 'fisher', 'total_fisher', 'noise_proj', 'noise_ratio'}
        self.requires_cov = {'total_noise', 'fisher', 'total_fisher', 'noise_proj', 'noise_ratio'}

    def analyze(self, model_file_name, metric_type='ALL', iid_mode=False):
        '''
        model_file_name: the name of the model file
        metric_type: the type of metric to compute
        iid_mode: whether to compute the metrics in iid grid cell population. Mathemtatically, this removes the non-diagonal terms of the covariance matrix.
        '''
        speed_list = self.get_speed_list()
        models = self.load_models(model_file_name)
        result = self.process_models(models, speed_list, metric_type=metric_type, iid_mode=iid_mode)
        speed_list = np.tile( speed_list, (len(models), 1) )
        result['speed'] = speed_list
        return result

    def get_random_mesh_points(self, label, speed_value):
        x_in_speed_idx = (label[:, -1] - speed_value) < self.speed_win_size # One side
        x_in_speed = label[x_in_speed_idx]
        try:
            query_mesh_space_idx = np.random.choice(x_in_speed.shape[0], self.mesh_size, replace=True)
        except:
            speed_min, speed_max = label[:, -1].min(), label[:, -1].max()
            print(f'Not enough samples for speed {speed_value}. Speed ranges from {speed_min} to {speed_max}.')
        query_mesh = x_in_speed[query_mesh_space_idx]
        return query_mesh

    def get_speed_list(self):
        return np.arange(self.speed_bound[0], self.speed_bound[1], 0.05)

    def load_models(self, model_file_name):
        model_file_name = os.path.join(self.dataroot, model_file_name)
        return hkl.load(model_file_name)

    def process_models(self, models, speed_list, metric_type='ALL', iid_mode=False):
        if metric_type == 'ALL':
            metric_type = self.metric_functions.keys()

        # Initialize a dictionary to store metric lists
        metrics_dict = {metric: [] for metric in metric_type}

        for i_model, model in enumerate(models):
            print(f'Processing model {i_model+1}/{len(models)}')

            # Initialize lists to collect the metrics for each speed
            speed_metrics = {metric: [] for metric in metric_type}

            for speed_value in speed_list:
                query_mesh = self.get_random_mesh_points(model.x, speed_value)

                # Determine if cov and J are needed
                compute_cov = any(metric in self.requires_cov for metric in metric_type)
                feamap, cov = model.predict(query_mesh, return_cov=compute_cov)
                if compute_cov and iid_mode:
                    n_sample, n_feature, _ = cov.shape
                    mask = np.eye(n_feature)
                    cov = cov * mask[None, :, :]

                J = None
                if any(metric in self.requires_jacobian for metric in metric_type):
                    model_wrap = create_fitter_wrapper(model)
                    J = compute_jacobian_central(model_wrap, query_mesh, h=0.01)

                # Compute metrics based on the requested types
                for metric in metric_type:
                    value = self.metric_functions[metric](feamap, cov, J)
                    speed_metrics[metric].append(value)

            # Store the results for this model
            for metric in metric_type:
                metrics_dict[metric].append(speed_metrics[metric])

        return metrics_dict

def save_torus_shape_results(dataroot, filename, result):
    # Apply transformations only to the specified metrics
    if 'speed' in result:
        result['speed'] = np.array(result['speed']) * 100
    
    metric_transformations = { # change unit from m to cm
        'lattice_area': lambda x: np.array(x) / 10000,
        'radius': np.array,
        'center_dist': np.array,
        'total_noise': np.array,
        'noise_proj': np.array,
        'noise_ratio': np.array,
        'fisher': lambda x: np.array(x) / 10000,
        'total_fisher': lambda x: np.array(x) / 10000
    }

    for metric in result.keys():
        if metric in result and metric in metric_transformations:
            result[metric] = metric_transformations[metric](result[metric])

    output_file = os.path.join(dataroot, filename)
    hkl.dump(result, output_file)

def create_fitter_wrapper(model):
    def fitter_wrapper(query_mesh):
        return model.predict(query_mesh, return_cov=False)[0]
    return fitter_wrapper
