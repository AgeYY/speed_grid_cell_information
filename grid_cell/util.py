import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
import scipy.stats as stats
from grid_cell.linear_regression import BayesianLinearRegression
from global_setting import *

def select_arr_by_t(arr, t, session='open_field_1', file_name='rat_q_grid_modules_1_2.npz'):
    if file_name == 'rat_q_grid_modules_1_2.npz':
        time_win = {'sleep_box_1': [9576, 18812], 'wagon_wheel_1': [18977, 25355], 'sleep_box_2': [25403, 27007], 'open_field_1': [27826, 31222]}
    elif file_name == 'rat_r_day1_grid_modules_1_2_3.npz':
        time_win = {'open_field_1': [7457, 14778], 'wagon_wheel_1': [16925, 18026], 'wagon_wheel_2': [20895, 21640]}
    elif file_name == 'rat_r_day2_grid_modules_1_2_3.npz':
        time_win = {'sleep_box_1': [396, 9941], 'open_field_1': [10617, 13004], 'sleep_box_2': [13143, 15973]}
    elif file_name == 'rat_s_grid_modules_1.npz':
        time_win = {'open_field_1': [9939, 12363], 'wagon_wheel_1': [13670, 14847], 'sleep_box_1': [14942, 23133], 'wagon_wheel_2': [23186, 24936]}
    selected_arr = arr[np.logical_and(t >= time_win[session][0], t < time_win[session][1])]
    return selected_arr

def apply_each_col(arr, func, *args, **kwargs):
    applied_arr = np.array([func(arr[:, i], *args, **kwargs) for i in range(arr.shape[1])])
    return np.moveaxis(applied_arr, 0, -1)

def get_data_filename_by_keywords(mouse_name, day):
    '''
    inputs:
        keywords: a list of keywords
        string: a list of string
    output:
        a list of string that contains all the keywords
    '''
    if mouse_name == 's':
        return 'rat_s_grid_modules_1.npz'
    elif mouse_name == 'q':
        return 'rat_q_grid_modules_1_2.npz'
    elif mouse_name == 'r':
        if day == 'day1':
            return 'rat_r_day1_grid_modules_1_2_3.npz'
        elif day == 'day2':
            return 'rat_r_day2_grid_modules_1_2_3.npz'
    else:
        raise ValueError('Invalid mouse name and day')

def remove_ax_frame(ax):
    for spine in ax.spines.values(): # remove the frame
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

class Shuffled_Matrix:
    def __init__(self, data, shuffle_method='permutation'):
        '''
        shuffle_method: time_shift or permutation
        '''
        self.data = data.copy()
        self.n_time, self.n_neuron = data.shape
        self.shuffle_method = shuffle_method

    def __getitem__(self, idx):
        if self.shuffle_method == 'time_shift':
            shuffled_data = np.empty_like(self.data, dtype=np.float32)
            for j in range(self.n_neuron):
                shift = np.random.randint(self.n_time)
                shuffled_data[:, j] = np.roll(self.data[:, j], shift)
        elif self.shuffle_method == 'permutation':
            shuffled_data = np.apply_along_axis(np.random.permutation, 0, self.data)
        
        return shuffled_data.copy()

    def __len__(self):
        return self.data.shape[0]


def get_line(ax, slope, intercept, **kwargs):
    x = np.array(ax.get_xlim())
    y = intercept + slope * x
    return x, y

def plot_horizontal_line_with_error_band(x_range, y_mean, y_std, ax, label=None):
    """
    Plot a horizontal dashed line with a grey error band.

    Parameters:
    x_range (tuple): Tuple of (start, end, num_points) for x-axis range.
    y_mean (float): The y-value for the mean (central line).
    y_std (float): The standard deviation for the error band.
    """
    x_values = np.linspace(*x_range)
    
    ax.plot(x_values, [y_mean]*len(x_values), 'k--', label=label)  # Red dashed horizontal line
    ax.fill_between(x_values, y_mean - y_std, y_mean + y_std, color='grey', alpha=0.5)  # Grey error band
    return ax

def remove_outliers(data):
    q25 = np.percentile(data, 25)
    q75 = np.percentile(data, 75)
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

def clean_and_mean_se(data_row):
    '''
    clean outliers then calculate mean and standard error
    '''
    q25, q75 = np.percentile(data_row, [25, 75])
    iqr = q75 - q25
    filter_mask = (data_row >= q25 - 1.5 * iqr) & (data_row <= q75 + 1.5 * iqr)
    filtered_row = data_row[filter_mask]
    mean = filtered_row.mean() if filtered_row.size > 0 else np.nan
    stderr = filtered_row.std(ddof=1) / np.sqrt(filtered_row.size) if filtered_row.size > 1 else np.nan
    return mean, stderr

def convert_to_circular_form(labels, circular_periods):
    n_samples, n_labels = labels.shape
    label_converted = np.empty((n_samples, n_labels * 2))

    for i in range(n_labels):
        angle = 2 * np.pi * labels[:, i] / circular_periods[i]
        label_converted[:, 2*i] = np.cos(angle)   # Cosine component
        label_converted[:, 2*i+1] = np.sin(angle) # Sine component

    return label_converted

def invert_to_original_form(label_converted, circular_periods):
    n_samples, _ = label_converted.shape
    n_labels = len(circular_periods)
    labels = np.empty((n_samples, n_labels))
    
    for i in range(n_labels):
        cos_component = label_converted[:, 2*i]
        sin_component = label_converted[:, 2*i+1]
        angle = np.arctan2(sin_component, cos_component)
        labels[:, i] = np.mod(angle / (2 * np.pi) * circular_periods[i], circular_periods[i])

    return labels

def pca_accumulated_variance_explained_ratio(data, cumsum=True):
    '''
    data shape: (n_samples, n_features)
    '''
    pca = PCA(n_components=None)
    pca.fit(data)
    if data.shape[0] < data.shape[1]: # if n_samples < n_features, extend the explained variance ratio to n_features by appending zeros
        var_ratio = np.append(pca.explained_variance_ratio_, np.zeros(data.shape[1] - data.shape[0]))
    else:
        var_ratio = pca.explained_variance_ratio_

    if cumsum:
        return var_ratio.cumsum()
    else:
        return var_ratio

def reduce_true_elements(array, m):
    # Check if m is valid
    if m > np.sum(array):
        raise ValueError("m cannot be greater than the number of True elements in the array")

    # Find indices of True elements
    true_indices = np.where(array)[0]

    # Randomly choose elements to set to False
    indices_to_flip = np.random.choice(true_indices, size=(len(true_indices) - m), replace=False)

    # Set chosen elements to False
    array[indices_to_flip] = False

    return array

def gram_schmidt(vectors, normalize=False):
    """
    Perform the Gram-Schmidt process on a set of vectors.
    
    Parameters:
    vectors (list of list of floats): A list where each element is a vector (also a list of floats). Or a numpy array with shape (n_vectors, n_dimensions)
    normalize (bool): If True, output orthonormal vectors. If False, output orthogonal vectors.
    
    Returns:
    np.ndarray: An array of orthogonal or orthonormal vectors. Shape is (n_vector, n_dimensions), same as the input vectors.
    """
    # Convert the input list of vectors to a numpy array
    vectors = np.array(vectors)
    # Initialize an empty list to store the orthogonal vectors
    orthogonal_vectors = []
    
    for v in vectors:
        # Start with the current vector
        u = v
        for u_prev in orthogonal_vectors:
            # Subtract the projection of v onto each of the previous orthogonal vectors
            u = u - proj_vec(v, u_prev)
        orthogonal_vectors.append(u)
    
    orthogonal_vectors = np.array(orthogonal_vectors)
    
    if normalize:
        # Normalize the orthogonal vectors to get an orthonormal set
        orthogonal_vectors = np.array([u / np.linalg.norm(u) for u in orthogonal_vectors])
    
    return orthogonal_vectors.T

def project_to_orthogonal_subspace(vector, orthogonal_matrix):
    """
    Compute the projection of a vector onto the subspace formed by the columns of an orthogonal matrix.
    
    Parameters:
    orthogonal_matrix (np.ndarray): A matrix of shape (m, n_dim) where each row represents an orthogonal vector.
    vector (np.ndarray): A vector of shape (n_dim,).
    
    Returns:
    np.ndarray: The projection of the vector onto the subspace formed by the columns of the orthogonal matrix.
    """
    # Convert the input to numpy arrays
    orthogonal_matrix = np.array(orthogonal_matrix)
    vector = np.array(vector)
    
    # Get the dimension of the vectors
    n_dim = vector.shape[0]
    
    # Compute the projection of the vector onto the subspace
    projection = np.zeros(n_dim)
    for u in orthogonal_matrix:
        projection += proj_vec(vector, u)
    
    return projection

def proj_vec(vec0, vec1):
    '''
    project vec0 onto vec1
    '''
    return np.dot(vec0, vec1) / np.dot(vec1, vec1) * vec1

def angle_two_vec(vec0, vec1):
    '''
    calculate the angle between two vectors
    input:
    vec0: vec1: 1D numpy array. (n_dim,)
    vec1: 1D numpy array. (n_dim,)
    output:
    angle: float. the angle between two vectors. unit is rad, ranges from 0 to np.pi
    '''
    return np.arccos(np.dot(vec0, vec1) / np.linalg.norm(vec0) / np.linalg.norm(vec1))

def digitize_arr(arr, n_bins, bound):
    bin_edges = np.linspace(bound[0], bound[1], n_bins + 1)
    bin_value = (bin_edges[:-1] + bin_edges[1:]) / 2

    binned_positions = np.digitize(arr, bin_edges, right=True)
    binned_positions = np.clip(binned_positions - 1, 0, n_bins - 1)
    return bin_value[binned_positions]

def compute_correlation_and_p_values(matrix):
    """
    Computes the correlation matrix and the corresponding p-values for a given matrix.

    Parameters:
    matrix (numpy.ndarray): Input matrix of shape (n, k) where n is the number of samples and k is the number of features.

    Returns:
    tuple: A tuple containing the correlation matrix and the p-value matrix.
    """
    n_features = matrix.shape[1]
    correlation_matrix = np.zeros((n_features, n_features))
    p_value_matrix = np.zeros((n_features, n_features))

    # Compute correlations and p-values
    for i in range(n_features):
        for j in range(n_features):
            if i > j:
                corr, p_val = stats.pearsonr(matrix[:, i], matrix[:, j])
                correlation_matrix[i, j] = corr
                p_value_matrix[i, j] = p_val

    return correlation_matrix, p_value_matrix


def permute_columns(matrix):
    # Get the number of columns
    num_cols = matrix.shape[1]
    
    # Permute each column independently
    for col in range(num_cols):
        np.random.shuffle(matrix[:, col])
    
    return matrix

def fisher_combined_p_value(p_values):
    """
    Combine p-values using Fisher's Combined Probability Test.

    Parameters:
    p_values (list): A list of p-values to combine.

    Returns:
    float: The combined p-value.
    """
    chi2_statistic = -2 * np.sum(np.log(p_values))
    degrees_of_freedom = 2 * len(p_values)
    combined_p_value = stats.chi2.sf(chi2_statistic, degrees_of_freedom)
    
    return combined_p_value
