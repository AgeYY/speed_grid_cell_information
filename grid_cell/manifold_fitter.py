from global_setting import *
from scipy.ndimage import gaussian_filter
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from scipy.stats import multivariate_normal
from sklearn.covariance import ledoit_wolf
import copy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import gpflow
import grid_cell.tuning as gct

class Avg_Fitter():
    def __init__(self, circular_period=None, bin_size_mean=0.5, bin_size_cov=0.5, hyper_bound=[(0.05, 2)], avg_method='kernel', bin_search=True, minimum_num_points_bin_avg=0, fit_cov_bin_size=True, use_bin_lw=False):
        '''
        avg_method: str, bin or kernel
        bin_search: bool, whether to search for the optimal bin size
        bin_size_mean: list. Each one is the bin size for each dimension
        bin_size_cov: list. Each one is the bin size for each dimension
        hyper_bound: list. Each one is the bound for each dimension
        circular_period: float. The period of the circular manifold. If None, it is a large number
        minimum_num_points_bin_avg: int. The minimum number of points to compute the bin average
        use_bin_lw: bool. Whether to use ledoit_wolf to estimate the covariance matrix in each bin. This will only works if avg_method is 'bin'
        '''
        self.LARGE_NUM = 9999
        self.bin_size_mean = bin_size_mean
        self.bin_size_cov = bin_size_cov
        self.hyper_bound = copy.deepcopy(hyper_bound)
        self.avg_method = avg_method
        self.bin_search = bin_search
        self.minimum_num_points_bin_avg = minimum_num_points_bin_avg
        self.fit_cov_bin_size = fit_cov_bin_size
        self.use_bin_lw = use_bin_lw
        if circular_period is None:
            self.circular_period = self.LARGE_NUM
        else:
            self.circular_period = circular_period


    def fit(self, feamap, label):
        self.feamap = feamap.copy()
        self.label = label.copy()

        if self.bin_search:
            feamap_cp, label_cp = feamap.copy(), label.copy()
            feamap_train, self.feamap_val_for_bin, label_train, self.label_val_for_bin = train_test_split(self.feamap, self.label, test_size=0.2) # Split training data into train and validation sets
            self.feamap, self.label = feamap_train, label_train

            result = minimize(self._bin_size_mean_loss, x0=self.bin_size_mean * np.ones(self.label.shape[1]), bounds=self.hyper_bound)
            self.bin_size_mean = result.x

            if self.fit_cov_bin_size:
                result = minimize(self._bin_size_cov_loss, x0=self.bin_size_cov * np.ones(self.label.shape[1]), bounds=self.hyper_bound)
                self.bin_size_cov = result.x
            else:
                self.bin_size_cov = self.bin_size_mean

            print('optimal bin size mean:', self.bin_size_mean, 'optimal bin size cov:',self.bin_size_cov)

            # Restore the original feature map and label data
            self.feamap, self.label = feamap_cp, label_cp
        return None

    def predict_mean_cov(self, query_mesh, bin_size_mean=None, bin_size_cov=None, output_valide_index=False, compute_mode='mean'):
        '''
        inputs:
        query_mesh: [n_sample, n_label], the query mesh
        bin_size_mean: list. Each one is the bin size for each dimension
        bin_size_cov: list. Each one is the bin size for each dimension
        output_valide_index: bool. Whether to output the valid index
        compute_mode: str, mean or cov
        outputs:
        result: [n_sample, n_feature] the predicted mean or [n_sample, n_feature, n_feature] for the predicted covariance
        query_mesh_valid: [n_sample, n_label], the valid query mesh
        sample_idx: [n_sample], the valid index
        '''
        if compute_mode == 'mean': bin_size = bin_size_mean if bin_size_mean is not None else self.bin_size_mean
        elif compute_mode == 'cov': bin_size = bin_size_cov if bin_size_cov is not None else self.bin_size_cov
        result = []
        query_mesh_valid = []
        sample_idx = np.full(len(query_mesh), False)

        for i, query_value in enumerate(query_mesh):
            diff = self._calculate_diff(self.label, query_value)

            if self.avg_method == 'bin':
                condition = np.all(diff < bin_size, axis=1)
                idx = np.where(condition)[0]

                if idx.shape[0] >= self.minimum_num_points_bin_avg:
                    sample_idx[i] = True
                    query_mesh_valid.append(query_value)
                    if compute_mode == 'mean':
                        result.append(np.mean(self.feamap[idx.flatten()], axis=0))
                    elif compute_mode == 'cov':
                        if self.use_bin_lw:
                            result.append(ledoit_wolf(self.feamap[idx.flatten()])[0])
                        else:
                            result.append(np.cov(self.feamap[idx.flatten()].T))
                    else:
                        raise ValueError("Invalid compute_mode: {}".format(compute_mode))
            elif self.avg_method == 'kernel':
                sample_idx[i] = True
                query_mesh_valid.append(query_value)

                scaled_diff = (diff / bin_size) ** 2
                y_value = -np.sum(scaled_diff, axis=1)
                exp_y = np.exp(0.5 * (y_value - np.max(y_value)))
                weights = exp_y / np.sum(exp_y) # equivalent to the normalized gaussian kernel

                if compute_mode == 'mean':
                    result.append(np.average(self.feamap, axis=0, weights=weights.flatten()))
                elif compute_mode == 'cov':
                    result.append(np.cov(self.feamap.T, aweights=weights.flatten()))
                    contains_invalid = np.isnan(result[-1]).any() or np.isinf(result[-1]).any()
                    if contains_invalid:
                        print('cov predict invalid', contains_invalid)
                else:
                    raise ValueError("Invalid compute_mode: {}".format(compute_mode))
            else:
                raise ValueError("Invalid avg_method: {}".format(avg_method))

        if output_valide_index:
            return (np.array(result), np.array(query_mesh_valid)) + (sample_idx,)
        else:
            return np.array(result), np.array(query_mesh_valid)

    def predict_1d_mean(self, query_mesh, bin_size_mean=None, output_valide_index=False):
        '''
        !!! Only to keep the interface consistent with the other predict functions
        '''
        return self.predict_mean_cov(query_mesh, bin_size_mean=bin_size_mean, output_valide_index=output_valide_index, compute_mode='mean')

    def predict_1d_cov(self, query_mesh, bin_size_cov=None, output_valide_index=False):
        '''
        !!! Only to keep the interface consistent with the other predict functions
        '''
        return self.predict_mean_cov(query_mesh, bin_size_cov=bin_size_cov, output_valide_index=output_valide_index, compute_mode='cov')

    def predict_1d(self, query_mesh, bin_size_mean=None, bin_size_cov=None, return_cov=False, output_valide_index=False):
        '''
        Do not be misled by the name. This function is used to predict any dimensions (e.g. surface), not only 1d (curve). We use this name because the name 'predict' was already used by the old code.
        inputs:
            query_mesh: 2d array (n_query, n_label)
            bin_size_mean: 1d array (n_label)
            bin_size_cov: 1d array (n_label)
            return_cov: bool. If yes, return the covariance of the prediction
            output_valide_index: bool. If yes, output which query_mesh is valid, i.e. have data within the bin
        outputs:
            pred: 2d array (n_valid_query, n_feature)
            query_mesh_valid: 2d array (n_valid_query, n_label)
            cov: 3d array (n_valid_query, n_label, n_label)
        '''
        if bin_size_mean is None: bin_size_mean = self.bin_size_mean
        if bin_size_cov is None: bin_size_cov = self.bin_size_cov
    
        pred, query_mesh_valid, sample_idx = self.predict_mean_cov(query_mesh, bin_size_mean=bin_size_mean, output_valide_index=True, compute_mode='mean')

        if return_cov:
            cov, query_mesh_valid_cov, sample_idx_cov = self.predict_mean_cov(query_mesh, bin_size_cov=bin_size_cov, output_valide_index=True, compute_mode='cov')

            if output_valide_index:
                return pred, query_mesh_valid, sample_idx, cov, query_mesh_valid_cov, sample_idx_cov
            else:
                return pred, query_mesh_valid, cov, query_mesh_valid_cov

        if output_valide_index:
            return pred, query_mesh_valid, sample_idx
        else:
            return pred

    def predict(self, n_spatial_bin, x_bound=None, y_bound=None, remove_unvisited=True):
        '''
        !!! This is an old version
        input:
            n_spatial_bin: number of spatial bins
            x_bound, y_bound: the bound of the position. If None, use the min and max of x, y
        output:
            rate_map: 2d array. The firing rate (zscored) at each spatial bin
            positions: 2d array. The position of each spatial bin
        '''
        x, y = self.label[:, 0], self.label[:, 1]
        # compute the average at each location
        n_cell = self.feamap.shape[1]
        rate_map = np.zeros([n_spatial_bin, n_spatial_bin, n_cell])
        for i in range(n_cell):
            rate_map[:, :, i], positions = gct.spatial_bin_fire_rate(self.feamap[:, i], x, y, n_bin=n_spatial_bin, x_bound=x_bound, y_bound=y_bound, verbose=True)

        # rate_map = np.nan_to_num(rate_map)
        # rate_map = gaussian_filter(rate_map, 2.75, axes=[0, 1]) # smooth the firing rate

        # plt.figure()
        # plt.imshow(rate_map[:, :, 0])
        # plt.show()

        rate_map = rate_map.reshape(-1, n_cell)
        positions = positions.reshape(-1, 2)

        if remove_unvisited:
            visited = ~np.isnan(rate_map).any(axis=1)
            positions = positions[visited]
            rate_map = rate_map[visited]
        return rate_map, positions

    def _calculate_diff(self, labels, query_value):
        '''
        Calculate the difference between the labels and the query value, considering the circular property.
        inputs:
            labels: 2d array (n_sample, n_label)
            query_value: 1d array (n_label)
        '''
        diff = np.mod(np.abs(labels - query_value), self.circular_period)
        return np.where(diff > self.circular_period/2.0, self.circular_period - diff, diff)

    def _bin_size_mean_loss(self, bs):
        """
        Calculate the mean squared error loss for a given bin size.

        Parameters:
        - bs (int): The bin size to be evaluated.

        Returns:
        - float: The mean squared error loss, or a large number if there are no predictions.
        """
        # Predict on the validation data
        pred, qm_valid, idx = self.predict_1d_mean(self.label_val_for_bin, bin_size_mean=bs, output_valide_index=True)

        # Return a large number if there are no predictions, otherwise calculate mean squared error
        if len(pred) == 0:
            return self.LARGE_NUM
        else:
            return np.mean((self.feamap_val_for_bin[idx] - pred)**2)

    def negative_log_likelihood_mult(self, samples, means, covs):
        neg_log_likelihood = 0
        i_valide_sample = 0
        for sample, mean, cov in zip(samples, means, covs):
            try:
                neg_log_likelihood -= multivariate_normal.logpdf(sample, mean=mean, cov=cov)
                i_valide_sample += 1
            except np.linalg.LinAlgError: # skip singular matrix
                continue
        if i_valide_sample == 0:
            return self.LARGE_NUM
        else:
            neg_log_likelihood /= i_valide_sample
        return neg_log_likelihood

    def _bin_size_cov_loss(self, bs):
        # Predict on the validation data
        pred, qm_valid, sample_idx = self.predict_1d_mean(self.label_val_for_bin, bin_size_mean=self.bin_size_mean, output_valide_index=True)
        cov, sample_idx_cov = self.predict_1d_cov(self.label_val_for_bin, bin_size_cov=bs)
        loss = self.negative_log_likelihood_mult(self.feamap_val_for_bin[sample_idx], pred, cov)

        return loss

def create_kernel(circular_period=None, input_dim=1):
    """
    Create a GPflow kernel based on the provided circular_period specification.

    Parameters:
    circular_period (None, scalar, or list): Specifies the periodicity for the kernel.
        - If None, a non-periodic kernel is created.
        - If a scalar, all dimensions use the same periodicity.
        - If a list, each dimension gets its specified periodicity; None for non-periodic.

    Returns:
    gpflow.kernels.Kernel: The constructed GPflow kernel.
    """
    var = [1] * input_dim
    lengthscales = [1] * input_dim
    var_white = [1] * input_dim
    var_const = [1] * input_dim
    # Case 1: No periodicity specified, create a standard kernel
    if circular_period is None:
        return gpflow.kernels.SquaredExponential(lengthscales=lengthscales) + gpflow.kernels.White() + gpflow.kernels.Constant()

    # Case 2: Single scalar provided, apply the same periodicity to all dimensions
    if np.isscalar(circular_period):
        return gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(lengthscales=lengthscales), period=circular_period) + gpflow.kernels.White() + gpflow.kernels.Constant()

    # Case 3: A list of periods is provided, create individual kernels for each dimension.
    if isinstance(circular_period, list):
        kernels = []
        for i, period in enumerate(circular_period):
            # For each dimension, create a periodic kernel if period is specified, otherwise a standard kernel
            if period is None:
                kernels.append(gpflow.kernels.SquaredExponential(active_dims=[i]))
            else:
                print(f"Period: {i, period}")
                base_kernel = gpflow.kernels.SquaredExponential(active_dims=[i])
                kernels.append(
                    gpflow.kernels.Periodic(base_kernel, period=period)
                               )
        # Combine all the kernels together
        combined_kernel = gpflow.kernels.Product(kernels) + gpflow.kernels.White() + gpflow.kernels.Constant()

        return combined_kernel

    # Raise an error if the input doesn't match the expected format
    raise ValueError("Invalid input for circular_period")

class GP_Fitter():
    def __init__(self, kernel=None, circular_period=None, standardize=False, n_inducing=100, seperate_kernel=False):
        '''
        kernel (): The kernel to use for the GP model. Can be None to use the default kernel settings.
        circular_period (None, scalar, or list): Specifies the periodicity for the kernel.
         - If None, a non-periodic kernel is created.
         - If a scalar, all dimensions use the same periodicity.
         - If a list, each dimension gets its specified periodicity; None for non-periodic.
        '''
        self.n_inducing = n_inducing # if None --> GPR, else SGPR
        self.circular_period = circular_period
        self.kernel = kernel

        self.standardize = standardize
        self.standardize_feamap_mean = 0
        self.standardize_feamap_std = 1
        self.seperate_kernel = seperate_kernel


    def fit(self, feamap, label):
        # standardize the data
        if self.standardize:
            self.standardize_feamap_mean = feamap.mean(axis=0)
            self.standardize_feamap_std = feamap.std(axis=0)
            feamap_cp = (feamap - self.standardize_feamap_mean) / self.standardize_feamap_std
        else:
            feamap_cp = feamap.copy()

        if self.seperate_kernel: # If using SVGP, the seperate kernel method is built-in in the GPFlow. However, I found SVGP is slower than SGPR. SGPR works well for multiple input, or shared kernel, but somehow not for seperate kernel. So I implement the seperate kernel model class here.
            if self.kernel is None: self.kernel = [create_kernel(circular_period=self.circular_period, input_dim=label.shape[1]) for _ in range(feamap_cp.shape[1])]
            self.model = MultiKernelGPModel(self.kernel, n_inducing=self.n_inducing)
        else:
            if self.kernel is None: self.kernel = create_kernel(circular_period=self.circular_period, input_dim=label.shape[1])
            self.model = SharedKernelGPModel(self.kernel, n_inducing=self.n_inducing)

        self.model.fit(label, feamap_cp)

    def predict(self, query_mesh=None, x_bound=[-0.8, 0.8], y_bound=[-0.8, 0.8], n_spatial_bin=50):
        '''
        input:
            query_mesh: 2d array. The position of each spatial bin. If none, mesh will be generated based bounds and n_spatial_bin
            x_bound, y_bound: the bound of the position. If None, use the min and max of x, y
            n_spatial_bin: number of spatial bins
        output:
            pred: 2d array. The predicted firing rate at each spatial bin
            query_mesh: 2d array. The position of each spatial bin
        '''
        if query_mesh is None:
            mesh_bound = [x_bound, y_bound]
            query_mesh = label_mesh(mesh_bound, mesh_size=n_spatial_bin, random=False, grid=True)

        pred, _ = self.model.predict_f(query_mesh)

        # unstandardize the data
        if self.standardize:
            pred = pred * self.standardize_feamap_std + self.standardize_feamap_mean

        return pred, query_mesh


def label_mesh(mesh_bound, mesh_size, random=False, grid=False):
    '''
    input:
        mesh_bound: a list of bound of each dimension, e.g. [[0, 1], [0, 1]]
        mesh_size: number of points. If grid is True, mesh_size is the number of points in each dimension, so total number of points is mesh_size ** len(mesh_bound). Otherwise, mesh_size is the number of points in total
        random: whether to generate random mesh
        grid: whether to generate grid mesh
    output:
        label_mesh: 2d array. The mesh of the label
    '''
    n_label = len(mesh_bound)
    label_mesh_list = []
    for i in range(n_label):
        if random:
            label_mesh_list.append( np.random.uniform(mesh_bound[i][0], mesh_bound[i][1], mesh_size) )
        else:
            label_mesh_list.append( np.linspace(mesh_bound[i][0], mesh_bound[i][1], mesh_size) )

    if grid: label_mesh = np.array( np.meshgrid(*label_mesh_list)).transpose().reshape( (-1, n_label) )
    else: label_mesh = np.array(label_mesh_list).transpose()

    return label_mesh

class SharedKernelGPModel:
    def __init__(self, kernel, n_inducing=None):
        self.kernel = kernel
        self.n_inducing = n_inducing
        self.model = None

    def fit(self, label, feamap_cp):

        if self.n_inducing is None:
            label_ = tf.cast(label, dtype=FLOAT_TYPE)
            feamap_cp_ = tf.cast(feamap_cp, dtype=FLOAT_TYPE)
            self.model = gpflow.models.GPR(
                (label_, feamap_cp_),
                kernel=self.kernel,
            )
        else:
            self.n_inducing = min(self.n_inducing, label.shape[0])
            iv_idx = np.random.choice(np.arange(label.shape[0]), self.n_inducing, replace=False)
            iv = label[iv_idx]

            label_ = tf.cast(label, dtype=FLOAT_TYPE)
            feamap_cp_ = tf.cast(feamap_cp, dtype=FLOAT_TYPE)
            iv_ = tf.cast(iv, dtype=FLOAT_TYPE)
            self.model = gpflow.models.SGPR(
                (label_, feamap_cp_),
                self.kernel,
                inducing_variable=iv_,
            )

        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.model.training_loss, self.model.trainable_variables, method="l-bfgs-b")

    def predict_f(self, Xnew):
        Xnew_ = tf.cast(Xnew, dtype=FLOAT_TYPE)
        pred, var = self.model.predict_f(Xnew_)
        return pred.numpy(), var.numpy()

class MultiKernelGPModel:
    def __init__(self, kernel_list, n_inducing=None):
        self.kernel_list = kernel_list
        self.n_inducing = n_inducing
        self.models = []

    def fit(self, label, feamap_cp):
        for i, kernel in enumerate(self.kernel_list):
            model = SharedKernelGPModel(kernel, n_inducing=self.n_inducing)
            model.fit(label, feamap_cp[:, [i]])
            self.models.append(model)

    def predict_f(self, Xnew):
        means = []
        variances = []
        for model in self.models:
            mean, var = model.predict_f(Xnew)
            means.append(mean)
            variances.append(var)

        mean_concat = np.concatenate(means, axis=1)
        variance_concat = np.concatenate(variances, axis=1)
        
        return mean_concat, variance_concat
