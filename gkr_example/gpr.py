from global_setting import *
import tensorflow as tf
from scipy.stats import norm
from scipy.stats import multivariate_normal
import numpy as np
import gpflow
import copy

class GP_Fitter():
    def __init__(self, kernel=None, circular_period=None, standardize=True, n_inducing=None, seperate_kernel=False):
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
