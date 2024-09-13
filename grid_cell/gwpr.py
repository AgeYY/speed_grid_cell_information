from gpflow.kernels import SharedIndependent
from grid_cell.gaussian_wishart.src.likelihoods.WishartProcessLikelihood import *
from grid_cell.gaussian_wishart.src.likelihoods.FactorizedWishartLikelihood import *
from grid_cell.gaussian_wishart.src.models.WishartProcess import *
from grid_cell.gaussian_wishart.src.models.FactorizedWishartProcess import *
from grid_cell.gaussian_wishart.util.training_util import *
import gpflow
from gpflow.inducing_variables import SharedIndependentInducingVariables, InducingPoints
from grid_cell.manifold_fitter import GP_Fitter
from global_setting import *


class GWPR_Fitter():
    def __init__(self, n_factors=None, nu=None, R=100, model_inverse=False, multiple_observations=False, max_iter=1000, learning_rate=0.1, circular_period=None, n_inducing=100, standardize=True, minibatch_size=64, with_cov=True):
        '''
        !! current code only surpport model_inverse=False
        :param n_factors: [int], number of factors, which is also writen as K in the reference. Number of latent factors = n_factors * nu
        :param nu: [int], degree of freedom for the wishart process, must larger than n_factors
        :param R: [int], number of samples for variational expectation
        :param model_inverse: [bool], whether to model the inverse of the covariance matrix
        :param multiple_observations: [bool], how many repeated observations of responses for each input label
        :param max_iter: [int], maximum number of iterations for optimization in wishart
        :param learning_rate: [float], learning rate for optimization in wishart
        :param circular_period: [float], period of the circular manifold. Current code only supports same circular period for all inputs
        :param n_inducing: [int], number of inducing points
        :param standardize: [bool], whether to standardize the input data when using GPR estimating the mean
        :param minibatch_size: [int], size of minibatch
        :param with_cov: [bool], whether to fit the covariance
        '''
        self.nu = nu
        self.n_factors = n_factors
        self.model_inverse = model_inverse
        self.R = R
        self.multiple_observations = multiple_observations
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.circular_period = circular_period
        self.n_inducing = n_inducing
        self.standardize = standardize
        self.minibatch_size = minibatch_size
        self.with_cov = with_cov

    def fit(self, response_train, label_train):
        '''
        Fit the model
        :param response_train: [np.ndarray], shape (n_samples, D), responses
        :param label_train: [np.ndarray], shape (n_samples, n_labels), input labels
        '''
        self.n_inducing = np.min([self.n_inducing, response_train.shape[0]])
        #### Fit the mean
        self.gpr_mean = GP_Fitter(circular_period=self.circular_period, standardize=self.standardize, n_inducing=self.n_inducing)
        self.gpr_mean.fit(response_train, label_train)

        if self.with_cov:
            #### construct the wishart model
            self.D = response_train.shape[1]
            self.n_factors = self.D if self.n_factors is None else self.n_factors
            self.nu = self.n_factors + 1 if self.nu is None else self.nu
        
            if self.circular_period is not None:
                kernel = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(), period=self.circular_period)
            else:
                kernel = gpflow.kernels.SquaredExponential()

            latent_dim = self.n_factors * self.nu
            kernel = SharedIndependent(kernel, output_dim=latent_dim)
            iv = label_train[np.random.choice(label_train.shape[0], self.n_inducing, replace=False)]
            iv = SharedIndependentInducingVariables(InducingPoints(iv))  # multi output inducing variables

            likelihood = FactorizedWishartLikelihood(D=self.D, nu=self.nu, R=self.R, n_factors=self.n_factors, model_inverse=self.model_inverse, multiple_observations=self.multiple_observations)
            self.wishart = FactorizedWishartModel(kernel, likelihood, D=self.n_factors, nu=self.nu, inducing_variable=iv, num_data=response_train.shape[0])
            #### train the model
            r_train_pred, _ = self.gpr_mean.predict(label_train) # compute the mean first
            data = (label_train, response_train - r_train_pred) # subtract the mean
            run_adam(self.wishart, iterations=self.max_iter, learning_rate=self.learning_rate, data=data, minibatch_size=self.minibatch_size)

    def predict(self, label_test):
        '''
        Predict the response
        :param label_test: [np.ndarray], shape (n_samples, n_labels), input labels
        :return: [np.ndarray], shape (n_samples, D), predicted responses
        '''
        r_pred, _ = self.gpr_mean.predict(label_test) # predict the mean

        if self.with_cov:
            cov_pred = self.wishart.predict_map(label_test) # MAP estimate
        else:
            cov_pred = None

        return r_pred, cov_pred
