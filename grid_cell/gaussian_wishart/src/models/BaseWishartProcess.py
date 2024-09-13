import numpy as np
import gpflow
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from grid_cell.gaussian_wishart.src.likelihoods.WishartProcessLikelihood import WishartLikelihood
from grid_cell.gaussian_wishart.src.models .SVGP_deprecated import SVGP_deprecated

class WishartProcessBase(SVGP_deprecated):
    """
    Wrapper around gpflow's SVGP class, with added functionality for estimating the covariance matrix.
    Class written by Creighton Heaukulani and Mark van der Wilk, and is adapted for gpflow 2.
    """
    def __init__(self, kernel, likelihood=None, D=1, nu=None, inducing_variable=None,
                 q_mu=None, q_sqrt=None, num_data=None):
        """
        :param kernel (gpflow.Kernel object)
        :param likelihood (gpflow.likelihood object)
        :param D (int) Covariance matrix dimension
        :param nu (int) Degrees of freedom
        :param inducing_variable ()
        """
        nu = D if nu is None else nu
        likelihood = WishartLikelihood(D, nu, R=10) if likelihood is None else likelihood

        super().__init__(kernel=kernel,
                         likelihood=likelihood,
                         num_latent_gps=int(D * nu),
                         inducing_variable=inducing_variable,
                         q_mu=q_mu,
                         q_sqrt=q_sqrt,
                         num_data=num_data)


    def predict_mc(self, X_test, n_samples):
        """
        Returns monte carlo samples of the covariance matrix $\Sigma$
        Abstract method, should be implemented by concrete class.

        :param X_test (N_test, D) input locations to predict covariance matrix over.
        :param n_samples (int) number of samples to estimate covariance matrix at each time point.
        :return Sigma (n_samples, N_test, D, D) covariance matrix sigma
        """
        raise NotImplementedError

    def predict_map(self, X_test):
        """
        Returns MAP estimate of the covariance matrix $\Sigma$
        Abstract method, should be implemented by concrete class.

        :param X_test (N_test, D) input locations to predict covariance matrix over.
        :param Y_test (N_test, D) observations to predict covariance matrix over.
        :return Sigma (N_test, D, D) covariance matrix sigma
        """
        raise NotImplementedError

    def get_additive_noise(self, n_samples):
        """
        Get n samples of white noise to add on diagonal matrix
        :param n_samples
        :return Lambda Additive white noise
        """
        sigma2inv_conc = self.likelihood.q_sigma2inv_conc
        sigma2inv_rate = self.likelihood.q_sigma2inv_rate

        dist = tfd.Gamma(sigma2inv_conc, sigma2inv_rate)
        sigma2_inv = dist.sample([n_samples])  # (R, D)
        sigma2_inv = tf.clip_by_value(sigma2_inv, 1e-8, np.inf)

        if self.likelihood.model_inverse:
            Lambda = sigma2_inv[:, None, :]
        else:
            sigma2 = np.power(sigma2_inv, -1.0)
            Lambda = sigma2[:, None, :]
        if Lambda.shape[0]==1:
            Lambda = np.reshape(Lambda, -1)
        return Lambda
