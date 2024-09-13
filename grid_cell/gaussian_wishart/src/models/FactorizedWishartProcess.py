from grid_cell.gaussian_wishart.src.models.WishartProcess import WishartProcessBase
import numpy as np
import gpflow
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from grid_cell.gaussian_wishart.src.likelihoods.WishartProcessLikelihood import WishartLikelihood

class FactorizedWishartModel(WishartProcessBase):
    def __init__(self, kernel, likelihood, **kwargs):
        super().__init__(kernel, likelihood, **kwargs)

    def predict_mc(self, X_test, n_samples):
        """
        Returns samples of the covariance matrix $\Sigma_n$ for each time point

        :param X_test: (N_test,D) input locations to predict covariance matrix over.
        :param n_samples: (int)
        :return: (n_samples, K, K)
         """
        A, cov_dim, nu = self.likelihood.A, self.likelihood.cov_dim, self.likelihood.nu
        N_test, _ = X_test.shape

        # Produce n_samples of F (latent GP points as the input locations X)
        f_sample = self.predict_f_samples(X_test, num_samples=n_samples)
        f_sample = tf.reshape(f_sample, [n_samples, N_test, cov_dim, -1])  # (n_samples, N_test, D, nu)

        # Construct Sigma from latent gp's
        AF = np.einsum('kl,ijlm->ijkm', A, f_sample) #np.einsum('kl,ijlm->ijkm', A, f_sample)
        affa = np.matmul(AF, np.transpose(AF, [0, 1, 3, 2]))  # (n_samples, N_test, D, D)
        if self.likelihood.additive_noise:
            Lambda = self.get_additive_noise(n_samples)
            affa = tf.linalg.set_diag(affa, tf.linalg.diag_part(affa) + Lambda)
        else:
            affa += 1e-6

        return affa

    def predict_map(self, X_test):
        """
        Get mean prediction
        :param X_test(N_test, D) input locations to predict covariance matrix over.
        :return: params (dictionary) contains the likelihood parameters, monitored by tensorboard.
        """
        A, D, nu, n_factors = self.likelihood.A, self.likelihood.D, self.likelihood.nu, self.likelihood.n_factors # A: (D, K = n_factors), others are integer
        N_test, _ = X_test.shape

        # Produce n_samples of F (latent GP points as the input locations X)
        mu, var = self.predict_f(X_test)  # (N_test, n_factors*nu)
        mu = tf.reshape(mu, [N_test, n_factors, -1])  # (N_test, n_factors, nu)
        AF = tf.einsum('kl,ilj->ikj', A, mu)  # (N_test, D, nu)
        affa = np.matmul(AF, np.transpose(AF, [0, 2, 1]))  # (N_test, D, D)
        if self.likelihood.additive_noise:
            Lambda = self.get_additive_noise(1)
            affa = tf.linalg.set_diag(affa, tf.linalg.diag_part(affa) + Lambda)
        else:
            affa += 1e-6
        return affa
