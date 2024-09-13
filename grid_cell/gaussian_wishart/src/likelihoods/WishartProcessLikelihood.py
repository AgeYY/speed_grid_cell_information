import numpy as np
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import positive
from gpflow import Parameter
from grid_cell.gaussian_wishart.src.likelihoods.BaseWishartLikelihood import WishartLikelihoodBase

class WishartLikelihood(WishartLikelihoodBase):
    """
    Concrete class for the full covariance likelihood models.
    The code is written by Heaukulani-van der Wilk (see references above)
    """
    def __init__(self, D, nu, A=None, **kwargs):
        """
        :param D (int) Dimensionality of covariance matrix
        :param nu (int) degrees of freedom
        :param A (DxD matrix) scale matrix. Default is a DxD identity matrix.
        """
        assert nu >= D, "Degrees of freedom must be larger or equal than the dimensionality of the covariance matrix"
        super().__init__(D, nu, **kwargs)
        self.cov_dim = D

        # this case assumes a square scale matrix, and it must lead with dimension D
        self.A = A if A is not None else Parameter(np.ones((self.D, self.D)), transform=positive(), dtype=tf.float64)
        gpflow.set_trainable(self.A, True)

        if self.additive_noise:
            self.q_sigma2inv_conc = Parameter(0.1 * np.ones(self.D), transform=positive(), dtype=tf.float64)
            self.q_sigma2inv_rate = Parameter(0.0001 * np.ones(self.D), transform=positive(), dtype=tf.float64)#

    def make_gaussian_components(self, F, Y):
        """
        An auxiliary function for logp that returns the complexity pentalty and the data fit term.
        Note: it is assumed that the mean function is 0.
        :param F: (R, N, D, __) - the (samples of the) matrix of GP outputs, where:
                R is the number of Monte Carlo samples,
                N is the numkber of observations and
                D the dimensionality of the covariance matrix.
        :param Y: (N, D) Tensor. observations
        :return:
            log_det_cov: (R,N) log determinant of the covariance matrix Sigma_n (complexity penalty)
            yt_inv_y: (R,N) (data fit term)
        """
        # Compute Sigma_n (aka AFFA)
        # AF = self.A[:, None] * F  # (R, N, D, nu)
        AF = tf.einsum('kl,ijlm->ijkm', self.A, F)
        AFFA = tf.matmul(AF, AF, transpose_b=True)  # (R, N, D, D)

        # additive white noise (Lambda) for numerical precision
        if self.additive_noise:
            n_samples = tf.shape(F)[0]  # could be 1 if making predictions
            dist = tfp.distributions.Gamma(self.q_sigma2inv_conc, self.q_sigma2inv_rate)
            sigma2_inv = dist.sample([n_samples])  # (R, D)
            sigma2_inv = tf.clip_by_value(sigma2_inv, 1e-8, np.inf)

            if self.model_inverse:
                Lambda = sigma2_inv[:, None, :]
            else:
                sigma2 = sigma2_inv**-1.
                Lambda = sigma2[:, None, :]
        else:
            Lambda = 1e-5

        # Compute log determinant of covariance matrix Sigma_n (aka AFFA)
        AFFA = tf.linalg.set_diag(AFFA, tf.linalg.diag_part(AFFA) + Lambda)
        L = tf.linalg.cholesky(AFFA)  # (R, N, D, D)
        log_det_cov = 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=2)  # (R, N)
        if self.model_inverse:
            log_det_cov = - log_det_cov

        # Compute (Y^time_window affa^-1 Y) term
        if self.model_inverse:
            y_prec = tf.einsum('jk,ijkl->ijl', Y, AFFA)  # (R, N, D)  # j=N, k=D, i=, l=
            yt_inv_y = tf.reduce_sum(y_prec * Y, axis=2)  # (R, N)

        else:
            n_samples = tf.shape(F)[0]  # could be 1 when computing MAP test metric
            Ys = tf.tile(Y[None, :, :, None], [n_samples, 1, 1, 1])  # this is inefficient, but can't get the shapes to play well with cholesky_solve otherwise
            L_solve_y = tf.linalg.triangular_solve(L, Ys, lower=True)  # (R, N, D, 1)
            yt_inv_y = tf.reduce_sum(L_solve_y**2, axis=(2, 3))  # (R, N)

        return log_det_cov, yt_inv_y
